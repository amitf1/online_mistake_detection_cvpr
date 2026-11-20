"""Slurm preemption / crash checkpoints that sit beside the existing epoch saves."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..models.checkpoint import write_trainer_checkpoint


LATEST_POINTER_NAME = "latest.json"
DEFAULT_CLUSTER_SAVE_INTERVAL_SECONDS = 4 * 60 * 60
DEFAULT_CLUSTER_KEEP_CHECKPOINTS = 2


def _env_path(name: str) -> str | None:
    value = os.environ.get(name, "").strip()
    return value or None


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return int(raw)


def is_global_rank_zero() -> bool:
    return int(os.environ.get("RANK", "0") or 0) == 0


def atomic_write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    return path


def read_latest_pointer(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def resolve_resume_path(resume: str | Path) -> Path:
    """Accept a checkpoint directory or a cluster ``latest.json`` pointer."""
    raw = Path(resume).expanduser()
    if raw.is_dir():
        return raw.resolve()
    if not raw.is_file():
        raise FileNotFoundError(f"--resume path not found: {raw}")
    if raw.suffix != ".json":
        raise ValueError(f"--resume file must be a directory or latest.json, got {raw}")
    payload = read_latest_pointer(raw)
    checkpoint = payload.get("checkpoint_dir") or payload.get("path") or payload.get("resume")
    if not checkpoint:
        raise ValueError(f"{raw} has no checkpoint_dir/path/resume field")
    resolved = Path(str(checkpoint)).expanduser()
    if not resolved.is_absolute():
        resolved = raw.parent / resolved
    if not resolved.is_dir():
        raise FileNotFoundError(f"{raw} points at missing checkpoint {resolved}")
    return resolved.resolve()


def periodic_save_due(
    last_monotonic: float | None,
    *,
    interval_seconds: int,
    now: float,
) -> bool:
    if interval_seconds <= 0 or last_monotonic is None:
        return False
    return (now - last_monotonic) >= interval_seconds


def prune_cluster_checkpoints(ckpt_dir: Path, keep: int) -> None:
    if keep < 1:
        keep = 1
    candidates = sorted(
        (
            path
            for path in ckpt_dir.glob("step_*")
            if path.is_dir() and not path.name.endswith(".tmp")
        ),
        key=lambda path: path.stat().st_mtime,
    )
    for stale in candidates[:-keep]:
        shutil.rmtree(stale, ignore_errors=True)


def add_preemption_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--ckpt-dir",
        default=_env_path("CKPT_DIR"),
        help="Shared filesystem directory for preemption/crash checkpoints. "
        "Defaults to $CKPT_DIR. Epoch/best saves still go to --output-dir.",
    )
    parser.add_argument(
        "--resume",
        default=_env_path("RESUME"),
        help="HF checkpoint directory or $CKPT_DIR/latest.json pointer. "
        "Overrides --resume-from-checkpoint when both are set.",
    )
    parser.add_argument(
        "--initial-resume-from-checkpoint",
        default=_env_path("INITIAL_RESUME_FROM_CHECKPOINT"),
        help=(
            "Checkpoint used only for a fresh Slurm submission. Requeued jobs ignore "
            "it and resume from $CKPT_DIR/latest.json instead."
        ),
    )
    parser.add_argument(
        "--cluster-save-interval-seconds",
        type=int,
        default=_env_int("CLUSTER_SAVE_INTERVAL_SECONDS", DEFAULT_CLUSTER_SAVE_INTERVAL_SECONDS),
        help="Periodic cluster checkpoint interval. Cluster contract is <= 4 hours.",
    )
    parser.add_argument(
        "--cluster-keep-checkpoints",
        type=int,
        default=_env_int("CLUSTER_KEEP_CHECKPOINTS", DEFAULT_CLUSTER_KEEP_CHECKPOINTS),
        help="How many $CKPT_DIR/step_* directories to retain.",
    )


def checkpoint_global_step(path: Path) -> int | None:
    """Read ``trainer_state.json`` from a Hugging Face checkpoint directory."""
    state_path = path / "trainer_state.json"
    if not state_path.is_file():
        return None
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    try:
        return int(payload.get("global_step"))
    except (TypeError, ValueError):
        return None


def _has_model_weights(path: Path) -> bool:
    return (path / "adapter_config.json").is_file() or (path / "config.json").is_file()


def iter_saved_checkpoints(
    output_dir: str | Path | None,
    ckpt_dir: str | Path | None,
) -> list[Path]:
    """Epoch saves and cluster step saves that can continue a run."""
    found: list[Path] = []
    if output_dir:
        root = Path(output_dir).expanduser()
        if root.is_dir():
            named = [root / "latest"]
            named.extend(path for path in root.glob("ckpt_ep*") if path.is_dir())
            found.extend(path for path in named if path.is_dir())
    if ckpt_dir:
        root = Path(ckpt_dir).expanduser()
        pointer = root / LATEST_POINTER_NAME
        if pointer.is_file():
            try:
                found.append(resolve_resume_path(pointer))
            except (FileNotFoundError, ValueError, json.JSONDecodeError, OSError):
                pass
        if root.is_dir():
            found.extend(
                path
                for path in root.glob("step_*")
                if path.is_dir() and not path.name.endswith(".tmp")
            )
    unique: dict[Path, Path] = {}
    for path in found:
        try:
            unique[path.resolve()] = path.resolve()
        except OSError:
            continue
    return list(unique.values())


def prefer_furthest_checkpoint(selected: Path, candidates: list[Path]) -> Path:
    """Keep training from the save with the largest ``global_step``."""
    selected_step = checkpoint_global_step(selected)
    if selected_step is None:
        return selected
    best = selected
    best_step = selected_step
    for candidate in candidates:
        if not _has_model_weights(candidate):
            continue
        step = checkpoint_global_step(candidate)
        if step is None or step <= best_step:
            continue
        best = candidate
        best_step = step
    if best != selected:
        print(
            f"[preemption] {selected} is global_step {selected_step}; "
            f"continuing from {best} at global_step {best_step}",
            flush=True,
        )
    return best


def apply_resume_checkpoint(args: argparse.Namespace) -> str | None:
    """Resolve --resume / --resume-from-checkpoint / Slurm requeue into an HF dir."""
    explicit = getattr(args, "resume", None) or getattr(args, "resume_from_checkpoint", None)
    ckpt_dir = getattr(args, "ckpt_dir", None) or _env_path("CKPT_DIR")
    restart_count = int(os.environ.get("SLURM_RESTART_COUNT", "0") or 0)
    initial_resume = getattr(args, "initial_resume_from_checkpoint", None)
    if not explicit and restart_count == 0 and initial_resume:
        explicit = initial_resume
        print(
            f"[preemption] fresh submission: initial resume from {initial_resume}",
            flush=True,
        )
    if not explicit and ckpt_dir and restart_count > 0:
        pointer = Path(ckpt_dir) / LATEST_POINTER_NAME
        if pointer.is_file():
            explicit = str(pointer)
            print(
                f"[preemption] SLURM restart #{restart_count}: resuming from {pointer}",
                flush=True,
            )
    if not explicit:
        args.resume_from_checkpoint = None
        return None

    raw = Path(str(explicit)).expanduser()
    try:
        resume_path = resolve_resume_path(raw)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"--resume failed: {exc}") from exc
    resume_path = prefer_furthest_checkpoint(
        resume_path,
        iter_saved_checkpoints(
            getattr(args, "output_dir", None),
            ckpt_dir,
        ),
    )

    has_adapter = (resume_path / "adapter_config.json").exists() or (
        resume_path / "config.json"
    ).exists()
    if not has_adapter:
        raise SystemExit(
            f"--resume={resume_path} has no adapter_config.json/config.json"
        )
    if not (resume_path / "optimizer.pt").exists():
        print(
            "WARNING: resume checkpoint has no optimizer.pt / scheduler.pt.\n"
            "  LoRA weights and trainer_state (step/epoch) will load, but Adam + LR "
            "schedule restart from scratch.",
            flush=True,
        )
    args.resume_from_checkpoint = str(resume_path)
    print(f"[preemption] resume_from_checkpoint={resume_path}", flush=True)
    return str(resume_path)


class ClusterCheckpointManager:
    def __init__(
        self,
        *,
        ckpt_dir: str | Path,
        save_interval_seconds: int = DEFAULT_CLUSTER_SAVE_INTERVAL_SECONDS,
        keep_checkpoints: int = DEFAULT_CLUSTER_KEEP_CHECKPOINTS,
    ) -> None:
        self.ckpt_dir = Path(ckpt_dir).expanduser().resolve()
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.save_interval_seconds = int(save_interval_seconds)
        self.keep_checkpoints = max(1, int(keep_checkpoints))
        self.trainer: Any | None = None
        self._saving = False
        self._preempt = False
        self._last_save_monotonic = time.monotonic()

    @property
    def pointer_path(self) -> Path:
        return self.ckpt_dir / LATEST_POINTER_NAME

    def bind_trainer(self, trainer: Any) -> None:
        self.trainer = trainer

    def request_preempt(self) -> None:
        self._preempt = True

    def save(self, *, reason: str) -> Path | None:
        if not is_global_rank_zero():
            return None
        if self.trainer is None:
            raise RuntimeError("ClusterCheckpointManager.trainer must be assigned before save.")
        if self._saving:
            print(f"[preemption] save already in progress; skipping reason={reason}", flush=True)
            return None
        self._saving = True
        try:
            step = int(getattr(self.trainer.state, "global_step", 0) or 0)
            epoch = getattr(self.trainer.state, "epoch", None)
            dest = self.ckpt_dir / f"step_{step}"
            tmp = self.ckpt_dir / f"step_{step}.tmp"
            if tmp.exists():
                shutil.rmtree(tmp)
            tmp.mkdir(parents=True, exist_ok=True)
            write_trainer_checkpoint(self.trainer, tmp)
            old = dest.with_name(f"{dest.name}.old")
            if old.exists():
                shutil.rmtree(old)
            if dest.exists():
                os.replace(dest, old)
            os.replace(tmp, dest)
            if old.exists():
                shutil.rmtree(old)
            atomic_write_json(
                self.pointer_path,
                {
                    "checkpoint_dir": str(dest),
                    "epoch": epoch,
                    "global_step": step,
                    "reason": reason,
                    "saved_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            prune_cluster_checkpoints(self.ckpt_dir, self.keep_checkpoints)
            self._last_save_monotonic = time.monotonic()
            print(
                f"[preemption] {reason} checkpoint at step={step} -> {self.pointer_path}",
                flush=True,
            )
            return dest
        finally:
            self._saving = False

    def wait_for_pointer(self, *, reason: str, timeout_seconds: float = 900.0) -> None:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if self.pointer_path.is_file():
                try:
                    payload = read_latest_pointer(self.pointer_path)
                except (OSError, json.JSONDecodeError):
                    payload = {}
                if payload.get("reason") == reason:
                    return
            time.sleep(1.0)

    def save_and_exit(self, signum: int, frame: Any) -> None:
        self.request_preempt()
        step = "?"
        if self.trainer is not None:
            step = getattr(self.trainer.state, "global_step", "?")
        print(f"[preemption] SIGTERM checkpoint at step={step}", flush=True)
        if is_global_rank_zero():
            try:
                self.save(reason="sigterm")
            except Exception as exc:
                print(f"[preemption] SIGTERM save failed: {exc}", flush=True)
                sys.exit(1)
            sys.exit(0)
        self.wait_for_pointer(reason="sigterm")
        sys.exit(0)

    def install_sigterm_handler(self) -> None:
        signal.signal(signal.SIGTERM, self.save_and_exit)
        print(
            f"[preemption] SIGTERM handler installed; ckpt_dir={self.ckpt_dir} "
            f"interval={self.save_interval_seconds}s",
            flush=True,
        )


class ClusterCheckpointCallback:
    def __init__(self, manager: ClusterCheckpointManager) -> None:
        from transformers import TrainerCallback

        self._callback_base = TrainerCallback()
        self.manager = manager

    def __getattr__(self, name: str) -> Any:
        return getattr(self._callback_base, name)

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        if self.manager._preempt:
            control.should_training_stop = True
            if state.is_world_process_zero:
                self.manager.save(reason="sigterm")
            return control
        if not state.is_world_process_zero:
            return control
        if periodic_save_due(
            self.manager._last_save_monotonic,
            interval_seconds=self.manager.save_interval_seconds,
            now=time.monotonic(),
        ):
            self.manager.save(reason="periodic")
        return control


def attach_cluster_checkpointing(trainer: Any, args: argparse.Namespace) -> ClusterCheckpointCallback | None:
    ckpt_dir = getattr(args, "ckpt_dir", None) or _env_path("CKPT_DIR")
    if not ckpt_dir:
        return None
    interval = int(
        getattr(args, "cluster_save_interval_seconds", DEFAULT_CLUSTER_SAVE_INTERVAL_SECONDS)
    )
    keep = int(getattr(args, "cluster_keep_checkpoints", DEFAULT_CLUSTER_KEEP_CHECKPOINTS))
    manager = ClusterCheckpointManager(
        ckpt_dir=ckpt_dir,
        save_interval_seconds=interval,
        keep_checkpoints=keep,
    )
    manager.bind_trainer(trainer)
    callback = ClusterCheckpointCallback(manager)
    trainer.add_callback(callback)
    manager.install_sigterm_handler()
    return callback
