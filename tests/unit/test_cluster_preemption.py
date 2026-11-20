from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

from online_mistake_detection.training.preemption import (
    LATEST_POINTER_NAME,
    add_preemption_args,
    apply_resume_checkpoint,
    atomic_write_json,
    periodic_save_due,
    prune_cluster_checkpoints,
    resolve_resume_path,
)


def _touch_adapter(path: Path, *, optimizer: bool = True) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    (path / "adapter_model.safetensors").write_bytes(b"x")
    if optimizer:
        (path / "optimizer.pt").write_bytes(b"opt")
    return path


def test_atomic_write_json_replaces_pointer(tmp_path: Path) -> None:
    pointer = tmp_path / LATEST_POINTER_NAME
    atomic_write_json(pointer, {"checkpoint_dir": "/first", "global_step": 1})
    atomic_write_json(pointer, {"checkpoint_dir": "/second", "global_step": 2})
    payload = json.loads(pointer.read_text(encoding="utf-8"))
    assert payload["checkpoint_dir"] == "/second"
    assert payload["global_step"] == 2
    assert not pointer.with_name("latest.json.tmp").exists()


def test_resolve_resume_path_from_latest_json(tmp_path: Path) -> None:
    checkpoint = _touch_adapter(tmp_path / "step_12")
    pointer = tmp_path / LATEST_POINTER_NAME
    atomic_write_json(pointer, {"checkpoint_dir": str(checkpoint), "global_step": 12})
    assert resolve_resume_path(pointer) == checkpoint.resolve()


def test_resolve_resume_path_relative_checkpoint_dir(tmp_path: Path) -> None:
    checkpoint = _touch_adapter(tmp_path / "step_4")
    pointer = tmp_path / LATEST_POINTER_NAME
    atomic_write_json(pointer, {"checkpoint_dir": "step_4", "global_step": 4})
    assert resolve_resume_path(pointer) == checkpoint.resolve()


def test_resolve_resume_path_accepts_directory(tmp_path: Path) -> None:
    checkpoint = _touch_adapter(tmp_path / "latest")
    assert resolve_resume_path(checkpoint) == checkpoint.resolve()


def test_apply_resume_prefers_resume_over_resume_from_checkpoint(tmp_path: Path) -> None:
    wanted = _touch_adapter(tmp_path / "step_9")
    other = _touch_adapter(tmp_path / "other")
    pointer = tmp_path / LATEST_POINTER_NAME
    atomic_write_json(pointer, {"checkpoint_dir": str(wanted)})
    args = SimpleNamespace(resume=str(pointer), resume_from_checkpoint=str(other), ckpt_dir=str(tmp_path))
    assert apply_resume_checkpoint(args) == str(wanted.resolve())
    assert args.resume_from_checkpoint == str(wanted.resolve())


def test_apply_resume_auto_on_slurm_restart(tmp_path: Path, monkeypatch) -> None:
    checkpoint = _touch_adapter(tmp_path / "step_3")
    atomic_write_json(tmp_path / LATEST_POINTER_NAME, {"checkpoint_dir": str(checkpoint)})
    monkeypatch.setenv("SLURM_RESTART_COUNT", "1")
    args = SimpleNamespace(resume=None, resume_from_checkpoint=None, ckpt_dir=str(tmp_path))
    assert Path(apply_resume_checkpoint(args) or "").resolve() == checkpoint.resolve()


def test_apply_resume_uses_initial_checkpoint_only_on_fresh_submission(
    tmp_path: Path,
    monkeypatch,
) -> None:
    initial = _touch_adapter(tmp_path / "ckpt_ep1")
    monkeypatch.setenv("SLURM_RESTART_COUNT", "0")
    args = SimpleNamespace(
        resume=None,
        resume_from_checkpoint=None,
        initial_resume_from_checkpoint=str(initial),
        ckpt_dir=str(tmp_path / "cluster"),
    )
    assert apply_resume_checkpoint(args) == str(initial.resolve())


def test_apply_resume_ignores_initial_checkpoint_on_requeue(
    tmp_path: Path,
    monkeypatch,
) -> None:
    initial = _touch_adapter(tmp_path / "ckpt_ep1")
    latest = _touch_adapter(tmp_path / "cluster" / "step_400")
    atomic_write_json(
        tmp_path / "cluster" / LATEST_POINTER_NAME,
        {"checkpoint_dir": str(latest)},
    )
    monkeypatch.setenv("SLURM_RESTART_COUNT", "1")
    args = SimpleNamespace(
        resume=None,
        resume_from_checkpoint=None,
        initial_resume_from_checkpoint=str(initial),
        ckpt_dir=str(tmp_path / "cluster"),
    )
    assert apply_resume_checkpoint(args) == str(latest.resolve())


def _write_step(path: Path, step: int) -> Path:
    _touch_adapter(path)
    (path / "trainer_state.json").write_text(
        json.dumps({"global_step": step, "epoch": step / 37}) + "\n",
        encoding="utf-8",
    )
    return path


def test_resume_prefers_later_epoch_checkpoint_over_stale_pointer(tmp_path: Path) -> None:
    stale = _write_step(tmp_path / "cluster" / "step_37", 37)
    epoch3 = _write_step(tmp_path / "run" / "ckpt_ep3", 111)
    _write_step(tmp_path / "run" / "latest", 74)
    pointer = tmp_path / "cluster" / LATEST_POINTER_NAME
    atomic_write_json(pointer, {"checkpoint_dir": str(stale), "global_step": 37})
    args = SimpleNamespace(
        resume=str(pointer),
        resume_from_checkpoint=None,
        ckpt_dir=str(tmp_path / "cluster"),
        output_dir=str(tmp_path / "run"),
    )
    assert apply_resume_checkpoint(args) == str(epoch3.resolve())


def test_apply_resume_does_not_auto_on_fresh_start(tmp_path: Path, monkeypatch) -> None:
    checkpoint = _touch_adapter(tmp_path / "step_3")
    atomic_write_json(tmp_path / LATEST_POINTER_NAME, {"checkpoint_dir": str(checkpoint)})
    monkeypatch.delenv("SLURM_RESTART_COUNT", raising=False)
    args = SimpleNamespace(resume=None, resume_from_checkpoint=None, ckpt_dir=str(tmp_path))
    assert apply_resume_checkpoint(args) is None
    assert args.resume_from_checkpoint is None


def test_periodic_save_due_waits_full_interval() -> None:
    assert periodic_save_due(None, interval_seconds=14400, now=100.0) is False
    assert periodic_save_due(0.0, interval_seconds=14400, now=14399.0) is False
    assert periodic_save_due(0.0, interval_seconds=14400, now=14400.0) is True
    assert periodic_save_due(0.0, interval_seconds=0, now=10_000.0) is False


def test_prune_cluster_checkpoints_keeps_newest(tmp_path: Path) -> None:
    older = tmp_path / "step_1"
    newer = tmp_path / "step_2"
    extra = tmp_path / "step_3"
    for path in (older, newer, extra):
        path.mkdir()
        (path / "marker").write_text(path.name, encoding="utf-8")
    os.utime(older, (1, 1))
    os.utime(newer, (2, 2))
    os.utime(extra, (3, 3))
    prune_cluster_checkpoints(tmp_path, keep=2)
    assert not older.exists()
    assert newer.exists()
    assert extra.exists()


def test_cluster_manager_writes_latest_json(tmp_path: Path, monkeypatch) -> None:
    import torch

    from online_mistake_detection.training.preemption import ClusterCheckpointManager

    class FakeState:
        global_step = 7
        epoch = 1.5

        def save_to_json(self, path: str) -> None:
            Path(path).write_text(json.dumps({"global_step": self.global_step}), encoding="utf-8")

    class FakeTrainer:
        args = SimpleNamespace(world_size=1)
        state = FakeState()
        optimizer = SimpleNamespace(state_dict=lambda: {"ok": True})
        lr_scheduler = SimpleNamespace(state_dict=lambda: {"lr": 1e-4})
        scaler = None

        def save_model(self, path: str) -> None:
            dest = Path(path)
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "adapter_config.json").write_text("{}\n", encoding="utf-8")
            (dest / "adapter_model.safetensors").write_bytes(b"x")

    monkeypatch.setenv("RANK", "0")
    manager = ClusterCheckpointManager(ckpt_dir=tmp_path, save_interval_seconds=14400, keep_checkpoints=2)
    manager.bind_trainer(FakeTrainer())
    dest = manager.save(reason="sigterm")
    assert dest is not None
    pointer = json.loads((tmp_path / LATEST_POINTER_NAME).read_text(encoding="utf-8"))
    assert pointer["global_step"] == 7
    assert pointer["reason"] == "sigterm"
    assert Path(pointer["checkpoint_dir"]).is_dir()
    assert (dest / "optimizer.pt").is_file()
    assert torch.load(dest / "optimizer.pt", weights_only=False) == {"ok": True}
    assert resolve_resume_path(tmp_path / LATEST_POINTER_NAME) == dest.resolve()


def test_add_preemption_args_read_env(monkeypatch) -> None:
    monkeypatch.setenv("CKPT_DIR", "/tmp/ckpts")
    monkeypatch.setenv("CLUSTER_SAVE_INTERVAL_SECONDS", "3600")
    monkeypatch.setenv("INITIAL_RESUME_FROM_CHECKPOINT", "/tmp/ckpt_ep1")
    parser = argparse.ArgumentParser()
    add_preemption_args(parser)
    args = parser.parse_args([])
    assert args.ckpt_dir == "/tmp/ckpts"
    assert args.cluster_save_interval_seconds == 3600
    assert args.initial_resume_from_checkpoint == "/tmp/ckpt_ep1"
    assert args.cluster_keep_checkpoints == 2
