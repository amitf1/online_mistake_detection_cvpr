from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .shards import exclusive_file_lock, ordered_queue_video_ids


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as writer:
            writer.write(
                json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
                + "\n"
            )
        tmp_path.replace(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def refresh_remaining_videos(
    queue_dir: str | Path,
    video_ids: Iterable[str],
    *,
    window_counts: Mapping[str, int] | None = None,
    session_id: str,
    remaining_name: str = "remaining.json",
) -> Path:
    """Rewrite the pending list once per job so a resubmit recovers videos that were claimed but never checkpointed."""
    root = Path(queue_dir)
    root.mkdir(parents=True, exist_ok=True)
    remaining_path = root / remaining_name
    session_path = root / f"{remaining_name}.session"
    with exclusive_file_lock(root / "queue.lock"):
        if session_path.is_file() and session_path.read_text(encoding="utf-8").strip() == session_id:
            return remaining_path
        remaining_path.write_text(
            json.dumps(ordered_queue_video_ids(video_ids, window_counts), indent=2) + "\n",
            encoding="utf-8",
        )
        session_path.write_text(f"{session_id}\n", encoding="utf-8")
    return remaining_path


def enqueue_video(
    queue_dir: str | Path,
    video_id: str,
    *,
    remaining_name: str = "bc_remaining.json",
) -> None:
    root = Path(queue_dir)
    remaining_path = root / remaining_name
    with exclusive_file_lock(root / "queue.lock"):
        remaining: list[str] = []
        if remaining_path.is_file():
            remaining = json.loads(remaining_path.read_text(encoding="utf-8"))
        if video_id not in remaining:
            remaining.append(video_id)
            remaining_path.write_text(json.dumps(remaining, indent=2) + "\n", encoding="utf-8")


def video_checkpoint_dir(checkpoint_root: str | Path, video_id: str) -> Path:
    return Path(checkpoint_root) / video_id


def module_a_checkpoint_path(checkpoint_root: str | Path, video_id: str) -> Path:
    return video_checkpoint_dir(checkpoint_root, video_id) / "module_a.json"


def events_checkpoint_path(checkpoint_root: str | Path, video_id: str) -> Path:
    return video_checkpoint_dir(checkpoint_root, video_id) / "events.json"


def causal_ab_events_path(checkpoint_root: str | Path, video_id: str) -> Path:
    return video_checkpoint_dir(checkpoint_root, video_id) / "causal_ab_events.json"


def write_causal_ab_events(
    checkpoint_root: str | Path,
    video_id: str,
    event_rows: list[dict[str, Any]],
) -> None:
    _atomic_write_json(
        causal_ab_events_path(checkpoint_root, video_id),
        {"video_id": video_id, "events": event_rows},
    )


def load_causal_ab_events(checkpoint_root: str | Path, video_id: str) -> list[dict[str, Any]]:
    payload = json.loads(causal_ab_events_path(checkpoint_root, video_id).read_text(encoding="utf-8"))
    return list(payload.get("events") or [])


def ground_truth_checkpoint_path(checkpoint_root: str | Path, video_id: str) -> Path:
    return video_checkpoint_dir(checkpoint_root, video_id) / "ground_truth.json"


def list_checkpoint_videos(checkpoint_root: str | Path, filename: str) -> list[str]:
    root = Path(checkpoint_root)
    if not root.is_dir():
        return []
    return sorted(
        path.parent.name
        for path in root.glob(f"*/{filename}")
        if path.is_file()
    )


def list_module_a_done(checkpoint_root: str | Path) -> list[str]:
    return list_checkpoint_videos(checkpoint_root, "module_a.json")


def list_pipeline_done(checkpoint_root: str | Path) -> list[str]:
    return list_checkpoint_videos(checkpoint_root, "events.json")


def write_module_a_checkpoint(
    checkpoint_root: str | Path,
    video_id: str,
    *,
    module_a_rows: list[dict[str, Any]],
    ground_truth_rows: list[dict[str, Any]],
) -> None:
    _atomic_write_json(
        module_a_checkpoint_path(checkpoint_root, video_id),
        {
            "video_id": video_id,
            "module_a_predictions": module_a_rows,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    _atomic_write_json(
        ground_truth_checkpoint_path(checkpoint_root, video_id),
        {
            "video_id": video_id,
            "ground_truth": ground_truth_rows,
        },
    )


def write_events_checkpoint(
    checkpoint_root: str | Path,
    video_id: str,
    event_rows: list[dict[str, Any]],
) -> None:
    _atomic_write_json(
        events_checkpoint_path(checkpoint_root, video_id),
        {
            "video_id": video_id,
            "events": event_rows,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def load_module_a_rows(checkpoint_root: str | Path, video_id: str) -> list[dict[str, Any]]:
    payload = json.loads(module_a_checkpoint_path(checkpoint_root, video_id).read_text(encoding="utf-8"))
    return list(payload.get("module_a_predictions") or [])


def load_event_rows(checkpoint_root: str | Path, video_id: str) -> list[dict[str, Any]]:
    payload = json.loads(events_checkpoint_path(checkpoint_root, video_id).read_text(encoding="utf-8"))
    return list(payload.get("events") or [])


def load_ground_truth_rows(checkpoint_root: str | Path, video_id: str) -> list[dict[str, Any]]:
    path = ground_truth_checkpoint_path(checkpoint_root, video_id)
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("ground_truth") or [])


def write_progress(
    queue_dir: str | Path,
    *,
    eval_split: str,
    val_video_ids: Iterable[str],
    module_a_done: Iterable[str],
    pipeline_done: Iterable[str],
) -> dict[str, Any]:
    eval_ids = sorted({str(video_id) for video_id in val_video_ids})
    done_a = sorted({str(video_id) for video_id in module_a_done})
    done_pipeline = sorted({str(video_id) for video_id in pipeline_done})
    payload = {
        "eval_split": eval_split,
        "eval_videos": len(eval_ids),
        "module_a_done": len(done_a),
        "pipeline_done": len(done_pipeline),
        "remaining_a": sorted(set(eval_ids) - set(done_a)),
        "remaining_bc": sorted(set(done_a) - set(done_pipeline)),
        "complete": bool(eval_ids) and set(done_pipeline) == set(eval_ids),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_write_json(Path(queue_dir) / "progress.json", payload)
    return payload


def write_run_manifest(
    queue_dir: str | Path,
    *,
    split: dict[str, Any],
    eval_split: str,
    eval_video_ids: Iterable[str],
) -> None:
    root = Path(queue_dir)
    path = root / "manifest.json"
    with exclusive_file_lock(root / "queue.lock"):
        if path.is_file():
            return
        _atomic_write_json(
            path,
            {
                "eval_split": eval_split,
                "eval_video_ids": sorted(eval_video_ids),
                "split": split,
            },
        )


def load_run_manifest(queue_dir: str | Path) -> dict[str, Any]:
    path = Path(queue_dir) / "manifest.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
