from __future__ import annotations

import fcntl
import json
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping


def shard_output_dir(output_dir: str | Path, rank: int, world_size: int) -> Path:
    root = Path(output_dir)
    if world_size <= 1:
        return root
    shard_name = f"rank{rank:02d}-of-{world_size:02d}"
    if root.name == shard_name and root.parent.name == "shards":
        return root
    return root / "shards" / shard_name


def assign_videos_to_shards(
    video_ids: Iterable[str],
    *,
    world_size: int,
    window_counts: Mapping[str, int] | None = None,
) -> list[list[str]]:
    """Partition videos across ranks."""
    ordered = sorted({str(video_id) for video_id in video_ids})
    if world_size <= 1:
        return [ordered]
    if world_size < 1:
        raise ValueError("world_size must be >= 1")
    counts = window_counts or {}
    shards: list[list[str]] = [[] for _ in range(world_size)]
    loads = [0] * world_size
    ranked = sorted(ordered, key=lambda video_id: (-int(counts.get(video_id, 0)), video_id))
    for video_id in ranked:
        target = min(range(world_size), key=lambda index: (loads[index], index))
        shards[target].append(video_id)
        loads[target] += int(counts.get(video_id, 0))
    return [sorted(shard) for shard in shards]


def shard_video_ids(
    video_ids: Iterable[str],
    *,
    rank: int,
    world_size: int,
    window_counts: Mapping[str, int] | None = None,
) -> list[str]:
    if world_size < 1:
        raise ValueError("world_size must be >= 1")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
    return assign_videos_to_shards(
        video_ids,
        world_size=world_size,
        window_counts=window_counts,
    )[rank]


def window_counts_by_video(windows: Iterable[Any]) -> dict[str, int]:
    return dict(Counter(str(window.video_id) for window in windows))


@contextmanager
def exclusive_file_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def ordered_queue_video_ids(
    video_ids: Iterable[str],
    window_counts: Mapping[str, int] | None = None,
) -> list[str]:
    counts = window_counts or {}
    return sorted({str(video_id) for video_id in video_ids}, key=lambda video_id: (-int(counts.get(video_id, 0)), video_id))


def ensure_video_queue(
    queue_dir: str | Path,
    video_ids: Iterable[str],
    window_counts: Mapping[str, int] | None = None,
) -> Path:
    root = Path(queue_dir)
    root.mkdir(parents=True, exist_ok=True)
    remaining_path = root / "remaining.json"
    ready_path = root / "READY"
    with exclusive_file_lock(root / "queue.lock"):
        if ready_path.is_file() and remaining_path.is_file():
            return remaining_path
        remaining_path.write_text(
            json.dumps(ordered_queue_video_ids(video_ids, window_counts), indent=2) + "\n",
            encoding="utf-8",
        )
        ready_path.write_text("ok\n", encoding="utf-8")
    return remaining_path


def claim_next_video(
    queue_dir: str | Path,
    *,
    rank: int,
    remaining_name: str = "remaining.json",
    claims_name: str = "claims.jsonl",
) -> str | None:
    root = Path(queue_dir)
    remaining_path = root / remaining_name
    claims_path = root / claims_name
    with exclusive_file_lock(root / "queue.lock"):
        if not remaining_path.is_file():
            return None
        remaining = json.loads(remaining_path.read_text(encoding="utf-8"))
        if not remaining:
            return None
        video_id = str(remaining.pop(0))
        remaining_path.write_text(json.dumps(remaining, indent=2) + "\n", encoding="utf-8")
        with open(claims_path, "a", encoding="utf-8") as claims:
            claims.write(
                json.dumps(
                    {
                        "rank": rank,
                        "video_id": video_id,
                        "queue": remaining_name,
                        "claimed_at": datetime.now(timezone.utc).isoformat(),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    return video_id


def mark_shard_done(shard_dir: str | Path) -> Path:
    path = Path(shard_dir) / "DONE"
    path.write_text("ok\n", encoding="utf-8")
    return path


def completed_shard_dirs(output_dir: str | Path, world_size: int) -> list[Path]:
    root = Path(output_dir)
    completed: list[Path] = []
    for rank in range(world_size):
        shard_dir = shard_output_dir(root, rank, world_size)
        if (shard_dir / "DONE").is_file() and (
            (shard_dir / "pipeline_artifacts.json").is_file()
            or (shard_dir / "pipeline_metrics.json").is_file()
        ):
            completed.append(shard_dir)
    return completed


def try_acquire_merge_lock(output_dir: str | Path) -> bool:
    lock_dir = Path(output_dir) / "work_queue" / "merge.lockdir"
    lock_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        lock_dir.mkdir()
    except FileExistsError:
        return False
    return True


def discover_shard_artifacts(output_dir: str | Path) -> list[Path]:
    root = Path(output_dir)
    artifacts = sorted(root.glob("shards/rank*-of-*/pipeline_artifacts.json"))
    if not artifacts:
        artifacts = sorted(root.glob("shards/rank*-of-*/pipeline_metrics.json"))
    if not artifacts:
        raise FileNotFoundError(f"No shard artifacts under {root / 'shards'}")
    return artifacts


def _event_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("video_id") or ""),
        float(row.get("trigger_time") or 0.0),
        int(row.get("step_index") or 0),
        float(row.get("accumulation_start") or 0.0),
        float(row.get("accumulation_end") or 0.0),
    )


def _gt_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("video_id") or ""),
        int(row.get("step_index") or 0),
        float(row.get("gt_start") or 0.0),
        float(row.get("gt_end") or 0.0),
        str(row.get("occurrence_id") or ""),
    )


def _module_a_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    window = row.get("window") or {}
    return (
        str(window.get("video_id") or ""),
        float(window.get("window_end") or 0.0),
        int(window.get("step_index") or 0),
        float(window.get("window_start") or 0.0),
    )


def _raise_if_overlap(items: list[tuple[str, tuple[Any, ...]]]) -> None:
    seen: dict[tuple[Any, ...], str] = {}
    for shard_label, key in items:
        previous = seen.get(key)
        if previous is not None:
            raise ValueError(f"Overlapping shard records for {key}: {previous} and {shard_label}")
        seen[key] = shard_label


def merge_pipeline_artifacts(
    payloads: list[dict[str, Any]],
    *,
    shard_paths: list[str] | list[Path] | None = None,
) -> dict[str, Any]:
    if not payloads:
        raise ValueError("No shard payloads to merge")
    schema = payloads[0].get("schema_version")
    worlds = {
        int(
            (payload.get("video_shard") or {}).get("world_size")
            or (payload.get("args") or {}).get("video_shard_world_size")
            or 1
        )
        for payload in payloads
    }
    if len(worlds) != 1:
        raise ValueError(f"Shards disagree on world_size: {sorted(worlds)}")
    world_size = worlds.pop()
    ranks: list[int] = []
    events: list[dict[str, Any]] = []
    ground_truth: list[dict[str, Any]] = []
    module_a_predictions: list[dict[str, Any]] = []
    shard_video_ids: dict[str, list[str]] = {}
    event_keys: list[tuple[str, tuple[Any, ...]]] = []
    gt_keys: list[tuple[str, tuple[Any, ...]]] = []
    module_a_keys: list[tuple[str, tuple[Any, ...]]] = []

    for payload in payloads:
        shard = payload.get("video_shard") or {}
        args = payload.get("args") or {}
        rank = int(shard.get("rank", args.get("video_shard_rank", 0)))
        ranks.append(rank)
        label = f"rank{rank}"
        videos = [str(video_id) for video_id in shard.get("video_ids") or []]
        shard_video_ids[str(rank)] = videos
        for row in payload.get("events") or []:
            events.append(row)
            event_keys.append((label, _event_sort_key(row)))
        for row in payload.get("ground_truth") or []:
            ground_truth.append(row)
            gt_keys.append((label, _gt_sort_key(row)))
        for row in payload.get("module_a_predictions") or []:
            module_a_predictions.append(row)
            module_a_keys.append((label, _module_a_sort_key(row)))

    expected = set(range(world_size)) if world_size > 1 else {0}
    if set(ranks) != expected:
        raise ValueError(f"Expected shard ranks {sorted(expected)}, got {sorted(ranks)}")
    _raise_if_overlap(event_keys)
    _raise_if_overlap(gt_keys)
    _raise_if_overlap(module_a_keys)

    events.sort(key=_event_sort_key)
    ground_truth.sort(key=_gt_sort_key)
    module_a_predictions.sort(key=_module_a_sort_key)

    merged_args = dict(payloads[0].get("args") or {})
    merged_args["video_shard_rank"] = 0
    merged_args["video_shard_world_size"] = 1
    if shard_paths:
        first_shard_dir = Path(shard_paths[0]).parent
        if first_shard_dir.parent.name == "shards":
            merged_args["output_dir"] = str(first_shard_dir.parent.parent)

    eligible = sum(1 for row in ground_truth if row.get("eligible"))
    return {
        "schema_version": schema or "online_pipeline_artifacts_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "args": merged_args,
        "split": payloads[0].get("split") or {},
        "checkpoint_provenance": payloads[0].get("checkpoint_provenance") or {},
        "metrics": {},
        "events": events,
        "ground_truth": ground_truth,
        "module_a_predictions": module_a_predictions,
        "artifact_counts": {
            "pipeline_events": len(events),
            "ground_truth_events": len(ground_truth),
            "eligible_ground_truth_events": eligible,
            "module_a_predictions": len(module_a_predictions),
        },
        "video_shard": {
            "merged": True,
            "world_size": world_size,
            "rank": None,
            "video_ids": sorted({video_id for videos in shard_video_ids.values() for video_id in videos}),
            "shard_video_ids": {key: shard_video_ids[key] for key in sorted(shard_video_ids, key=int)},
            "shard_paths": [str(path) for path in (shard_paths or [])],
        },
    }
