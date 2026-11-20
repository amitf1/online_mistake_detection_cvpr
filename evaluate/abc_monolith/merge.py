#!/usr/bin/env python3
"""Merge distributed monolith shards and compute the final test metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Launching this file directly puts evaluate/abc_monolith/ first on sys.path.
# The sibling evaluate.py then shadows the evaluate package.
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]
sys.path[:] = [
    entry
    for entry in sys.path
    if not entry or Path(entry).resolve() != _SCRIPT_DIR
]
sys.path.insert(0, str(_REPO_ROOT))

from evaluate.abc_monolith.evaluate import evaluate_artifact
from online_mistake_detection.metrics.abc_monolith import build_monolith_artifact
from online_mistake_detection.pipeline.runner import json_safe
from online_mistake_detection.pipeline.video_checkpoints import (
    list_checkpoint_videos,
    load_run_manifest,
)


def _consensus_metadata(
    artifacts: list[dict[str, Any]],
    key: str,
    default: str,
) -> str:
    values = {
        str((artifact.get("metadata") or {}).get(key) or default)
        for artifact in artifacts
    }
    if len(values) != 1:
        raise ValueError(f"Shard metadata {key} disagrees: {sorted(values)}")
    return values.pop()


def _consensus_with_fallback(
    artifacts: list[dict[str, Any]],
    key: str,
    fallback_key: str,
    default: str,
) -> str:
    values: set[str] = set()
    for artifact in artifacts:
        metadata = artifact.get("metadata") or {}
        raw = metadata.get(key)
        if raw is None or raw == "":
            raw = metadata.get(fallback_key) or default
        values.add(str(raw))
    if len(values) != 1:
        raise ValueError(f"Shard metadata {key} disagrees: {sorted(values)}")
    return values.pop()


def discover_video_artifacts(video_dir: str | Path) -> list[Path]:
    root = Path(video_dir)
    return sorted(
        path
        for path in root.glob("*/monolith.json")
        if path.is_file()
    )


def _combine_artifacts(
    artifacts: list[dict[str, Any]],
    *,
    shard_paths: list[Path] | None = None,
    require_all_ranks: bool = True,
    expected_video_ids: list[str] | None = None,
    defer_metrics: bool = False,
) -> dict[str, Any]:
    if not artifacts:
        raise ValueError("At least one artifact is required.")
    world_sizes = {
        int(artifact.get("metadata", {}).get("video_shard_world_size", 1))
        for artifact in artifacts
    }
    ranks = {
        int(artifact.get("metadata", {}).get("video_shard_rank", 0))
        for artifact in artifacts
    }
    if len(world_sizes) != 1:
        raise ValueError(f"Shard world sizes disagree: {sorted(world_sizes)}")
    world_size = world_sizes.pop()
    if require_all_ranks:
        expected_ranks = set(range(world_size))
        if ranks != expected_ranks:
            raise ValueError(
                f"Expected shard ranks {sorted(expected_ranks)}, got {sorted(ranks)}"
            )

    split_names = {
        str(artifact.get("split", {}).get("name")) for artifact in artifacts
    }
    checkpoints = {
        str(artifact.get("metadata", {}).get("checkpoint"))
        for artifact in artifacts
    }
    candidate_scoring = {
        bool(artifact.get("metadata", {}).get("candidate_scoring"))
        for artifact in artifacts
    }
    if len(split_names) != 1 or len(checkpoints) != 1:
        raise ValueError("Shard split/checkpoint metadata disagree.")

    video_owners: dict[str, int] = {}
    for artifact in artifacts:
        rank = int(artifact.get("metadata", {}).get("video_shard_rank", 0))
        for video_id in artifact.get("split", {}).get("video_ids", []):
            if video_id in video_owners:
                raise ValueError(
                    f"Video {video_id!r} appears in ranks "
                    f"{video_owners[video_id]} and {rank}."
                )
            video_owners[str(video_id)] = rank
    if expected_video_ids is not None:
        expected = set(expected_video_ids)
        found = set(video_owners)
        if found != expected:
            missing = sorted(expected - found)
            extra = sorted(found - expected)
            raise ValueError(
                f"Video set mismatch: missing={missing[:5]} extra={extra[:5]}"
            )

    def rows(key: str) -> list[Any]:
        return [
            row
            for artifact in sorted(
                artifacts,
                key=lambda item: (
                    str((item.get("split") or {}).get("video_ids") or [""])[0],
                    int(item.get("metadata", {}).get("video_shard_rank", 0)),
                ),
            )
            for row in artifact.get(key, [])
        ]

    generation_seconds = sum(
        float(artifact.get("metadata", {}).get("one_pass_generation_seconds") or 0)
        for artifact in artifacts
    )
    score_seconds = sum(
        float(artifact.get("metadata", {}).get("score_pass_seconds") or 0)
        for artifact in artifacts
    )
    streamed_seconds = sum(
        float(artifact.get("metadata", {}).get("streamed_video_seconds") or 0)
        for artifact in artifacts
    )
    replica_wall_seconds = [
        float(artifact.get("metadata", {}).get("end_to_end_one_pass_seconds") or 0)
        for artifact in artifacts
    ]
    first = artifacts[0]
    return build_monolith_artifact(
        polls=rows("polls"),
        events=rows("events"),
        ground_truth=rows("ground_truth"),
        metrics={},
        metadata={
            "evaluation_view": _consensus_metadata(
                artifacts,
                "evaluation_view",
                "realistic_predicted_history_sequential",
            ),
            "history_source": _consensus_metadata(
                artifacts,
                "history_source",
                "predicted",
            ),
            "history_mode": _consensus_metadata(
                artifacts,
                "history_mode",
                "predicted",
            ),
            "dataset": _consensus_with_fallback(
                artifacts,
                "dataset",
                "data_dataset",
                "egoper",
            ),
            "data_dataset": _consensus_with_fallback(
                artifacts,
                "data_dataset",
                "dataset",
                "egoper",
            ),
            "prompt_contract": _consensus_with_fallback(
                artifacts,
                "prompt_contract",
                "dataset",
                "egoper",
            ),
            "prompt_version": _consensus_metadata(
                artifacts,
                "prompt_version",
                "abc_monolith_v1",
            ),
            "streamed_video_seconds": streamed_seconds,
            "gpu_count": 1,
            "checkpoint": checkpoints.pop(),
            "split": split_names.pop(),
            "candidate_scoring": candidate_scoring == {True},
            "metrics_deferred": defer_metrics,
            "score_pass_seconds": score_seconds,
            "one_pass_generation_seconds": generation_seconds,
            "end_to_end_one_pass_seconds": generation_seconds,
            "parallel_worker_wall_seconds_max": max(
                replica_wall_seconds, default=0.0
            ),
            "video_shard_world_size": world_size,
            "queue_mode": any(
                artifact.get("metadata", {}).get("queue_mode")
                for artifact in artifacts
            ),
            "merged_shards": (
                [str(path) for path in shard_paths] if shard_paths else world_size
            ),
        },
        module_a_predictions=rows("module_a_predictions"),
        args={
            **dict(first.get("args") or {}),
            "video_shard_world_size": world_size,
        },
        split={
            "name": first.get("split", {}).get("name", "test"),
            "video_ids": sorted(video_owners),
        },
    )


def merge_shard_artifacts(
    artifacts: list[dict[str, Any]],
    *,
    shard_paths: list[Path] | None = None,
    egoper_root: str | Path | None = None,
    require_complete_manifest: bool = True,
    compute_bertscore: bool = True,
    bertscore_model_type: str = "microsoft/deberta-xlarge-mnli",
    bertscore_batch_size: int = 16,
    bertscore_device: str = "auto",
    defer_metrics: bool = False,
    require_all_ranks: bool = True,
    expected_video_ids: list[str] | None = None,
) -> dict[str, Any]:
    if not artifacts:
        raise ValueError("At least one shard artifact is required.")
    candidate_scoring = {
        bool(artifact.get("metadata", {}).get("candidate_scoring"))
        for artifact in artifacts
    }
    if (
        candidate_scoring != {True}
        and require_complete_manifest
        and not defer_metrics
    ):
        raise ValueError(
            "A complete monolith metric manifest requires candidate scoring "
            "on every shard."
        )
    merged = _combine_artifacts(
        artifacts,
        shard_paths=shard_paths,
        require_all_ranks=require_all_ranks,
        expected_video_ids=expected_video_ids,
        defer_metrics=defer_metrics,
    )
    if defer_metrics:
        return merged
    return evaluate_artifact(
        merged,
        egoper_root=egoper_root,
        require_complete_manifest=require_complete_manifest,
        compute_bertscore=compute_bertscore,
        bertscore_model_type=bertscore_model_type,
        bertscore_batch_size=bertscore_batch_size,
        bertscore_device=bertscore_device,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-json", type=Path, nargs="*")
    parser.add_argument("--video-dir", type=Path)
    parser.add_argument("--queue-dir", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--egoper-root", type=Path)
    parser.add_argument(
        "--require-complete-manifest",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--compute-bertscore",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--bertscore-model-type",
        default="microsoft/deberta-xlarge-mnli",
    )
    parser.add_argument("--bertscore-batch-size", type=int, default=16)
    parser.add_argument("--bertscore-device", default="auto")
    parser.add_argument(
        "--defer-metrics",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    args = parser.parse_args()

    expected_video_ids = None
    if args.queue_dir is not None:
        manifest = load_run_manifest(args.queue_dir)
        expected_video_ids = [
            str(video_id) for video_id in manifest.get("eval_video_ids") or []
        ]
        if not expected_video_ids:
            raise ValueError(f"No eval_video_ids in {args.queue_dir / 'manifest.json'}")
        done = set(list_checkpoint_videos(args.video_dir or args.queue_dir.parent / "videos", "monolith.json"))
        missing = sorted(set(expected_video_ids) - done)
        if missing:
            print(
                json.dumps(
                    {
                        "status": "incomplete",
                        "done": len(done),
                        "expected": len(expected_video_ids),
                        "remaining": missing[:20],
                    },
                    sort_keys=True,
                )
            )
            raise SystemExit(2)

    if args.video_dir is not None:
        shard_paths = discover_video_artifacts(args.video_dir)
        require_all_ranks = False
    else:
        shard_paths = list(args.shard_json or [])
        require_all_ranks = True
    if not shard_paths:
        raise ValueError("Provide --shard-json or --video-dir with completed artifacts.")
    artifacts = [
        json.loads(path.read_text(encoding="utf-8")) for path in shard_paths
    ]
    merged = merge_shard_artifacts(
        artifacts,
        shard_paths=shard_paths,
        egoper_root=args.egoper_root,
        require_complete_manifest=args.require_complete_manifest,
        compute_bertscore=args.compute_bertscore,
        bertscore_model_type=args.bertscore_model_type,
        bertscore_batch_size=args.bertscore_batch_size,
        bertscore_device=args.bertscore_device,
        defer_metrics=args.defer_metrics,
        require_all_ranks=require_all_ranks,
        expected_video_ids=expected_video_ids,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(json_safe(merged), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
