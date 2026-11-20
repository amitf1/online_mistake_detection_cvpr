import json
import os
import subprocess
import sys
from pathlib import Path

from evaluate.abc_monolith.merge import merge_shard_artifacts
from online_mistake_detection.metrics.abc_monolith import (
    build_monolith_artifact,
    monolith_runtime_metrics,
)


def test_merge_script_can_import_the_evaluate_package() -> None:
    repo = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo / 'src'}"
    completed = subprocess.run(
        [sys.executable, str(repo / "evaluate/abc_monolith/merge.py"), "--help"],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout


def test_artifact_round_trip_and_runtime_cost() -> None:
    polls = [
        {
            "monolith_prediction_seconds": 2.0,
            "parse_valid": True,
            "completed": False,
        },
        {
            "monolith_prediction_seconds": 4.0,
            "parse_valid": True,
            "completed": True,
        },
    ]
    runtime = monolith_runtime_metrics(
        polls, streamed_video_seconds=60.0, gpu_count=2
    )
    assert runtime["prediction_runtime/monolith/mean_seconds"] == 3.0
    assert (
        runtime[
            "prediction_runtime/monolith/gpu_seconds_per_streamed_video_minute"
        ]
        == 12.0
    )
    artifact = build_monolith_artifact(
        polls=polls,
        events=[],
        ground_truth=[],
        metrics=runtime,
    )
    assert json.loads(json.dumps(artifact))["schema_version"] == "egoper_abc_monolith_v1"


def test_distributed_shards_merge_into_one_test_artifact() -> None:
    artifacts = []
    for rank in range(2):
        artifacts.append(
            build_monolith_artifact(
                polls=[],
                events=[],
                ground_truth=[],
                metrics={},
                metadata={
                    "checkpoint": "/checkpoint/best_ep6",
                    "split": "test",
                    "candidate_scoring": True,
                    "video_shard_rank": rank,
                    "video_shard_world_size": 2,
                    "streamed_video_seconds": 10.0,
                    "one_pass_generation_seconds": 2.0,
                    "score_pass_seconds": 3.0,
                    "end_to_end_one_pass_seconds": 2.5,
                },
                split={"name": "test", "video_ids": [f"video-{rank}"]},
            )
        )

    merged = merge_shard_artifacts(
        artifacts,
        require_complete_manifest=False,
        compute_bertscore=False,
    )

    assert merged["split"]["video_ids"] == ["video-0", "video-1"]
    assert merged["metadata"]["video_shard_world_size"] == 2
    assert merged["metadata"]["streamed_video_seconds"] == 20.0
    assert merged["metadata"]["one_pass_generation_seconds"] == 4.0
    assert merged["metadata"]["score_pass_seconds"] == 6.0

    deferred = merge_shard_artifacts(
        artifacts,
        require_complete_manifest=False,
        compute_bertscore=False,
        defer_metrics=True,
    )
    assert deferred["metrics"] == {}
    assert deferred["metadata"]["metrics_deferred"] is True


def test_queue_video_merge_does_not_require_every_rank() -> None:
    artifacts = [
        build_monolith_artifact(
            polls=[],
            events=[],
            ground_truth=[],
            metrics={},
            metadata={
                "checkpoint": "/checkpoint/best_ep6",
                "split": "test",
                "candidate_scoring": True,
                "video_shard_rank": 0,
                "video_shard_world_size": 2,
                "queue_mode": True,
                "streamed_video_seconds": 10.0,
                "one_pass_generation_seconds": 2.0,
                "score_pass_seconds": 3.0,
                "end_to_end_one_pass_seconds": 2.5,
            },
            split={"name": "test", "video_ids": [f"video-{index}"]},
        )
        for index in range(2)
    ]

    merged = merge_shard_artifacts(
        artifacts,
        require_complete_manifest=False,
        compute_bertscore=False,
        defer_metrics=True,
        require_all_ranks=False,
        expected_video_ids=["video-0", "video-1"],
    )
    assert merged["split"]["video_ids"] == ["video-0", "video-1"]
    assert merged["metadata"]["queue_mode"] is True
    assert merged["metadata"]["history_mode"] == "predicted"


def test_deferred_merge_allows_shards_without_candidate_scoring() -> None:
    artifacts = []
    for rank in range(2):
        artifacts.append(
            build_monolith_artifact(
                polls=[],
                events=[],
                ground_truth=[],
                metrics={},
                metadata={
                    "checkpoint": "/checkpoint/best_ep6",
                    "split": "test",
                    "candidate_scoring": False,
                    "video_shard_rank": rank,
                    "video_shard_world_size": 2,
                    "history_mode": "none",
                    "streamed_video_seconds": 1.0,
                    "one_pass_generation_seconds": 1.0,
                    "score_pass_seconds": 0.0,
                    "end_to_end_one_pass_seconds": 1.0,
                },
                split={"name": "test", "video_ids": [f"video-{rank}"]},
            )
        )
    merged = merge_shard_artifacts(
        artifacts,
        require_complete_manifest=True,
        compute_bertscore=False,
        defer_metrics=True,
    )
    assert merged["metadata"]["metrics_deferred"] is True
    assert merged["metadata"]["candidate_scoring"] is False
    assert merged["metrics"] == {}

    try:
        merge_shard_artifacts(
            artifacts,
            require_complete_manifest=True,
            compute_bertscore=False,
            defer_metrics=False,
        )
    except ValueError as exc:
        assert "candidate scoring" in str(exc)
    else:
        raise AssertionError("scored manifests still require candidate NLLs")


def test_merge_preserves_one_history_mode_and_rejects_a_mixture() -> None:
    def artifact(mode: str, rank: int) -> dict:
        return build_monolith_artifact(
            polls=[],
            events=[],
            ground_truth=[],
            metrics={},
            metadata={
                "checkpoint": "/checkpoint/best_ep6",
                "split": "test",
                "candidate_scoring": False,
                "video_shard_rank": rank,
                "video_shard_world_size": 2,
                "evaluation_view": f"view-{mode}",
                "history_source": mode,
                "history_mode": mode,
                "streamed_video_seconds": 1.0,
                "one_pass_generation_seconds": 1.0,
                "score_pass_seconds": 0.0,
                "end_to_end_one_pass_seconds": 1.0,
            },
            split={"name": "test", "video_ids": [f"video-{rank}"]},
        )

    merged = merge_shard_artifacts(
        [artifact("gold", 0), artifact("gold", 1)],
        require_complete_manifest=False,
        compute_bertscore=False,
        defer_metrics=True,
    )
    assert merged["metadata"]["history_mode"] == "gold"
    assert merged["metadata"]["evaluation_view"] == "view-gold"

    try:
        merge_shard_artifacts(
            [artifact("gold", 0), artifact("none", 1)],
            require_complete_manifest=False,
            compute_bertscore=False,
            defer_metrics=True,
        )
    except ValueError as exc:
        assert "disagrees" in str(exc)
    else:
        raise AssertionError("mixed history modes must fail")

