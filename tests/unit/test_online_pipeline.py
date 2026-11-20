from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from online_mistake_detection.pipeline.runner import (
    GroundTruthEvent,
    PipelineEvent,
    b_example_from_window,
    compute_metrics,
    match_events,
    module_a_extra_probability,
    write_outputs,
)
from online_mistake_detection.schemas import WindowSpec


def gt(
    *,
    step_index: int = 0,
    start: float = 10.0,
    end: float = 20.0,
    mistake: bool = True,
    video_id: str = "video",
) -> GroundTruthEvent:
    return GroundTruthEvent(
        video_id=video_id,
        task_id="task",
        step_index=step_index,
        instruction=f"step {step_index}",
        gt_start=start,
        gt_end=end,
        gt_mistake=mistake,
        gt_reasoning="reference reasoning",
        eligible=True,
    )


def event(
    *,
    step_index: int = 0,
    start: float = 10.0,
    end: float = 20.0,
    trigger_time: float | None = None,
    module_a_seconds: float | None = None,
    module_b_seconds: float | None = None,
    module_c_seconds: float | None = None,
    mistake: bool | None = True,
    video_id: str = "video",
) -> PipelineEvent:
    pipeline_seconds = None
    if module_a_seconds is not None or module_b_seconds is not None or module_c_seconds is not None:
        pipeline_seconds = sum(
            value
            for value in (module_a_seconds, module_b_seconds, module_c_seconds)
            if value is not None
        )
    return PipelineEvent(
        video_id=video_id,
        task_id="task",
        step_index=step_index,
        instruction=f"step {step_index}",
        trigger_time=end if trigger_time is None else trigger_time,
        accumulation_start=0.0,
        accumulation_end=end,
        module_a_prediction="COMPLETE",
        module_b_prediction=(start, end),
        pred_global_start=start,
        pred_global_end=end,
        module_c_mistake=mistake,
        module_c_reasoning="predicted reasoning" if mistake is not None else None,
        module_a_prediction_seconds=module_a_seconds,
        module_b_prediction_seconds=module_b_seconds,
        module_c_prediction_seconds=module_c_seconds,
        pipeline_prediction_seconds=pipeline_seconds,
    )


def window(label: str = "COMPLETE") -> WindowSpec:
    return WindowSpec(
        video_path=Path("/tmp/fake.mp4"),
        video_id="video",
        task_id="task",
        step_index=0,
        current_step="step 0",
        window_start=0.0,
        window_end=20.0,
        gt_start=10.0,
        gt_end=20.0,
        label=label,
    )


def test_online_module_b_uses_exact_module_a_window_without_extension() -> None:
    trigger_window = window()
    example = b_example_from_window(trigger_window, "step 0")
    assert example.window_start == trigger_window.window_start
    assert example.window_end == trigger_window.window_end
    assert example.metadata["clip_duration"] == 20.0


def test_matching_keeps_best_and_marks_duplicate() -> None:
    gt_events = [gt(start=10.0, end=20.0)]
    events = [
        event(start=9.0, end=20.0),
        event(start=10.0, end=20.0),
    ]
    match_events(events, gt_events)
    assert events[1].matched_gt_index == 0
    assert events[1].matched_iou == 1.0
    assert events[0].matched_gt_index is None
    assert events[0].duplicate is True


def test_end_to_end_mistake_metrics_count_fp_and_fn() -> None:
    gt_events = [
        gt(step_index=0, mistake=True, start=10.0, end=20.0),
        gt(step_index=1, mistake=False, start=30.0, end=40.0),
        gt(step_index=2, mistake=True, start=50.0, end=60.0),
    ]
    events = [
        event(step_index=0, start=10.0, end=20.0, mistake=True),
        event(step_index=1, start=30.0, end=40.0, mistake=True),
        event(step_index=9, start=70.0, end=80.0, mistake=True),
    ]
    match_events(events, gt_events)
    metrics = compute_metrics(events, gt_events)
    assert metrics["end_to_end_mistake_at_iou_0.5/tp"] == 1.0
    assert metrics["end_to_end_mistake_at_iou_0.5/fp"] == 2.0
    assert metrics["end_to_end_mistake_at_iou_0.5/fn"] == 1.0
    assert metrics["end_to_end_mistake_precision_at_iou_0.5"] == 1 / 3
    assert metrics["end_to_end_mistake_recall_at_iou_0.5"] == 0.5
    matrix = metrics["end_to_end_mistake_at_iou_0.5/confusion_matrix"]
    assert matrix["gt_mistake"]["pred_mistake"] == 1
    assert matrix["gt_mistake"]["missed"] == 1
    assert matrix["gt_correct"]["pred_mistake"] == 1
    assert matrix["unmatched_prediction"]["pred_mistake"] == 1


def test_conditional_module_c_metrics_only_use_matched_windows() -> None:
    gt_events = [
        gt(step_index=0, mistake=True, start=10.0, end=20.0),
        gt(step_index=1, mistake=False, start=30.0, end=40.0),
    ]
    events = [
        event(step_index=0, start=10.0, end=20.0, mistake=True),
        event(step_index=1, start=30.0, end=40.0, mistake=False),
        event(step_index=9, start=70.0, end=80.0, mistake=True),
    ]
    match_events(events, gt_events)
    metrics = compute_metrics(events, gt_events)
    assert metrics["module_c_given_match_iou_0.5/tp"] == 1.0
    assert metrics["module_c_given_match_iou_0.5/tn"] == 1.0
    assert metrics["module_c_given_match_iou_0.5/fp"] == 0.0
    assert metrics["mistake_f1_given_match_iou_0.5"] == 1.0


def test_latency_statistics_include_early_and_late_triggers() -> None:
    gt_events = [
        gt(step_index=0, start=10.0, end=20.0),
        gt(step_index=1, start=30.0, end=40.0),
    ]
    events = [
        event(step_index=0, start=10.0, end=20.0, trigger_time=15.0),
        event(step_index=1, start=30.0, end=40.0, trigger_time=45.0),
    ]
    match_events(events, gt_events)
    metrics = compute_metrics(events, gt_events)
    assert metrics["latency/count"] == 2.0
    assert metrics["latency/mean_seconds"] == 0.0
    assert metrics["latency/mean_absolute_seconds"] == 5.0
    assert metrics["latency/early_count"] == 1.0
    assert metrics["latency/late_count"] == 1.0
    assert metrics["latency_at_iou_0.5/count"] == 2.0


def test_prediction_runtime_statistics_for_completed_actions() -> None:
    gt_events = [gt(step_index=0, start=10.0, end=20.0)]
    events = [
        event(
            step_index=0,
            start=10.0,
            end=20.0,
            module_a_seconds=1.0,
            module_b_seconds=2.0,
            module_c_seconds=3.0,
        )
    ]
    match_events(events, gt_events)
    metrics = compute_metrics(events, gt_events)
    assert metrics["prediction_runtime/module_a/count"] == 1.0
    assert metrics["prediction_runtime/module_a/mean_seconds"] == 1.0
    assert metrics["prediction_runtime/module_b/mean_seconds"] == 2.0
    assert metrics["prediction_runtime/module_c/mean_seconds"] == 3.0
    assert metrics["prediction_runtime/pipeline_completed_action/mean_seconds"] == 6.0


def test_stage_metrics_are_separated_from_full_pipeline_metrics() -> None:
    gt_events = [gt(step_index=0, start=10.0, end=20.0)]
    events = [event(step_index=0, start=10.0, end=20.0, mistake=True)]
    match_events(events, gt_events)
    module_a_predictions = [(window("COMPLETE"), "COMPLETE", "COMPLETE", 0.25)]
    metrics = compute_metrics(events, gt_events, module_a_predictions=module_a_predictions)
    assert metrics["module_a/num_windows"] == 1.0
    assert metrics["module_a/recall"] == 1.0
    assert metrics["module_b_given_module_a_trigger/num_triggers"] == 1.0
    assert metrics["module_b_given_module_a_trigger/recall_at_0.5"] == 1.0


def test_module_a_extra_probability_uses_saved_candidate_nlls() -> None:
    raw = json.dumps({"candidate_nlls": {"A": 2.0, "B": 0.0, "C": 1.0}})
    probability = module_a_extra_probability(raw)
    assert probability is not None
    assert 0.66 < probability < 0.67
    assert module_a_extra_probability("not-json") is None


def test_deferred_pipeline_artifact_preserves_offline_evaluation_inputs(
    tmp_path: Path,
) -> None:
    args = SimpleNamespace(
        output_dir=str(tmp_path),
        output_json=None,
        output_jsonl=None,
        output_csv=None,
        defer_metrics=True,
        module_a_checkpoint=None,
        module_b_checkpoint=None,
        module_c_checkpoint=None,
        module_a_label_mode="step_id",
        online_reset_from_b=True,
        dataset="egoper",
    )
    pipeline_event = event(module_a_seconds=0.25, module_b_seconds=0.5)
    ground_truth = gt()
    module_a_predictions = [(window("C"), "C", "raw C", 0.25)]
    write_outputs(
        args,
        [pipeline_event],
        {},
        {"val_video_ids": ["video"]},
        gt_events=[ground_truth],
        module_a_predictions=module_a_predictions,
    )

    payload = json.loads(
        (tmp_path / "pipeline_artifacts.json").read_text(encoding="utf-8")
    )
    assert payload["schema_version"] == "online_pipeline_artifacts_v1"
    assert payload["events"][0]["module_b_prediction"] == [10.0, 20.0]
    assert payload["ground_truth"][0]["gt_reasoning"] == "reference reasoning"
    assert payload["module_a_predictions"][0]["prediction"] == "C"
    assert payload["module_a_predictions"][0]["raw_output"] == "raw C"
    assert payload["module_a_predictions"][0]["window"]["label"] == "C"
    assert payload["metrics"] == {}
    assert payload["video_shard"]["world_size"] == 1
    assert (tmp_path / "pipeline_ground_truth.json").is_file()
    assert (tmp_path / "module_a_predictions.jsonl").is_file()


def test_zero_shot_eval_allows_missing_checkpoints(monkeypatch) -> None:
    from online_mistake_detection.pipeline.runner import parse_args

    monkeypatch.setattr(
        "sys.argv",
        ["online_pipeline", "--zero-shot-eval", "--model-name", "unsloth/Qwen3.5-35B-A3B"],
    )
    args = parse_args()
    assert args.zero_shot_eval is True
    assert args.module_a_checkpoint is None
    assert args.module_b_checkpoint is None
    assert args.module_c_checkpoint is None


def test_pipeline_requires_checkpoints_without_zero_shot(monkeypatch) -> None:
    import pytest

    from online_mistake_detection.pipeline.runner import parse_args

    monkeypatch.setattr("sys.argv", ["online_pipeline"])
    with pytest.raises(SystemExit):
        parse_args()


def test_eval_split_and_merge_if_complete_skip_checkpoints(monkeypatch) -> None:
    from online_mistake_detection.pipeline.runner import parse_args

    monkeypatch.setattr(
        "sys.argv",
        ["online_pipeline", "--eval-split", "test", "--merge-if-complete", "/tmp/out"],
    )
    args = parse_args()
    assert args.eval_split == "test"
    assert args.merge_if_complete == "/tmp/out"


def test_video_checkpoints_resume_skips_finished_videos(tmp_path: Path) -> None:
    from online_mistake_detection.pipeline.video_checkpoints import (
        list_module_a_done,
        list_pipeline_done,
        refresh_remaining_videos,
        write_events_checkpoint,
        write_module_a_checkpoint,
        write_progress,
    )

    checkpoint_root = tmp_path / "videos"
    queue_dir = tmp_path / "work_queue"
    write_module_a_checkpoint(
        checkpoint_root,
        "done_a",
        module_a_rows=[{"window": {"video_id": "done_a"}, "prediction": "C"}],
        ground_truth_rows=[{"video_id": "done_a", "eligible": True}],
    )
    write_events_checkpoint(checkpoint_root, "done_a", [{"video_id": "done_a"}])
    write_module_a_checkpoint(
        checkpoint_root,
        "need_bc",
        module_a_rows=[{"window": {"video_id": "need_bc"}, "prediction": "C"}],
        ground_truth_rows=[{"video_id": "need_bc", "eligible": True}],
    )
    eval_ids = ["done_a", "need_bc", "need_a"]
    done_a = set(list_module_a_done(checkpoint_root))
    done_pipeline = set(list_pipeline_done(checkpoint_root))
    refresh_remaining_videos(
        queue_dir,
        set(eval_ids) - done_a,
        session_id="job1:a",
    )
    refresh_remaining_videos(
        queue_dir,
        done_a - done_pipeline,
        session_id="job1:bc",
        remaining_name="bc_remaining.json",
    )
    remaining_a = json.loads((queue_dir / "remaining.json").read_text(encoding="utf-8"))
    remaining_bc = json.loads((queue_dir / "bc_remaining.json").read_text(encoding="utf-8"))
    assert remaining_a == ["need_a"]
    assert remaining_bc == ["need_bc"]
    progress = write_progress(
        queue_dir,
        eval_split="test",
        val_video_ids=eval_ids,
        module_a_done=done_a,
        pipeline_done=done_pipeline,
    )
    assert progress["eval_split"] == "test"
    assert progress["complete"] is False
    assert progress["pipeline_done"] == 1


def test_write_run_manifest_is_safe_for_concurrent_ranks(tmp_path: Path) -> None:
    from concurrent.futures import ThreadPoolExecutor

    from online_mistake_detection.pipeline.video_checkpoints import (
        load_run_manifest,
        write_run_manifest,
    )

    queue_dir = tmp_path / "work_queue"
    payload = {
        "split": {"name": "test", "video_ids": ["video-0", "video-1"]},
        "eval_split": "test",
        "eval_video_ids": ["video-0", "video-1"],
    }

    def write() -> None:
        write_run_manifest(queue_dir, **payload)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _index: write(), range(8)))

    manifest = load_run_manifest(queue_dir)
    assert manifest["eval_video_ids"] == ["video-0", "video-1"]
    leftover = list(queue_dir.glob(".manifest.json.*.tmp"))
    assert leftover == []


def test_merge_shards_dir_skips_checkpoint_requirement(monkeypatch) -> None:
    from online_mistake_detection.pipeline.runner import parse_args

    monkeypatch.setattr(
        "sys.argv",
        ["online_pipeline", "--merge-shards-dir", "/tmp/pipeline_shards"],
    )
    args = parse_args()
    assert args.merge_shards_dir == "/tmp/pipeline_shards"


def test_video_shards_partition_without_overlap() -> None:
    from online_mistake_detection.pipeline.shards import (
        assign_videos_to_shards,
        shard_video_ids,
    )

    videos = [f"v{index:02d}" for index in range(16)]
    counts = {video_id: (index + 1) * 10 for index, video_id in enumerate(videos)}
    shards = [
        shard_video_ids(videos, rank=rank, world_size=8, window_counts=counts)
        for rank in range(8)
    ]
    assigned = [video_id for shard in shards for video_id in shard]
    assert sorted(assigned) == videos
    assert len(assigned) == len(set(assigned))
    pair = assign_videos_to_shards(
        ["heavy_a", "heavy_b", "light_a", "light_b"],
        world_size=2,
        window_counts={"heavy_a": 100, "heavy_b": 99, "light_a": 1, "light_b": 1},
    )
    assert ("heavy_a" in pair[0]) != ("heavy_b" in pair[0])


def test_merge_rejects_missing_or_overlapping_shards() -> None:
    import pytest

    from online_mistake_detection.pipeline.shards import merge_pipeline_artifacts

    base = {
        "schema_version": "online_pipeline_artifacts_v1",
        "args": {"video_shard_rank": 0, "video_shard_world_size": 2},
        "split": {"val_video_ids": ["a", "b"]},
        "checkpoint_provenance": {},
        "metrics": {},
        "events": [],
        "ground_truth": [],
        "module_a_predictions": [],
    }
    shard0 = {
        **base,
        "video_shard": {"rank": 0, "world_size": 2, "video_ids": ["a"]},
        "ground_truth": [{"video_id": "a", "step_index": 0, "gt_start": 0, "gt_end": 1, "occurrence_id": "a:0"}],
    }
    with pytest.raises(ValueError, match="Expected shard ranks"):
        merge_pipeline_artifacts([shard0])

    shard1_overlap = {
        **base,
        "args": {"video_shard_rank": 1, "video_shard_world_size": 2},
        "video_shard": {"rank": 1, "world_size": 2, "video_ids": ["a"]},
        "ground_truth": [{"video_id": "a", "step_index": 0, "gt_start": 0, "gt_end": 1, "occurrence_id": "a:0"}],
    }
    with pytest.raises(ValueError, match="Overlapping shard records"):
        merge_pipeline_artifacts([shard0, shard1_overlap])


def test_video_queue_claims_are_exclusive(tmp_path: Path) -> None:
    from online_mistake_detection.pipeline.shards import (
        claim_next_video,
        ensure_video_queue,
    )

    queue_dir = tmp_path / "work_queue"
    videos = [f"v{index}" for index in range(5)]
    ensure_video_queue(queue_dir, videos, {video: 10 - index for index, video in enumerate(videos)})
    claimed = []
    for rank in range(3):
        while True:
            video_id = claim_next_video(queue_dir, rank=rank)
            if video_id is None:
                break
            claimed.append((rank, video_id))
    assert [video_id for _rank, video_id in claimed] == ["v0", "v1", "v2", "v3", "v4"]
    assert claim_next_video(queue_dir, rank=0) is None
    assert len({video_id for _rank, video_id in claimed}) == 5


def test_write_merged_shard_directory_rebuilds_portable_artifact(tmp_path: Path) -> None:
    from online_mistake_detection.pipeline.runner import write_merged_shard_directory

    for rank, video_id in enumerate(("alpha", "beta")):
        shard_dir = tmp_path / "shards" / f"rank{rank:02d}-of-02"
        shard_dir.mkdir(parents=True)
        payload = {
            "schema_version": "online_pipeline_artifacts_v1",
            "args": {
                "output_dir": str(shard_dir),
                "video_shard_rank": rank,
                "video_shard_world_size": 2,
                "defer_metrics": True,
            },
            "split": {"val_video_ids": ["alpha", "beta"]},
            "checkpoint_provenance": {},
            "metrics": {},
            "events": [
                {
                    "video_id": video_id,
                    "trigger_time": 10.0 + rank,
                    "step_index": 0,
                    "accumulation_start": 0.0,
                    "accumulation_end": 10.0,
                }
            ],
            "ground_truth": [
                {
                    "video_id": video_id,
                    "step_index": 0,
                    "gt_start": 0.0,
                    "gt_end": 10.0,
                    "occurrence_id": f"{video_id}:0",
                    "eligible": True,
                }
            ],
            "module_a_predictions": [
                {
                    "window": {
                        "video_id": video_id,
                        "window_start": 0.0,
                        "window_end": 10.0,
                        "step_index": 0,
                    },
                    "prediction": "C",
                }
            ],
            "video_shard": {
                "rank": rank,
                "world_size": 2,
                "video_ids": [video_id],
            },
        }
        (shard_dir / "pipeline_artifacts.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )

    merged_path = write_merged_shard_directory(tmp_path)
    merged = json.loads(merged_path.read_text(encoding="utf-8"))
    assert merged_path.name == "pipeline_artifacts.json"
    assert merged["video_shard"]["merged"] is True
    assert merged["video_shard"]["video_ids"] == ["alpha", "beta"]
    assert [row["video_id"] for row in merged["events"]] == ["alpha", "beta"]
    assert merged["artifact_counts"]["module_a_predictions"] == 2
    assert (tmp_path / "pipeline_events.jsonl").is_file()
    assert (tmp_path / "module_a_predictions.json").is_file()
