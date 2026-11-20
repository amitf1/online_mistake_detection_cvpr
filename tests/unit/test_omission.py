from pathlib import Path

from online_mistake_detection.metrics.omission import (
    collapse_consecutive_duplicates,
    infer_omissions,
    omission_metrics,
    online_omission_metrics,
)
from online_mistake_detection.pipeline.events import PipelineEvent
from online_mistake_detection.pipeline.omission import (
    OnlineOmissionPipeline,
    OnlineOmissionTracker,
)
from online_mistake_detection.pipeline.procedural_graph import (
    DependencyGraphTracker,
    DetectedOccurrence,
)
from online_mistake_detection.pipeline.runner import apply_graph_stage
from online_mistake_detection.pipeline.task_graph import TaskGraph
from online_mistake_detection.schemas import VideoRecord


def branching_graph() -> TaskGraph:
    return TaskGraph(
        task_id="branch",
        edges=((0, 1), (1, 2), (1, 3), (2, 4), (3, 4), (4, 5)),
        start_node=0,
        end_node=5,
    )


def test_branching_path_does_not_invent_optional_omission() -> None:
    result = infer_omissions([0, 3], branching_graph())
    assert result.omitted_steps == frozenset()
    assert len(result.tied_references) == 2


def test_only_common_omissions_survive_ambiguous_ties() -> None:
    result = infer_omissions([0], branching_graph())
    assert result.omitted_steps == frozenset({3})


def test_repeats_additions_and_corrections_do_not_create_omissions() -> None:
    # Consecutive repeats collapse; negative EXTRA events are ignored.
    result = infer_omissions([0, 0, -1, 1, 3], branching_graph())
    assert result.omitted_steps == frozenset()
    assert collapse_consecutive_duplicates([1, 1, 2, 2, 1]) == (1, 2, 1)


def test_omission_metrics_are_separate_set_metrics() -> None:
    metrics = omission_metrics(
        [{2, 4}, set()],
        [{2, 4}, {3}],
    )
    assert metrics["omission/accuracy"] == 0.5
    assert metrics["omission/mean_iou"] == 0.5


def test_online_tracker_treats_all_fork_prerequisites_as_required() -> None:
    tracker = OnlineOmissionTracker(branching_graph())
    assert tracker.update(0).candidate_omitted_steps == frozenset()

    update = tracker.update(3)
    assert update.transition_valid
    assert update.possibilities == (frozenset({1, 2}),)
    assert update.candidate_omitted_steps == frozenset({1, 2})
    assert update.confirmed_omitted_steps == frozenset()


def test_online_tracker_does_not_mark_future_steps_until_finalize() -> None:
    tracker = OnlineOmissionTracker(branching_graph())
    tracker.update(0)
    update = tracker.update(1)
    assert update.candidate_omitted_steps == frozenset()

    final = tracker.finalize()
    assert final is not None
    assert final.possibilities == (frozenset({2, 3}),)
    assert final.confirmed_omitted_steps == frozenset({2, 3})


def test_online_omission_evaluation_accepts_any_matching_possibility() -> None:
    metrics = online_omission_metrics(
        [[{1}, {2}]],
        [{2}],
    )
    assert metrics["online_omission/alert_tp"] == 1.0
    assert metrics["online_omission/alert_fp"] == 0.0
    assert metrics["online_omission/step_recall"] == 1.0


def test_online_pipeline_keeps_state_per_video() -> None:
    pipeline = OnlineOmissionPipeline({"branch": branching_graph()})
    pipeline.on_step_completion(video_id="first", task_id="branch", detected_step=0)
    pipeline.on_step_completion(video_id="second", task_id="branch", detected_step=0)
    update = pipeline.on_step_completion(video_id="first", task_id="branch", detected_step=3)
    assert update.candidate_omitted_steps == frozenset({1, 2})
    second_final = pipeline.finalize_video("second")
    assert second_final is not None
    assert second_final.observed_steps == (0,)


def test_dependency_tracker_keeps_and_dag_omissions_provisional_until_end() -> None:
    tracker = DependencyGraphTracker(branching_graph())
    tracker.update(
        DetectedOccurrence("step-0", 0, 0.0, 1.0, 1.0, localized=True)
    )
    updates = tracker.update(
        DetectedOccurrence("step-3", 3, 4.0, 5.0, 5.0, localized=True)
    )
    missing = [alert for alert in updates if alert.kind == "MISSING_STEP"]
    assert {alert.step_index for alert in missing} == {1, 2}
    assert {alert.status for alert in missing} == {"provisional"}

    finalized = tracker.finalize(trigger_time=6.0)
    confirmed = [alert for alert in finalized if alert.kind == "MISSING_STEP"]
    assert {alert.step_index for alert in confirmed} == {1, 2}
    assert {alert.status for alert in confirmed} == {"confirmed"}


def test_graph_stage_consumes_module_a_prediction_not_gt_window_step() -> None:
    graph = branching_graph()
    record = VideoRecord(
        dataset="egoper",
        task_id="branch",
        video_id="video",
        video_path=Path("video.mp4"),
        instructions=["zero", "one", "two", "three"],
        segments=[],
        task_graph=graph,
    )
    event = PipelineEvent(
        video_id="video",
        task_id="branch",
        step_index=1,
        instruction="GT step one",
        trigger_time=5.0,
        accumulation_start=0.0,
        accumulation_end=5.0,
        module_a_prediction="F",
        pred_global_start=4.0,
        pred_global_end=5.0,
    )

    apply_graph_stage([event], [record])

    missing = {
        int(alert["step_index"])
        for alert in event.graph_alerts
        if alert["kind"] == "MISSING_STEP"
    }
    assert missing == {0, 1, 2}
