from __future__ import annotations

from online_mistake_detection.metrics import egoper
from online_mistake_detection.pipeline.events import GroundTruthEvent, PipelineEvent


def _gt(
    *,
    step: int,
    start: float,
    end: float,
    mistake: bool,
    fine_class: str,
    video_id: str = "video",
) -> GroundTruthEvent:
    return GroundTruthEvent(
        video_id=video_id,
        task_id="tea",
        step_index=step,
        instruction=f"step {step}",
        gt_start=start,
        gt_end=end,
        gt_mistake=mistake,
        gt_reasoning="",
        eligible=True,
        gt_fine_class=fine_class,
    )


def _prediction(
    *,
    step: int,
    start: float,
    end: float,
    mistake: bool | None,
    fine_class: str | None,
    score: float | None = None,
    video_id: str = "video",
) -> PipelineEvent:
    return PipelineEvent(
        video_id=video_id,
        task_id="tea",
        step_index=step,
        instruction=f"step {step}",
        trigger_time=end,
        accumulation_start=0.0,
        accumulation_end=end,
        module_a_prediction="B" if step < 0 else chr(ord("C") + step),
        pred_global_start=start,
        pred_global_end=end,
        module_c_classification=fine_class,
        module_c_mistake=mistake,
        module_c_error_score=score,
    )


def test_tas_and_mistake_f1_are_independent() -> None:
    targets = [
        _gt(step=0, start=0.0, end=10.0, mistake=True, fine_class="MODIFICATION"),
        _gt(step=1, start=10.0, end=20.0, mistake=False, fine_class="CORRECT"),
    ]
    predictions = [
        _prediction(
            step=9,
            start=0.0,
            end=10.0,
            mistake=True,
            fine_class="MODIFICATION",
        ),
        _prediction(
            step=1,
            start=10.0,
            end=20.0,
            mistake=False,
            fine_class="CORRECT",
        ),
    ]

    assert egoper.tas_f1(predictions, targets)["f1_at_0.5"] == 0.5
    assert egoper.binary_mistake_f1(predictions, targets)["macro_f1_at_0.5"] == 1.0
    assert egoper.fine_mistake_f1(predictions, targets)["macro_f1_at_0.5"] == 1.0


def test_tas_scores_addition_segments(monkeypatch) -> None:
    monkeypatch.setattr(
        egoper,
        "_annotated_addition_steps",
        lambda: {("annotated", 10.0, 20.0): 4},
    )
    targets = [
        _gt(step=0, start=0.0, end=10.0, mistake=False, fine_class="CORRECT"),
        _gt(step=-1, start=10.0, end=20.0, mistake=True, fine_class="ADDITION"),
    ]
    extra = [
        _prediction(step=0, start=0.0, end=10.0, mistake=False, fine_class="CORRECT"),
        _prediction(step=-1, start=10.0, end=20.0, mistake=True, fine_class="ADDITION"),
    ]
    assert egoper.tas_f1(extra, targets)["f1_at_0.5"] == 1.0
    assert egoper.tas_f1(extra, targets, include_additions=False)["f1_at_0.5"] == 1.0

    named_step = [
        _prediction(step=4, start=10.0, end=20.0, mistake=False, fine_class="CORRECT", video_id=video)
        for video in ("annotated", "video")
    ]
    annotated_targets = [
        _gt(step=-1, start=10.0, end=20.0, mistake=True, fine_class="ADDITION", video_id=video)
        for video in ("annotated", "video")
    ]
    result = egoper.tas_f1(named_step, annotated_targets)
    assert (result["tp"], result["fp"], result["fn"]) == (1.0, 1.0, 1.0)


def test_unclassified_prediction_is_not_treated_as_correct() -> None:
    targets = [
        _gt(step=0, start=0.0, end=10.0, mistake=False, fine_class="CORRECT"),
    ]
    predictions = [
        _prediction(
            step=0,
            start=0.0,
            end=10.0,
            mistake=None,
            fine_class=None,
        ),
    ]

    metric = egoper.binary_mistake_f1(predictions, targets)
    assert metric["tp"] == 0.0
    assert metric["fp"] == 1.0
    assert metric["fn"] == 1.0


def test_binary_error_metrics_include_correction_segments() -> None:
    targets = [
        _gt(
            step=0,
            start=0.0,
            end=10.0,
            mistake=False,
            fine_class="CORRECTION",
        ),
    ]
    predictions = [
        _prediction(
            step=0,
            start=0.0,
            end=10.0,
            mistake=False,
            fine_class="CORRECTION",
        ),
    ]

    metric = egoper.binary_mistake_f1(predictions, targets)
    assert metric["error_f1_at_0.5"] == 1.0
    assert egoper.error_detection_accuracy(predictions, targets)["eda"] == 1.0


def test_mistake_f1_reports_binary_and_fine_results_at_all_iou_thresholds() -> None:
    targets = [
        _gt(step=0, start=0.0, end=10.0, mistake=True, fine_class="SLIP"),
    ]
    predictions = [
        _prediction(
            step=0,
            start=0.0,
            end=3.0,
            mistake=True,
            fine_class="SLIP",
        ),
    ]

    binary = egoper.binary_mistake_f1(predictions, targets)
    fine = egoper.fine_mistake_f1(predictions, targets)

    assert tuple(binary["by_threshold"]) == ("0.1", "0.2", "0.3", "0.5", "0.7", "0.9")
    assert binary["error_f1_by_threshold"]["0.3"] == 1.0
    assert binary["error_f1_by_threshold"]["0.5"] == 0.0
    assert binary["error_f1_at_iou_0.3"] == 1.0
    assert fine["by_threshold"]["0.3"]["macro_f1_at_0.3"] == 1.0
    assert fine["by_threshold"]["0.5"]["macro_f1_at_0.5"] == 0.0
    assert fine["macro_f1_at_iou_0.3"] == 1.0
    assert fine["by_threshold"]["0.3"]["per_class"]["SLIP"]["f1_at_0.3"] == 1.0


def test_eda_uses_any_predicted_error_overlap_with_gt_segment() -> None:
    targets = [
        _gt(step=0, start=0.0, end=10.0, mistake=True, fine_class="SLIP"),
        _gt(step=1, start=10.0, end=20.0, mistake=False, fine_class="CORRECT"),
    ]
    predictions = [
        _prediction(
            step=7,
            start=4.0,
            end=5.0,
            mistake=True,
            fine_class="SLIP",
        ),
    ]

    metric = egoper.error_detection_accuracy(predictions, targets)
    assert metric["eda"] == 1.0
    # One second of a ten-second segment is not a majority of its frames.
    assert metric["majority"]["eda"] == 0.5


def test_eda_majority_merges_overlaps_and_requires_more_than_half() -> None:
    targets = [
        _gt(step=0, start=0.0, end=10.0, mistake=True, fine_class="SLIP"),
    ]
    predictions = [
        _prediction(
            step=0,
            start=0.0,
            end=3.0,
            mistake=True,
            fine_class="SLIP",
        ),
        _prediction(
            step=0,
            start=2.0,
            end=5.0,
            mistake=True,
            fine_class="SLIP",
        ),
    ]

    # The union is [0, 5], exactly 50% of the GT segment. Overlap is counted
    # once and a tie remains NORMAL, matching EgoPER's strict majority vote.
    metric = egoper.error_detection_accuracy(predictions, targets)
    assert metric["majority"]["eda"] == 0.0

    predictions[1].pred_global_end = 5.1
    metric = egoper.error_detection_accuracy(predictions, targets)
    assert metric["majority"]["eda"] == 1.0


def test_framewise_auc_requires_scores_and_uses_them_when_present() -> None:
    targets = [
        _gt(step=0, start=0.0, end=1.0, mistake=True, fine_class="SLIP"),
        _gt(step=1, start=1.0, end=2.0, mistake=False, fine_class="CORRECT"),
    ]
    hard_predictions = [
        _prediction(
            step=0,
            start=0.0,
            end=1.0,
            mistake=True,
            fine_class="SLIP",
        ),
    ]
    assert egoper.framewise_auc(hard_predictions, targets)["available"] is False

    scored_predictions = [
        _prediction(
            step=0,
            start=0.0,
            end=1.0,
            mistake=True,
            fine_class="SLIP",
            score=0.9,
        ),
        _prediction(
            step=1,
            start=1.0,
            end=2.0,
            mistake=False,
            fine_class="CORRECT",
            score=0.1,
        ),
    ]
    metric = egoper.framewise_auc(scored_predictions, targets, fps=4.0)
    assert metric["available"] is True
    assert metric["auc"] == 1.0

    scored_predictions[1].module_c_error_score = None
    partial = egoper.framewise_auc(scored_predictions, targets, fps=4.0)
    assert partial["available"] is False
    assert partial["prediction_score_coverage"] == 0.5


def test_omission_metrics_compare_missing_step_sets(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        egoper,
        "_task_graphs",
        lambda _ground_truth: ({}, {"tea": {0, 1, 2}}, "test"),
    )
    targets = [
        _gt(step=0, start=0.0, end=1.0, mistake=False, fine_class="CORRECT"),
        _gt(step=2, start=2.0, end=3.0, mistake=False, fine_class="CORRECT"),
    ]
    predictions = [
        _prediction(
            step=0,
            start=0.0,
            end=1.0,
            mistake=False,
            fine_class="CORRECT",
        ),
        _prediction(
            step=1,
            start=1.0,
            end=2.0,
            mistake=False,
            fine_class="CORRECT",
        ),
    ]

    metric = egoper.omission_metrics(predictions, targets)
    assert metric["o_acc"] == 0.0
    assert metric["o_iou"] == 0.0
    assert metric["num_true_omission_videos"] == 1.0


def test_gtg2vid_omission_metrics_use_micro_counts_and_skip_complete_videos(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        egoper,
        "_task_graphs",
        lambda _ground_truth: ({}, {"tea": {0, 1, 2}}, "test"),
    )
    targets = [
        _gt(
            step=0,
            start=0.0,
            end=1.0,
            mistake=False,
            fine_class="CORRECT",
            video_id="first",
        ),
        _gt(
            step=0,
            start=0.0,
            end=1.0,
            mistake=False,
            fine_class="CORRECT",
            video_id="second",
        ),
        _gt(
            step=1,
            start=1.0,
            end=2.0,
            mistake=False,
            fine_class="CORRECT",
            video_id="second",
        ),
        _gt(
            step=0,
            start=0.0,
            end=1.0,
            mistake=False,
            fine_class="CORRECT",
            video_id="complete",
        ),
        _gt(
            step=1,
            start=1.0,
            end=2.0,
            mistake=False,
            fine_class="CORRECT",
            video_id="complete",
        ),
        _gt(
            step=2,
            start=2.0,
            end=3.0,
            mistake=False,
            fine_class="CORRECT",
            video_id="complete",
        ),
    ]
    predictions = [
        _prediction(
            step=0,
            start=0.0,
            end=1.0,
            mistake=False,
            fine_class="CORRECT",
            video_id="first",
        ),
        _prediction(
            step=1,
            start=1.0,
            end=2.0,
            mistake=False,
            fine_class="CORRECT",
            video_id="first",
        ),
        _prediction(
            step=0,
            start=0.0,
            end=1.0,
            mistake=False,
            fine_class="CORRECT",
            video_id="second",
        ),
    ]

    metric = egoper.gtg2vid_omission_metrics(predictions, targets)
    # first: target {1,2}, prediction {2}; second: target {2},
    # prediction {1,2}. The complete video is excluded.
    assert metric["num_videos"] == 2.0
    assert metric["intersection"] == 2.0
    assert metric["target_omissions"] == 3.0
    assert metric["union"] == 4.0
    assert metric["o_acc"] == 2.0 / 3.0
    assert metric["o_iou"] == 0.5


def test_official_egoper_omission_is_per_video_and_includes_complete_videos(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        egoper,
        "_task_graphs",
        lambda _ground_truth: ({}, {"tea": {0, 1}}, "test"),
    )
    targets = [
        _gt(
            step=0,
            start=0.0,
            end=1.0,
            mistake=False,
            fine_class="CORRECT",
            video_id="omission",
        ),
        _gt(
            step=0,
            start=0.0,
            end=1.0,
            mistake=False,
            fine_class="CORRECT",
            video_id="complete",
        ),
        _gt(
            step=1,
            start=1.0,
            end=2.0,
            mistake=False,
            fine_class="CORRECT",
            video_id="complete",
        ),
    ]
    predictions = [
        _prediction(
            step=0,
            start=0.0,
            end=1.0,
            mistake=False,
            fine_class="CORRECT",
            video_id="omission",
        ),
    ]

    metric = egoper.egoper_omission_metrics(predictions, targets)
    # The omission video is perfect. The complete video receives O-Acc=1 by
    # EgoPER's empty-target rule but O-IoU=0 because both steps were missed.
    assert metric["num_videos"] == 2.0
    assert metric["num_true_omission_videos"] == 1.0
    assert metric["o_acc"] == 1.0
    assert metric["o_iou"] == 0.5


def test_system_level_segments_skip_module_rejections() -> None:
    targets = [_gt(step=0, start=0.0, end=10.0, mistake=False, fine_class="CORRECT")]
    final = _prediction(step=0, start=0.0, end=10.0, mistake=False, fine_class="CORRECT")
    final.stage_status = "C_CLASSIFIED"
    rejected = _prediction(step=0, start=0.5, end=9.5, mistake=None, fine_class=None)
    rejected.stage_status = "C_NOT_COMPLETED"

    assert egoper._predicted_segments([final, rejected]) == [final]
    assert egoper.tas_f1([final, rejected], targets)["f1_at_0.5"] == 1.0
