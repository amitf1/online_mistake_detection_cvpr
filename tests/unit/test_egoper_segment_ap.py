from online_mistake_detection.metrics.egoper import segment_average_precision
from online_mistake_detection.pipeline.events import GroundTruthEvent, PipelineEvent


def _gt(step: int, start: float, end: float, label: str) -> GroundTruthEvent:
    return GroundTruthEvent(
        "v", "task", step, "instruction", start, end, label != "CORRECT", "", True,
        gt_fine_class=label,
    )


def _prediction(step: int, start: float, end: float, label: str, score: float) -> PipelineEvent:
    return PipelineEvent(
        "v",
        "task",
        step,
        "instruction",
        end,
        0.0,
        end,
        f"STEP_{step}",
        pred_global_start=start,
        pred_global_end=end,
        module_c_classification=label,
        module_c_mistake=True,
        module_c_error_score=score,
        module_c_class_probabilities={label: score},
    )


def test_segment_ap_is_ranked_one_to_one_and_step_aware() -> None:
    ground_truth = [_gt(0, 0.0, 2.0, "SLIP")]
    predictions = [
        _prediction(1, 0.0, 2.0, "SLIP", 0.9),
        _prediction(0, 0.0, 2.0, "SLIP", 0.8),
    ]
    metric = segment_average_precision(predictions, ground_truth)
    assert metric["by_threshold"]["0.5"]["binary_ap"] == 1.0
    assert metric["by_threshold"]["0.5"]["fine_map"] == 0.5


def test_segment_ap_uses_interpolated_precision_envelope() -> None:
    ground_truth = [
        _gt(0, 0.0, 2.0, "SLIP"),
        _gt(1, 3.0, 5.0, "SLIP"),
    ]
    predictions = [
        _prediction(0, 10.0, 12.0, "SLIP", 0.9),
        _prediction(0, 0.0, 2.0, "SLIP", 0.8),
        _prediction(1, 3.0, 5.0, "SLIP", 0.7),
    ]
    metric = segment_average_precision(predictions, ground_truth)
    assert metric["by_threshold"]["0.5"]["binary_ap"] == 2 / 3
    assert metric["by_threshold"]["0.5"]["fine_map"] == 2 / 3

