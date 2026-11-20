import math

from online_mistake_detection.metrics.abc_monolith import (
    parse_monolith_prediction,
    score_coverage,
    scores_to_probabilities,
)
from online_mistake_detection.training.abc_monolith import ABCMonolithMetricsCallback


def test_candidate_nll_normalization_and_coverage() -> None:
    probabilities = scores_to_probabilities({"CORRECT": 2.0, "SLIP": 0.0})
    assert math.isclose(sum(probabilities.values()), 1.0)
    assert probabilities["SLIP"] > probabilities["CORRECT"]
    partial = score_coverage(
        [{"candidate_nlls": {"CORRECT": 1.0}}, {"candidate_nlls": None}]
    )
    assert partial["coverage"] == 0.5
    assert partial["complete"] is False


def test_invalid_prediction_candidate_scoring_does_not_copy_gold_target() -> None:
    example = {
        "target": (
            '{"step_id":"STEP_0","start":7,"end":9,'
            '"classification":"SLIP","reasoning":"secret gold rationale"}'
        ),
        "window_start": 0.0,
        "window_end": 10.0,
        "metadata": {"all_instructions": ["only step"]},
    }
    step_targets, class_targets = ABCMonolithMetricsCallback._candidate_targets(
        example,
        parse_monolith_prediction("not json"),
    )
    assert class_targets == {}
    assert all("secret gold rationale" not in target for target in step_targets.values())
    scored_steps, scored_classes = ABCMonolithMetricsCallback.scoring_targets(
        example,
        parse_monolith_prediction("not json"),
    )
    assert scored_steps == {}
    assert scored_classes == {}


def test_not_completed_polls_are_not_scored() -> None:
    example = {
        "window_start": 0.0,
        "window_end": 10.0,
        "metadata": {"all_instructions": ["boil water", "pour water"]},
    }
    prediction = parse_monolith_prediction('{"status":"NOT_COMPLETED"}')
    step_targets, class_targets = ABCMonolithMetricsCallback.scoring_targets(
        example, prediction
    )
    assert step_targets == {}
    assert class_targets == {}


def test_completed_response_scores_only_that_steps_classes() -> None:
    example = {
        "window_start": 0.0,
        "window_end": 10.0,
        "metadata": {"all_instructions": ["boil water", "pour water"]},
    }
    prediction = parse_monolith_prediction(
        '{"step_id":"STEP_1","start":1.0,"end":4.0,'
        '"classification":"SLIP","reasoning":"Water was spilled."}'
    )
    step_targets, class_targets = ABCMonolithMetricsCallback.scoring_targets(
        example, prediction
    )
    assert step_targets == {}
    assert set(class_targets) == {"CORRECT", "SLIP", "MODIFICATION", "CORRECTION"}
    assert all('"step_id":"STEP_1"' in target for target in class_targets.values())

