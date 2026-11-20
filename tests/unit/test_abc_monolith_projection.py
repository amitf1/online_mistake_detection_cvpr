import json

from online_mistake_detection.metrics.abc_monolith import (
    parse_monolith_prediction,
    project_monolith_prediction,
)


def test_strict_contract_and_global_projection() -> None:
    raw = json.dumps(
        {
            "step_id": "STEP_2",
            "start": 1.5,
            "end": 4.0,
            "classification": "SLIP",
            "reasoning": "The action was omitted.",
        }
    )
    parsed = parse_monolith_prediction(raw, clip_duration=5.0)
    assert parsed.valid
    projected = project_monolith_prediction(
        parsed,
        video_id="v",
        task_id="t",
        instruction="step",
        window_start=10.0,
        window_end=15.0,
    )
    assert projected.module_a["prediction"] == "E"
    assert projected.event is not None
    assert projected.event.pred_global_start == 11.5
    assert projected.event.pred_global_end == 14.0


def test_parser_rejects_legacy_status_and_extra_class_mismatch() -> None:
    assert not parse_monolith_prediction('{"status":"COMPLETE"}').valid
    assert not parse_monolith_prediction('{"status":"WAIT"}').valid
    raw = (
        '{"step_id":"EXTRA","start":0,"end":1,'
        '"classification":"CORRECT","reasoning":"x"}'
    )
    assert not parse_monolith_prediction(raw, clip_duration=1.0).valid

