import json
from pathlib import Path

from online_mistake_detection.metrics.temporal import parse_temporal_prediction


def test_checked_in_golden_contracts() -> None:
    fixture = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "fixtures"
            / "golden_windows.json"
        ).read_text(encoding="utf-8")
    )
    assert [row["label"] for row in fixture["module_a"]] == ["NONE", "STEP_0"]
    module_b = fixture["module_b"]
    accepted = parse_temporal_prediction(
        json.dumps({"relevant_windows": [module_b["accepted"]]}),
        clip_duration=module_b["clip_duration"],
    )
    clamped = parse_temporal_prediction(
        json.dumps({"relevant_windows": [module_b["raw_out_of_bounds"]]}),
        clip_duration=module_b["clip_duration"],
    )
    assert accepted.span == tuple(module_b["accepted"])
    assert clamped.raw_span == tuple(module_b["raw_out_of_bounds"])
    assert clamped.span == tuple(module_b["clamped"])
    assert clamped.clamp_reason == module_b["clamp_reason"]
    assert fixture["module_c"]["aggregate_classes"] == [
        "MISTAKE",
        "CORRECTION",
        "CORRECT",
    ]
