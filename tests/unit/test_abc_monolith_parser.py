from __future__ import annotations

import json

from online_mistake_detection.prompts.abc_monolith import (
    MONOLITH_SYSTEM_PROMPT,
    format_monolith_target,
    monolith_global_span,
    parse_monolith_prediction,
    system_prompt_for_dataset,
)
from online_mistake_detection.prompts.contracts import NOT_COMPLETED_TARGET


def test_parser_accepts_only_the_two_monolith_shapes() -> None:
    negative = parse_monolith_prediction(NOT_COMPLETED_TARGET, clip_duration=5.0)
    assert negative.valid and not negative.completed

    target = format_monolith_target(
        step_id="STEP_2",
        start=1.25,
        end=4.5,
        classification="SLIP",
        reasoning="Water enters the wrong bowl.",
    )
    positive = parse_monolith_prediction(target, clip_duration=5.0)
    assert positive.valid and positive.completed
    assert monolith_global_span(positive, window_start=10.0) == (11.25, 14.5)

    assert not parse_monolith_prediction('{"status":"WAIT"}').valid
    assert not parse_monolith_prediction('{"status":"COMPLETED"}').valid
    assert not parse_monolith_prediction(
        '{"step_id":"STEP_2","start":1,"end":2,"classification":"SLIP",'
        '"reasoning":"visible evidence","status":"COMPLETED"}'
    ).valid
    assert "WAIT" in MONOLITH_SYSTEM_PROMPT
    assert "Never return WAIT" in MONOLITH_SYSTEM_PROMPT


def test_parser_enforces_extra_addition_nonempty_reasoning_and_local_bounds() -> None:
    base = {
        "step_id": "EXTRA",
        "start": 1.0,
        "end": 2.0,
        "classification": "ADDITION",
        "reasoning": "An off-script action finishes.",
    }
    assert parse_monolith_prediction(json.dumps(base), clip_duration=3.0).valid

    inconsistent = {**base, "classification": "SLIP"}
    assert not parse_monolith_prediction(json.dumps(inconsistent)).valid
    inconsistent = {**base, "step_id": "STEP_0"}
    assert not parse_monolith_prediction(json.dumps(inconsistent)).valid
    assert not parse_monolith_prediction(json.dumps({**base, "reasoning": "  "})).valid
    assert not parse_monolith_prediction(json.dumps({**base, "end": 4.0}), clip_duration=3.0).valid
    assert not parse_monolith_prediction(
        "prefix " + json.dumps(base), clip_duration=3.0
    ).valid


def test_parser_accepts_displayed_clip_boundary_and_clamps_to_exact_media() -> None:
    payload = {
        "step_id": "STEP_0",
        "start": 4.0,
        "end": 5.0,
        "classification": "CORRECT",
        "reasoning": "The action finishes at the displayed clip boundary.",
    }
    parsed = parse_monolith_prediction(json.dumps(payload), clip_duration=4.997)
    assert parsed.valid
    assert parsed.end == 4.997


def test_egoper_system_prompt_uses_egoper_classes() -> None:
    prompt = system_prompt_for_dataset("egoper")
    assert prompt == MONOLITH_SYSTEM_PROMPT
    assert "MISTAKE" not in prompt
