from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from online_mistake_detection.training.abc_monolith import (
    ABCMonolithUnslothEnvGuardCallback,
    add_stage_composite,
    geometric_stage_composite,
    parse_monolith_response,
    compute_monolith_metrics,
    restore_abc_monolith_training_state,
)


class _FakeModel:
    def __init__(self) -> None:
        self.training = False
        self.is_gradient_checkpointing = False
        self.config = SimpleNamespace(
            use_cache=True,
            text_config=SimpleNamespace(use_cache=True),
        )
        self.checkpointing_enabled = False

    def train(self) -> None:
        self.training = True

    def gradient_checkpointing_enable(self) -> None:
        self.checkpointing_enabled = True
        self.is_gradient_checkpointing = True


def test_restore_training_state_reenables_unsloth_checkpointing(monkeypatch) -> None:
    calls: list[tuple[object, str]] = []

    class FakeFastVisionModel:
        @staticmethod
        def for_training(model, *, use_gradient_checkpointing):
            calls.append((model, use_gradient_checkpointing))

    monkeypatch.setitem(
        sys.modules,
        "unsloth",
        SimpleNamespace(FastVisionModel=FakeFastVisionModel),
    )
    model = _FakeModel()
    restore_abc_monolith_training_state(model)
    assert calls == [(model, "unsloth")]
    assert model.training is True
    assert model.checkpointing_enabled is True
    assert model.config.use_cache is False
    assert model.config.text_config.use_cache is False


def test_guard_defers_restore_until_next_step(monkeypatch) -> None:
    callback = ABCMonolithUnslothEnvGuardCallback.__new__(
        ABCMonolithUnslothEnvGuardCallback
    )
    callback.trainer = None
    callback._restore_training_state_at_next_step = False
    restored: list[object] = []
    released: list[bool] = []
    monkeypatch.setattr(
        "online_mistake_detection.training.abc_monolith.disable_unsloth_extra_outputs",
        lambda: None,
    )
    monkeypatch.setattr(
        "online_mistake_detection.training.abc_monolith.restore_abc_monolith_training_state",
        lambda model: restored.append(model),
    )
    monkeypatch.setattr(
        "online_mistake_detection.training.abc_monolith.release_cuda_memory",
        lambda: released.append(True),
    )
    model = object()
    control = object()
    assert callback.on_evaluate(None, None, control) is control
    assert restored == []
    callback.on_step_begin(None, None, control, model=model)
    assert restored == [model]
    assert released == [True]
    assert callback._restore_training_state_at_next_step is False


def test_monolith_parser_enforces_joint_contract() -> None:
    assert parse_monolith_response('{"status":"NOT_COMPLETED"}') == {
        "status": "NOT_COMPLETED"
    }
    assert parse_monolith_response(
        '{"step_id":"STEP_2","start":1,"end":2,'
        '"classification":"SLIP","reasoning":"visible slip"}'
    ) is not None
    assert parse_monolith_response(
        '{"step_id":"EXTRA","start":1,"end":2,'
        '"classification":"CORRECT","reasoning":"wrong cross-field pair"}'
    ) is None
    assert parse_monolith_response('{"status":"WAIT"}') is None
    assert parse_monolith_response('{"status":"COMPLETED"}') is None


def test_stage_composite_is_geometric_mean() -> None:
    assert geometric_stage_composite(0.8, 0.5, 0.2) == pytest.approx(
        (0.8 * 0.5 * 0.2) ** (1 / 3)
    )
    metrics = add_stage_composite(
        {
            "a_total_macro_f1": 0.8,
            "b_mixed_raw_macro": 0.5,
            "c_all_class_macro_f1": 0.2,
        }
    )
    assert metrics["eval_monolith/a_total_macro_f1"] == 0.8
    assert metrics["eval_monolith/stage_composite"] == pytest.approx(
        (0.8 * 0.5 * 0.2) ** (1 / 3)
    )


def test_golden_native_projection_is_perfect_for_identical_joint_outputs() -> None:
    completed = (
        '{"step_id":"STEP_0","start":1.0,"end":2.0,'
        '"classification":"CORRECT","reasoning":"The step is correct."}'
    )
    negative = '{"status":"NOT_COMPLETED"}'

    def row(stage: str, target: str, family: str) -> dict:
        return {
            "video_id": f"video-{stage}-{family}",
            "task_id": "task",
            "step_index": 0,
            "window_start": 0.0,
            "window_end": 3.0,
            "gt_start": 1.0,
            "gt_end": 2.0,
            "target": target,
            "raw_output": target,
            "label": "NOT_COMPLETED" if '"status"' in target else "STEP_0",
            "metadata": {
                "projection_stage": stage,
                "cohort_family": family,
                "all_instructions": ["step"],
                "late_trigger_augmented": family == "b_late",
            },
        }

    rows = [
        row("A", completed, "a_natural"),
        row("A", negative, "a_natural"),
        row("B", completed, "b_on_time"),
        row("B", completed, "b_late"),
        row("B", negative, "b_sampled_incomplete"),
        *[
            row(
                "C",
                completed.replace('"CORRECT"', f'"{class_name}"'),
                "c_completed_non_addition",
            )
            for class_name in ("CORRECT", "SLIP", "MODIFICATION", "CORRECTION")
        ],
        row("C", negative, "c_sampled_incomplete"),
    ]
    metrics = compute_monolith_metrics(rows)
    assert metrics["eval_monolith/a_total_macro_f1"] == 1.0
    assert metrics["eval_monolith/b_mixed_raw_macro"] == 1.0
    assert metrics["eval_monolith/c_all_class_macro_f1"] == 1.0
    assert metrics["eval_monolith/stage_composite"] == 1.0
    assert metrics["eval_monolith/parse_valid_rate"] == 1.0
