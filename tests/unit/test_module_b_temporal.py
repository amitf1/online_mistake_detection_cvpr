from __future__ import annotations

import json
import sys
from types import SimpleNamespace

from online_mistake_detection.config import VideoSamplingConfig
from online_mistake_detection.metrics.temporal import (
    TemporalPrediction,
    parse_temporal_prediction,
)
from online_mistake_detection.models.video import build_sample_timestamps
from online_mistake_detection.prompts.module_b import build_module_b_prompt
from online_mistake_detection.prompts.time_tokens import render_seconds_span_target
from online_mistake_detection.training.module_a import (
    inference_runtime_metrics,
    trainer_inference_runtime_metrics,
)
from online_mistake_detection.training.module_b import (
    isolated_temporal_metrics,
    restore_module_b_training_state,
    temporal_metrics,
)


def test_restore_module_b_training_state_reenables_memory_safeguards(
    monkeypatch,
) -> None:
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
    text_config = SimpleNamespace(use_cache=True)
    config = SimpleNamespace(use_cache=True, text_config=text_config)

    class FakeModel:
        is_gradient_checkpointing = False

        def __init__(self) -> None:
            self.config = config
            self.checkpointing_enabled = False
            self.training = False

        def gradient_checkpointing_enable(self) -> None:
            self.checkpointing_enabled = True

        def train(self) -> None:
            self.training = True

    model = FakeModel()
    restore_module_b_training_state(model)
    assert calls == [(model, "unsloth")]
    assert model.checkpointing_enabled is True
    assert model.training is True
    assert config.use_cache is False
    assert text_config.use_cache is False


def test_render_seconds_span_target() -> None:
    target = render_seconds_span_target(gt_start=13.2, gt_end=17.8, window_start=10.0)
    parsed = json.loads(target)
    assert parsed == {"start": 3.2, "end": 7.8}


def test_parse_temporal_prediction() -> None:
    assert parse_temporal_prediction(
        '{"relevant_windows":[["3.20","7.80"]]}',
        clip_duration=10,
    ).span == (3.2, 7.8)
    assert parse_temporal_prediction(
        '{"start_time":"3.20","end_time":"7.80"}',
        clip_duration=10,
    ).span == (3.2, 7.8)
    assert parse_temporal_prediction(
        "the span is 3.20 to 7.80 seconds",
        clip_duration=10,
    ).span == (3.2, 7.8)
    assert not parse_temporal_prediction('{"relevant_windows":[]}', clip_duration=10).valid
    assert not parse_temporal_prediction("not completed", clip_duration=10).valid
    assert not parse_temporal_prediction('{"status":"NOT_COMPLETED"}', clip_duration=10).valid
    assert parse_temporal_prediction(
        '{"start":3.2,"end":7.8}',
        clip_duration=10,
    ).span == (3.2, 7.8)


def test_module_b_enforces_physical_clip_bounds() -> None:
    prompt = build_module_b_prompt("pour water", 30.0)
    assert "0.00 <= start < end <= 30.00" in prompt
    assert parse_temporal_prediction(
        '{"relevant_windows":[["0","30"]]}',
        clip_duration=30,
    ).span == (0.0, 30.0)
    clamped = parse_temporal_prediction(
        '{"relevant_windows":[["20","32"]]}',
        clip_duration=30,
    )
    assert clamped.span == (20.0, 30.0)
    assert clamped.raw_span == (20.0, 32.0)
    assert clamped.rejection_reason is None
    assert clamped.was_clamped
    assert clamped.clamp_reason == "out_of_bounds"


def test_module_b_rejects_only_spans_that_collapse_after_clamping() -> None:
    collapsed = parse_temporal_prediction(
        '{"relevant_windows":[["32","34"]]}',
        clip_duration=30,
    )
    assert collapsed.span is None
    assert collapsed.raw_span == (32.0, 34.0)
    assert collapsed.was_clamped
    assert collapsed.rejection_reason == "collapsed_after_clamp"


def test_temporal_metrics() -> None:
    metrics = temporal_metrics(
        targets=[(13.0, 18.0), (20.0, 25.0)],
        predictions=[(13.0, 18.0), None],
    )
    assert metrics["eval_temporal/num_samples"] == 2.0
    assert metrics["eval_temporal/invalid_rate"] == 0.5
    assert metrics["eval_temporal/recall_at_0.5"] == 0.5
    assert metrics["eval_temporal/f1_at_0.5"] == 2 / 3

    mixed_metrics = temporal_metrics(
        targets=[(13.0, 18.0), (20.0, 25.0)],
        predictions=[(13.0, 18.0), None],
        labels=["LOCALIZE", "NO_ACTION"],
    )
    assert mixed_metrics["eval_temporal/num_positive_samples"] == 1.0
    assert mixed_metrics["eval_temporal/num_no_action_samples"] == 1.0
    assert mixed_metrics["eval_temporal/invalid_rate"] == 0.0
    assert mixed_metrics["eval_temporal/no_action_accuracy"] == 1.0
    assert mixed_metrics["eval_temporal/recall_at_0.5"] == 1.0
    assert mixed_metrics["eval_temporal/not_completed/precision"] == 1.0
    assert mixed_metrics["eval_temporal/not_completed/recall"] == 1.0
    assert mixed_metrics["eval_temporal/not_completed/f1"] == 1.0
    assert mixed_metrics["eval_temporal/completed/num_samples"] == 1.0


def test_isolated_temporal_metrics_separates_trigger_cohorts() -> None:
    rows = [
        {
            "prediction": TemporalPrediction(
                1.0,
                3.0,
                raw_start=1.0,
                raw_end=3.0,
            ),
            "target": (1.0, 3.0),
            "label": "LOCALIZE",
            "late_trigger_augmented": False,
        },
        {
            "prediction": TemporalPrediction(None, None, "parse_error"),
            "target": (1.0, 3.0),
            "label": "LOCALIZE",
            "late_trigger_augmented": False,
        },
        {
            "prediction": TemporalPrediction(
                1.0,
                2.0,
                raw_start=1.0,
                raw_end=2.0,
            ),
            "target": (1.0, 3.0),
            "label": "LOCALIZE",
            "late_trigger_augmented": True,
        },
        {
            "prediction": TemporalPrediction(None, None, "no_action"),
            "target": (0.0, 1.0),
            "label": "NO_ACTION",
            "late_trigger_augmented": False,
        },
        {
            "prediction": TemporalPrediction(
                1.0,
                2.0,
                raw_start=1.0,
                raw_end=2.0,
            ),
            "target": (0.0, 1.0),
            "label": "NO_ACTION",
            "late_trigger_augmented": False,
        },
        {
            "prediction": TemporalPrediction(None, None, "parse_error"),
            "target": (0.0, 1.0),
            "label": "NO_ACTION",
            "late_trigger_augmented": False,
        },
    ]

    metrics = isolated_temporal_metrics(rows)

    assert metrics["eval_temporal/on_time/raw/num_samples"] == 2.0
    assert metrics["eval_temporal/on_time/raw/mean_iou"] == 0.5
    assert metrics["eval_temporal/late/raw/num_samples"] == 1.0
    assert metrics["eval_temporal/late/raw/mean_iou"] == 0.5
    assert metrics["eval_temporal/not_completed/num_samples"] == 3.0
    assert metrics["eval_temporal/not_completed/accuracy"] == 1 / 3
    assert metrics["eval_temporal/not_completed/predicted_span_rate"] == 1 / 3
    assert metrics["eval_temporal/not_completed/malformed_rate"] == 1 / 3
    assert "eval_temporal/raw/mean_iou" not in metrics


def test_eval_inference_runtime_metrics_distinguish_latency_and_throughput() -> None:
    timed = inference_runtime_metrics(
        [4.0, 4.0, 6.0, 6.0],
        amortized_seconds=[2.0, 2.0, 3.0, 3.0],
    )
    assert timed["eval_inference/mean_response_seconds"] == 5.0
    assert timed["eval_inference/p95_response_seconds"] == 6.0
    assert timed["eval_inference/mean_seconds_per_example"] == 2.5
    assert timed["eval_inference/examples_per_second_per_rank"] == 0.4

    derived = trainer_inference_runtime_metrics(
        {"eval_runtime": 10.0},
        num_examples=8,
        world_size=2,
        per_device_batch_size=2,
    )
    assert derived["eval_inference/mean_response_seconds"] == 5.0
    assert derived["eval_inference/mean_seconds_per_example"] == 2.5
    assert derived["eval_inference/examples_per_second_per_rank"] == 0.4
    assert "eval_inference/p95_response_seconds" not in derived


def test_module_b_batched_generation_temporarily_uses_left_padding() -> None:
    import torch
    from types import SimpleNamespace

    from online_mistake_detection.training.module_b import (
        ModuleBTemporalMetricsCallback,
    )

    class FakeTokenizer:
        padding_side = "right"

    class FakeProcessor:
        def __init__(self) -> None:
            self.tokenizer = FakeTokenizer()

        def apply_chat_template(self, *args, **kwargs) -> str:
            return "prompt"

        def __call__(self, **kwargs):
            assert self.tokenizer.padding_side == "left"
            return {
                "input_ids": torch.tensor([[0, 1], [2, 3]]),
                "attention_mask": torch.tensor([[0, 1], [1, 1]]),
            }

    processor = FakeProcessor()
    callback = ModuleBTemporalMetricsCallback.__new__(
        ModuleBTemporalMetricsCallback
    )
    callback.enable_thinking = False
    callback.data_collator = SimpleNamespace(
        _base=SimpleNamespace(processor=processor),
        _extract_images_videos_metadata=lambda messages: (
            None,
            None,
            {},
            None,
        ),
    )
    example = {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "prompt"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "{}"}]},
        ]
    }

    batch = callback._generation_inputs([example, example])

    assert batch["attention_mask"].tolist() == [[0, 1], [1, 1]]
    assert processor.tokenizer.padding_side == "right"


def test_downsampled_timestamps_preserve_window_coverage() -> None:
    timestamps = build_sample_timestamps(
        start_time=0.0,
        end_time=10.0,
        config=VideoSamplingConfig(sample_fps=4.0, max_frames=5),
    )
    assert len(timestamps) == 5
    assert timestamps[0] == 0.0
    assert timestamps[-1] > 9.0
    assert timestamps == sorted(timestamps)


def test_zero_shot_eval_rejects_dry_run(monkeypatch) -> None:
    import pytest

    from online_mistake_detection.training.module_b import parse_args

    monkeypatch.setattr(
        "sys.argv",
        ["module_b", "--zero-shot-eval", "--dry-run"],
    )
    with pytest.raises(SystemExit):
        parse_args()


def test_zero_shot_eval_rejects_skipped_generation(monkeypatch) -> None:
    import pytest

    from online_mistake_detection.training.module_b import parse_args

    monkeypatch.setattr(
        "sys.argv",
        ["module_b", "--zero-shot-eval", "--eval-generation-max-samples", "0"],
    )
    with pytest.raises(SystemExit):
        parse_args()


def test_module_b_generation_disables_thinking_by_default(monkeypatch) -> None:
    from online_mistake_detection.training.module_b import parse_args

    monkeypatch.setattr("sys.argv", ["module_b"])
    assert parse_args().generation_enable_thinking is False

    monkeypatch.setattr("sys.argv", ["module_b", "--generation-enable-thinking"])
    assert parse_args().generation_enable_thinking is True


def test_generation_metric_rejects_disabled_generation(monkeypatch) -> None:
    """IoU is parsed out of generated timestamps, so disabling generation leaves the metric permanently absent and no best checkpoint is ever saved."""
    import pytest

    from online_mistake_detection.training.module_b import parse_args

    monkeypatch.setattr(
        "sys.argv",
        [
            "module_b",
            "--metric-for-best-model",
            "eval_temporal/clamped/mean_iou",
            "--eval-generation-max-samples",
            "0",
        ],
    )
    with pytest.raises(SystemExit):
        parse_args()


def test_loss_metric_allows_disabled_generation(monkeypatch) -> None:
    from online_mistake_detection.training.module_b import parse_args

    monkeypatch.setattr(
        "sys.argv",
        [
            "module_b",
            "--metric-for-best-model",
            "eval_loss",
            "--eval-generation-max-samples",
            "0",
        ],
    )
    args = parse_args()
    assert args.metric_for_best_model == "eval_loss"
