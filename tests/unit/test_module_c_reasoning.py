from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from online_mistake_detection.config import ModuleCConfig
from online_mistake_detection.schemas import Segment, VideoRecord
from online_mistake_detection.training.module_c import (
    apply_module_c_history_dropout,
    enable_module_c_training_hidden_states,
    module_c_candidate_target,
    module_c_multiclass_metrics,
    module_c_reasoning_metrics,
    module_c_response_only_loss,
    parse_module_c_prediction,
    restore_module_c_training_state,
)
from online_mistake_detection.windowing.module_c import build_module_c_examples


class FakeProvider:
    def __init__(self, records: list[VideoRecord]) -> None:
        self.records = records

    def iter_video_records(self) -> list[VideoRecord]:
        return self.records


def test_module_c_candidate_target_is_compact_and_deterministic() -> None:
    target = module_c_candidate_target("MODIFICATION")
    assert target == '{"classification":"MODIFICATION","reasoning":""}'
    assert json.loads(target)["classification"] == "MODIFICATION"


def test_module_c_forced_choice_scorer_returns_finite_nlls() -> None:
    import torch

    from online_mistake_detection.training.module_c import (
        ModuleCMistakeMetricsCallback,
    )

    class FakeCollator:
        label_ids = {
            "CORRECT": 1,
            "CORRECTION": 2,
            "MODIFICATION": 3,
            "SLIP": 4,
        }

        def __call__(self, examples):
            target = json.loads(examples[0]["messages"][-1]["content"][0]["text"])
            token_id = self.label_ids[target["classification"]]
            return {
                "input_ids": torch.tensor([[0, token_id]]),
                "attention_mask": torch.tensor([[1, 1]]),
                "labels": torch.tensor([[-100, token_id]]),
            }

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.marker = torch.nn.Parameter(torch.zeros(()))
            self.output_head = torch.nn.Identity()

        def get_output_embeddings(self):
            return self.output_head

        def forward(self, input_ids, attention_mask, **_kwargs):
            hidden = torch.zeros((1, 2, 5))
            hidden[0, 0, int(input_ids[0, 1])] = 5.0
            return SimpleNamespace(hidden_states=(hidden,))

    callback = ModuleCMistakeMetricsCallback.__new__(
        ModuleCMistakeMetricsCallback
    )
    callback.data_collator = FakeCollator()
    example = {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "classify"}]},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": module_c_candidate_target("CORRECT")}],
            },
        ]
    }
    scores = callback._score_class_nlls(
        FakeModel(),
        example,
        ["CORRECT", "CORRECTION", "MODIFICATION", "SLIP"],
    )
    assert set(scores) == {"CORRECT", "CORRECTION", "MODIFICATION", "SLIP"}
    assert all(0.0 <= value < 0.1 for value in scores.values())


def test_restore_module_c_training_state_reenables_memory_safeguards(
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
    restore_module_c_training_state(model)
    assert calls == [(model, "unsloth")]
    assert model.checkpointing_enabled is True
    assert model.training is True
    assert config.use_cache is False
    assert text_config.use_cache is False


def test_module_c_response_loss_projects_only_labeled_tokens() -> None:
    import torch

    class RecordingHead(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(7, 4))
            self.seen_shapes: list[tuple[int, ...]] = []

        def forward(self, hidden):
            self.seen_shapes.append(tuple(hidden.shape))
            return torch.nn.functional.linear(hidden, self.weight)

    class FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.marker = torch.nn.Parameter(torch.zeros(()))
            self.head = RecordingHead()

        def get_output_embeddings(self):
            return self.head

        def forward(self, input_ids, **kwargs):
            assert kwargs["output_hidden_states"] is True
            assert kwargs["use_cache"] is False
            hidden = torch.randn(input_ids.shape[0], input_ids.shape[1], 4)
            return SimpleNamespace(hidden_states=(hidden,))

    model = FakeModel()
    loss = module_c_response_only_loss(
        model,
        {
            "input_ids": torch.tensor([[0, 1, 2, 3, 4]]),
            "labels": torch.tensor([[-100, -100, -100, 2, 3]]),
        },
    )
    assert torch.isfinite(loss)
    assert model.head.seen_shapes == [(2, 4)]


def test_enable_module_c_training_hidden_states(monkeypatch) -> None:
    monkeypatch.setenv("UNSLOTH_RETURN_LOGITS", "1")
    monkeypatch.setenv("UNSLOTH_RETURN_HIDDEN_STATES", "0")
    enable_module_c_training_hidden_states()
    assert os.environ["UNSLOTH_RETURN_LOGITS"] == "0"
    assert os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] == "1"


def build_record() -> VideoRecord:
    return VideoRecord(
        dataset="egoper",
        task_id="task",
        video_id="video",
        video_path=Path("/tmp/fake.mp4"),
        instructions=["pick up the red block", "place it on the table"],
        segments=[
            Segment(
                start=1.0,
                end=3.0,
                instruction_index=0,
                instruction="pick up the red block",
                is_mistake=False,
                raw={"labels": []},
            ),
            Segment(
                start=4.0,
                end=6.0,
                instruction_index=1,
                instruction="place it on the table",
                is_mistake=True,
                error_label="Slip",
                caption="The person places the wrong object on the table.",
                event_type="MISTAKE",
                fine_class="SLIP",
                aggregate_class="MISTAKE",
                observed_description="The person places the wrong object on the table.",
                raw={"labels": [0]},
            ),
            Segment(
                start=7.0,
                end=8.0,
                instruction_index=-1,
                instruction="undefined / extra step",
                is_mistake=True,
                error_label="Addition",
                caption="",
                event_type="MISTAKE",
                fine_class="ADDITION",
                aggregate_class="MISTAKE",
                raw={"labels": [5]},
            ),
        ],
    )


EGOPER_TAXONOMY = ("CORRECT", "CORRECTION", "MODIFICATION", "SLIP")
EGOPER_MAPPING = {
    "CORRECT": "CORRECT",
    "CORRECTION": "CORRECT",
    "MODIFICATION": "MISTAKE",
    "SLIP": "MISTAKE",
    "ADDITION": "MISTAKE",
}


def build_dataset() -> list:
    return build_module_c_examples(
        [build_record()],
        fine_taxonomy=EGOPER_TAXONOMY,
        aggregate_mapping=EGOPER_MAPPING,
        config=ModuleCConfig(
            min_duration_seconds=0.5,
            jitter_ratio_min=0.05,
            jitter_ratio_max=0.10,
            include_not_completed_examples=False,
            history_max_events=12,
            seed=123,
        ),
    )


def build_egoper_dataset() -> list:
    source = build_record()
    record = replace(
        source,
        dataset="egoper",
        segments=[
            source.segments[0],
            replace(
                source.segments[2],
                start=3.2,
                end=3.8,
                event_type="MISTAKE",
                fine_class="ADDITION",
                aggregate_class="MISTAKE",
            ),
            replace(
                source.segments[1],
                fine_class="SLIP",
                aggregate_class="MISTAKE",
            ),
        ],
    )
    return build_module_c_examples(
        [record],
        fine_taxonomy=("CORRECT", "CORRECTION", "MODIFICATION", "SLIP"),
        aggregate_mapping={
            "CORRECT": "CORRECT",
            "CORRECTION": "CORRECT",
            "MODIFICATION": "MISTAKE",
            "SLIP": "MISTAKE",
        },
        config=ModuleCConfig(
            min_duration_seconds=0.5,
            jitter_ratio_min=0.05,
            jitter_ratio_max=0.10,
            include_not_completed_examples=False,
            history_max_events=12,
            seed=123,
        ),
    )


def test_module_c_target_json_and_reasoning() -> None:
    dataset = build_dataset()
    assert len(dataset) == 2

    correct = dataset[0]
    correct_target = json.loads(correct.target_text)
    assert list(correct_target) == ["classification", "reasoning"]
    assert correct_target["classification"] == "CORRECT"
    assert correct_target["reasoning"] == (
        "The visible action follows the instructed step without a procedural deviation."
    )
    assert correct.label == "CORRECT"

    mistake = dataset[1]
    mistake_target = json.loads(mistake.target_text)
    assert mistake_target == {
        "classification": "SLIP",
        "reasoning": "The person places the wrong object on the table.",
    }
    assert mistake.fine_class == "SLIP"
    assert mistake.aggregate_class == "MISTAKE"
    assert "Recent prior action history" in mistake.prompt_text
    assert "Classification: CORRECT" in mistake.prompt_text
    assert "pick up the red block" in mistake.prompt_text
    assert mistake.metadata["module_c_history_mode"] == "oracle"


def test_module_c_history_keeps_only_last_twelve_events() -> None:
    from online_mistake_detection.prompts.module_c import append_module_c_history

    history = [
        {
            "action": f"action {index}",
            "classification": "CORRECT",
            "evidence": f"evidence {index}",
        }
        for index in range(14)
    ]
    prompt = append_module_c_history("current", history, max_events=12)

    assert "action 0 |" not in prompt
    assert "action 1 |" not in prompt
    assert "action 2 |" in prompt
    assert "action 13 |" in prompt


def test_module_c_history_dropout_is_training_only_transform() -> None:
    original = build_egoper_dataset()[1]
    assert "Classification: CORRECT" in original.prompt_text
    assert "ADDITION" not in original.prompt_text
    dropped = apply_module_c_history_dropout(
        [original],
        probability=1.0,
        max_events=12,
        seed=3407,
    )[0]

    assert "unavailable; classify from the current instruction and video" in dropped.prompt_text
    assert dropped.metadata["module_c_history_mode"] == "dropped"
    assert "Classification: CORRECT" in original.prompt_text
    assert original.metadata["module_c_history_mode"] == "oracle"


def test_non_correct_oversampling_excludes_correct_and_not_completed() -> None:
    from collections import Counter
    from types import SimpleNamespace

    from online_mistake_detection.training.module_c import (
        oversample_non_correct_examples,
    )

    examples = [
        SimpleNamespace(fine_class="CORRECT"),
        SimpleNamespace(fine_class="CORRECTION"),
        SimpleNamespace(fine_class="MODIFICATION"),
        SimpleNamespace(fine_class="SLIP"),
        SimpleNamespace(fine_class="NOT_COMPLETED"),
    ]
    sampled = oversample_non_correct_examples(examples, factor=4.0, seed=3407)
    counts = Counter(example.fine_class for example in sampled)
    assert counts == {
        "CORRECT": 1,
        "CORRECTION": 4,
        "MODIFICATION": 4,
        "SLIP": 4,
        "NOT_COMPLETED": 1,
    }


def test_module_c_prompt_mentions_mistake_and_json() -> None:
    dataset = build_dataset()
    prompt = dataset[0].prompt_text
    assert "Allowed outcomes" in prompt
    assert "CORRECTION" in prompt
    assert "NOT_COMPLETED" in prompt


def test_module_c_jitter_keeps_valid_windows() -> None:
    dataset = build_dataset()
    for example in dataset:
        assert 0.0 <= example.window_start < example.window_end <= 10.0
        assert example.window_end - example.window_start >= 0.5
        assert example.gt_start is not None
        assert example.gt_end is not None
        assert example.window_start <= example.gt_end
        assert example.window_end >= example.gt_start


def test_parse_module_c_prediction() -> None:
    parsed = parse_module_c_prediction(
        '{"classification":"slip","reasoning":"spills the water"}'
    )
    assert parsed is not None
    assert parsed["classification"] == "SLIP"
    assert parsed["mistake"] is True
    correction = parse_module_c_prediction(
        '{"classification":"correction","reasoning":"replaces the wrong block"}'
    )
    assert correction is not None
    assert correction["mistake"] is False
    legacy = '{"mistake":true,"reasoning":"uses the wrong object"}'
    assert parse_module_c_prediction(legacy) is None
    assert parse_module_c_prediction(
        legacy,
        allow_legacy_binary=True,
    )["legacy"] is True
    assert parse_module_c_prediction('{"classification":"CORRECT"}') is None
    assert parse_module_c_prediction("not json") is None
    not_completed = parse_module_c_prediction('{"status":"NOT_COMPLETED"}')
    assert not_completed is not None
    assert not_completed["classification"] == "NOT_COMPLETED"
    assert not_completed["reasoning"] is None
    # The task requires JSON; mentioning the words in malformed prose must not
    # count as a correct negative prediction.
    assert parse_module_c_prediction("The action is not completed.") is None
    assert parse_module_c_prediction('["NOT_COMPLETED"]') is None


def test_module_c_metrics() -> None:
    metrics = module_c_multiclass_metrics(
        ["SLIP", "MODIFICATION", "CORRECTION", "CORRECT"],
        ["SLIP", None, "MODIFICATION", "CORRECT"],
    )
    assert metrics["eval_fine/invalid_rate"] == 0.25
    assert metrics["eval_fine/SLIP/f1"] == 1.0
    assert "eval_aggregate/MISTAKE/f1" in metrics
    assert "eval_aggregate/CORRECTION/f1" in metrics
    assert "eval_aggregate/CORRECT/f1" in metrics
    assert "eval_all/macro_f1" in metrics


def test_module_c_reasoning_metrics_include_eval_only_bertscore(monkeypatch) -> None:
    calls: list[tuple[list[str], list[str], dict]] = []

    class FakeScores:
        def __init__(self, value: float) -> None:
            self.value = value

        def mean(self) -> "FakeScores":
            return self

        def item(self) -> float:
            return self.value

    def fake_score(
        candidates: list[str],
        references: list[str],
        **kwargs,
    ) -> tuple[tuple[FakeScores, FakeScores, FakeScores], str]:
        calls.append((candidates, references, kwargs))
        return (
            (FakeScores(0.7), FakeScores(0.8), FakeScores(0.75)),
            "fake-hash",
        )

    monkeypatch.setitem(sys.modules, "bert_score", SimpleNamespace(score=fake_score))
    metrics, score_hash = module_c_reasoning_metrics(
        ["used the red cup", "spilled water", "empty target"],
        ["picked up the red cup", None, ""],
        compute_bertscore=True,
        bertscore_model_type="test-encoder",
        bertscore_batch_size=4,
        bertscore_device="cpu",
    )

    assert metrics["eval_reasoning/nonempty_rate"] == 1 / 3
    assert metrics["eval_reasoning/bertscore_num_scored"] == 1.0
    assert metrics["eval_reasoning/bertscore_precision"] == 0.7
    assert metrics["eval_reasoning/bertscore_recall"] == 0.8
    assert metrics["eval_reasoning/bertscore_f1"] == 0.75
    assert score_hash == "fake-hash"
    assert calls == [(
        ["picked up the red cup"],
        ["used the red cup"],
        {
            "model_type": "test-encoder",
            "batch_size": 4,
            "device": "cpu",
            "verbose": False,
            "return_hash": True,
        },
    )]


def test_module_c_reasoning_metrics_skip_bertscore_during_training(
    monkeypatch,
) -> None:
    def fail_if_called(*args, **kwargs):
        raise AssertionError("BERTScore must not run during epoch evaluation")

    monkeypatch.setitem(
        sys.modules,
        "bert_score",
        SimpleNamespace(score=fail_if_called),
    )
    metrics, score_hash = module_c_reasoning_metrics(
        ["same words", "reference"],
        ["same words", None],
        compute_bertscore=False,
    )

    assert metrics["eval_reasoning/nonempty_rate"] == 0.5
    assert metrics["eval_reasoning/lexical_overlap"] == 1.0
    assert not any("bertscore" in key for key in metrics)
    assert score_hash is None


def test_module_c_reasoning_metrics_handle_no_valid_predictions() -> None:
    metrics, score_hash = module_c_reasoning_metrics(
        ["reference", ""],
        [None, ""],
        compute_bertscore=True,
    )

    assert metrics["eval_reasoning/nonempty_rate"] == 0.0
    assert metrics["eval_reasoning/bertscore_num_scored"] == 0.0
    assert metrics["eval_reasoning/bertscore_f1"] == 0.0
    assert score_hash is None


def test_module_c_batched_generation_temporarily_uses_left_padding() -> None:
    from types import SimpleNamespace

    import torch

    from online_mistake_detection.training.module_c import (
        ModuleCMistakeMetricsCallback,
    )

    class FakeTokenizer:
        padding_side = "right"

    class FakeProcessor:
        def __init__(self) -> None:
            self.tokenizer = FakeTokenizer()

        def apply_chat_template(self, *args, **kwargs) -> str:
            assert kwargs["enable_thinking"] is False
            return "prompt"

        def __call__(self, **kwargs):
            assert self.tokenizer.padding_side == "left"
            return {
                "input_ids": torch.tensor([[0, 1], [2, 3]]),
                "attention_mask": torch.tensor([[0, 1], [1, 1]]),
            }

    processor = FakeProcessor()
    callback = ModuleCMistakeMetricsCallback.__new__(
        ModuleCMistakeMetricsCallback
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


def test_zero_shot_eval_rejects_generation_eval_only(monkeypatch) -> None:
    import pytest

    from online_mistake_detection.training.module_c import parse_args

    monkeypatch.setattr(
        "sys.argv",
        ["module_c", "--zero-shot-eval", "--generation-eval-only"],
    )
    with pytest.raises(SystemExit):
        parse_args()


def test_generation_eval_only_allows_base_model(monkeypatch) -> None:
    from online_mistake_detection.training.module_c import parse_args

    monkeypatch.setattr(
        "sys.argv",
        ["module_c", "--generation-eval-only"],
    )
    args = parse_args()
    assert args.generation_eval_only is True
    assert args.eval_checkpoint is None


def test_stale_metric_name_is_rejected(monkeypatch) -> None:
    """eval_mistake/f1 was renamed to deprecated_eval_mistake/f1 by the multiclass metrics path, so the old name silently disabled checkpoint selection for every run that still named it."""
    import pytest

    from online_mistake_detection.training.module_c import parse_args

    monkeypatch.setattr(
        "sys.argv",
        ["module_c", "--metric-for-best-model", "eval_mistake/f1"],
    )
    with pytest.raises(SystemExit):
        parse_args()


def test_generation_metric_rejects_disabled_generation(monkeypatch) -> None:
    import pytest

    from online_mistake_detection.training.module_c import parse_args

    monkeypatch.setattr(
        "sys.argv",
        [
            "module_c",
            "--metric-for-best-model",
            "eval_fine/present_class_macro_f1",
            "--eval-generation-max-samples",
            "0",
        ],
    )
    with pytest.raises(SystemExit):
        parse_args()


def test_loss_metric_allows_disabled_generation(monkeypatch) -> None:
    """eval_loss comes from the trainer itself, so it stays valid without generation."""
    from online_mistake_detection.training.module_c import parse_args

    monkeypatch.setattr(
        "sys.argv",
        [
            "module_c",
            "--metric-for-best-model",
            "eval_loss",
            "--eval-generation-max-samples",
            "0",
        ],
    )
    args = parse_args()
    assert args.metric_for_best_model == "eval_loss"
