from __future__ import annotations

from pathlib import Path

from online_mistake_detection.config import ModuleAConfig, ModuleBConfig
from online_mistake_detection.datasets import build_module_b_training_dataset
from online_mistake_detection.datasets.labels import (
    EXTRA_STEP_INSTRUCTION,
    is_module_a_completion_label,
    module_a_instruction_for_step_id_letter,
    module_a_step_id_letter_for_instruction_index,
)
from online_mistake_detection.schemas import Segment, VideoRecord
from online_mistake_detection.training.module_a import extract_module_a_label


def fake_record(
    *,
    instructions: list[str],
    segments: list[Segment],
    video_id: str = "vid",
    task_id: str = "tea",
) -> VideoRecord:
    return VideoRecord(
        dataset="egoper",
        task_id=task_id,
        video_id=video_id,
        video_path=Path("/tmp/fake_missing.mp4"),
        instructions=instructions,
        segments=segments,
    )


def test_module_b_uses_module_a_window_with_optional_late_trigger() -> None:
    record = fake_record(
        instructions=["step zero", "step one"],
        segments=[
            Segment(0.0, 17.6, 0, "step zero", False),
            Segment(17.6, 30.0, 1, "step one", False),
        ],
    )

    class Provider:
        def iter_video_records(self):
            return iter([record])

    module_a_config = ModuleAConfig(stride_seconds=5.0, label_mode="step_id")
    normal = build_module_b_training_dataset(
        Provider(),
        module_a_config=module_a_config,
        config=ModuleBConfig(late_trigger_probability=0.0),
    )
    delayed = build_module_b_training_dataset(
        Provider(),
        module_a_config=module_a_config,
        config=ModuleBConfig(late_trigger_probability=1.0),
    )
    assert (normal[0].window_start, normal[0].window_end) == (0.0, 20.0)
    assert (delayed[0].window_start, delayed[0].window_end) == (0.0, 25.0)
    assert normal[0].target_text == delayed[0].target_text
    assert normal[1].window_start == delayed[1].window_start == 17.6


def test_letter_parser_and_instruction_mapping() -> None:
    assert extract_module_a_label("C", label_mode="step_id") == "C"
    assert extract_module_a_label("  b. EXTRA", label_mode="step_id") == "B"
    assert extract_module_a_label("D", label_mode="step_id") == "D"
    assert extract_module_a_label("D\n", label_mode="step_id") == "D"
    assert extract_module_a_label("COMPLETE", label_mode="legacy") == "COMPLETE"
    assert extract_module_a_label("please WAIT", label_mode="legacy") == "WAIT"
    assert module_a_step_id_letter_for_instruction_index(-1) == "B"
    assert module_a_step_id_letter_for_instruction_index(0) == "C"
    assert module_a_step_id_letter_for_instruction_index(1) == "D"
    instructions = ("alpha", "beta")
    assert module_a_instruction_for_step_id_letter("A", instructions) is None
    assert module_a_instruction_for_step_id_letter("B", instructions) == EXTRA_STEP_INSTRUCTION
    assert module_a_instruction_for_step_id_letter("C", instructions) == "alpha"
    assert module_a_instruction_for_step_id_letter("D", instructions) == "beta"
    assert is_module_a_completion_label("A", label_mode="step_id") is False
    assert is_module_a_completion_label("C", label_mode="step_id") is True
    assert is_module_a_completion_label("COMPLETE", label_mode="legacy") is True
    assert is_module_a_completion_label("WAIT", label_mode="legacy") is False


def test_multiclass_f1_and_confusion_and_auc() -> None:
    from online_mistake_detection.training.module_a import (
        binary_roc_auc,
        confusion_matrix_from_metrics,
        multiclass_letter_metrics,
        scalar_metrics_for_logging,
    )

    targets = ["A", "A", "C", "C", "B"]
    preds = ["A", "C", "C", "A", "B"]
    metrics = multiclass_letter_metrics(targets, preds, prefix="eval_step_id")
    assert "eval_step_id/macro_f1" in metrics
    assert "eval_step_id/micro_f1" in metrics
    assert "eval_step_id/weighted_f1" in metrics
    assert "eval_step_id/gate/f1" in metrics
    assert metrics["eval_step_id/f1_B"] == 1.0
    assert not any("/cm_" in key for key in metrics)
    matrix = confusion_matrix_from_metrics(metrics, prefix="eval_step_id")
    assert matrix["labels"] == ["A", "B", "C"]
    assert matrix["matrix"][0][0] == 1  # A->A
    logged = scalar_metrics_for_logging(metrics)
    assert "confusion_matrix" not in logged
    assert abs(binary_roc_auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) - 0.75) < 1e-6
    assert binary_roc_auc([1, 1, 1], [0.2, 0.3, 0.9]) is None


def test_step_id_semantic_metrics_per_task_then_mean() -> None:
    from online_mistake_detection.training.module_a import (
        SEMANTIC_EXTRA,
        SEMANTIC_NONE,
        scalar_metrics_for_logging,
        step_id_letter_to_semantic_class,
        step_id_semantic_metrics,
    )

    assert (
        step_id_letter_to_semantic_class(
            "C",
            task_id="blacklight",
            all_instructions=["alpha", "beta"],
        )
        == "STEP_0"
    )
    assert (
        step_id_letter_to_semantic_class(
            "A",
            task_id="blacklight",
            all_instructions=["alpha"],
        )
        == SEMANTIC_NONE
    )
    assert (
        step_id_letter_to_semantic_class(
            "B",
            task_id="blacklight",
            all_instructions=["alpha"],
        )
        == SEMANTIC_EXTRA
    )

    # Task A: perfect NONE/EXTRA/STEP_0 over catalog [NONE, EXTRA, STEP_0]
    # Task B: confuse STEP_0 with NONE
    task_ids = ["task_a", "task_a", "task_a", "task_b", "task_b", "task_b"]
    instructions = [
        ["only"],
        ["only"],
        ["only"],
        ["first", "second"],
        ["first", "second"],
        ["first", "second"],
    ]
    targets = ["A", "B", "C", "A", "C", "D"]
    preds = ["A", "B", "C", "C", "A", "D"]
    metrics = step_id_semantic_metrics(
        task_ids=task_ids,
        all_instructions=instructions,
        targets=targets,
        predictions=preds,
        prefix="eval_step_id",
    )
    assert metrics["eval_step_id/task_task_a/macro_f1"] == 1.0
    assert metrics["eval_step_id/task_task_a/f1_NONE"] == 1.0
    assert metrics["eval_step_id/task_task_a/f1_EXTRA"] == 1.0
    assert metrics["eval_step_id/task_task_a/f1_STEP_0"] == 1.0
    # Task B has no EXTRA target. Its diagnostic F1 remains zero, but the
    # unsupported class is excluded from the task macro.
    assert metrics["eval_step_id/task_task_b/f1_EXTRA"] == 0.0
    assert metrics["eval_step_id/task_task_b/num_classes"] == 4.0
    assert metrics["eval_step_id/task_task_b/num_supported_classes"] == 3.0
    assert 0.0 < metrics["eval_step_id/task_task_b/macro_f1"] < 1.0
    assert metrics["eval_step_id/task_task_b/f1_STEP_1"] == 1.0
    expected_total = (
        metrics["eval_step_id/task_task_a/macro_f1"]
        + metrics["eval_step_id/task_task_b/macro_f1"]
    ) / 2.0
    assert abs(metrics["eval_step_id/total_macro_f1"] - expected_total) < 1e-9
    assert abs(metrics["eval_step_id/macro_f1"] - expected_total) < 1e-9
    assert "eval_step_id/trigger/f1" in metrics
    assert "eval_step_id/gate/f1" in metrics
    assert "task_a" in metrics["task_confusion_matrices"]
    cm_a = metrics["task_confusion_matrices"]["task_a"]
    assert cm_a["label_keys"] == ["NONE", "EXTRA", "STEP_0"]
    assert cm_a["labels"] == ["NONE", "EXTRA", "only"]
    # Flat CM / count scalars must not flood W&B during training eval.
    assert not any("/cm_" in key for key in metrics)
    assert "eval_step_id/task_task_a/tp_NONE" not in metrics
    assert "eval_step_id/task_task_a/support_NONE" not in metrics
    logged = scalar_metrics_for_logging(metrics)
    assert "task_confusion_matrices" not in logged
    assert "eval_step_id/total_macro_f1" in logged


def test_write_step_id_confusion_heatmaps(tmp_path) -> None:
    from online_mistake_detection.training.module_a import write_step_id_confusion_heatmaps

    task_confusion = {
        "task_a": {
            "labels": ["NONE", "EXTRA", "only"],
            "label_keys": ["NONE", "EXTRA", "STEP_0"],
            "matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        }
    }
    figures_dir = tmp_path / "figures"
    written = write_step_id_confusion_heatmaps(task_confusion, figures_dir)
    assert len(written) == 1
    assert written[0].exists()
    assert written[0].suffix == ".png"


def test_generation_eval_command_runs_as_module_with_dataset() -> None:
    from argparse import Namespace
    from pathlib import Path

    from online_mistake_detection.training.module_a import build_generation_eval_command

    args = Namespace(
        dataset="egoper",
        egoper_root="/data/EgoPER",
        max_videos=389,
        fps=1.0,
        min_frames=2,
        max_frames=54,
        stride_seconds=5.0,
        completion_margin=0.1,
        negative_to_positive_ratio=2,
        keep_last_wait_windows=2,
        max_negative_windows_per_episode=3,
        max_seq_length=25600,
        vision_resize=0,
        eval_generation_max_samples=-1,
        eval_generation_max_new_tokens=8,
        fbeta_beta=2.0,
        module_a_label_mode="step_id",
        output_dir="/tmp/out",
        seed=3407,
        split_file="configs/splits/egoper_video_split_seed3407.json",
        video_ids=None,
        task_ids=None,
        load_in_4bit=False,
        load_in_8bit=False,
        load_in_16bit=True,
        label_score_eval=True,
    )
    command = build_generation_eval_command(
        args=args,
        checkpoint_path=Path("/tmp/latest"),
        output_json=Path("/tmp/eval.json"),
    )
    assert command[1:3] == ["-m", "online_mistake_detection.training.module_a"]
    assert "--dataset" in command and command[command.index("--dataset") + 1] == "egoper"
    assert "--egoper-root" in command
    assert str(Path(__file__).resolve()) not in command


def test_shard_and_merge_round_robin_preserves_order() -> None:
    from online_mistake_detection.training.distributed_eval import (
        merge_round_robin_shards,
        shard_examples_for_rank,
    )

    examples = list(range(10))
    shards = [
        shard_examples_for_rank(examples, rank=rank, world_size=8)
        for rank in range(8)
    ]
    assert shards[0] == [0, 8]
    assert shards[1] == [1, 9]
    assert shards[7] == [7]
    assert merge_round_robin_shards(shards) == examples
    assert shard_examples_for_rank(examples, rank=0, world_size=1) == examples


def test_unwrap_trainer_model_peels_ddp_module() -> None:
    from types import SimpleNamespace

    from online_mistake_detection.training.distributed_eval import unwrap_trainer_model

    inner = object()
    assert unwrap_trainer_model(None, SimpleNamespace(module=inner)) is inner
    assert unwrap_trainer_model(None, inner) is inner

    class FakeAccelerator:
        def unwrap_model(self, model):
            return model.module

    trainer = SimpleNamespace(accelerator=FakeAccelerator())
    assert unwrap_trainer_model(trainer, SimpleNamespace(module=inner)) is inner


def test_disable_dynamo_lru_cache_is_safe_without_torch() -> None:
    from online_mistake_detection.training.module_a import disable_dynamo_lru_cache

    assert disable_dynamo_lru_cache() in {True, False}


def test_disable_dynamo_lru_cache_calls_setter() -> None:
    import sys
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from online_mistake_detection.training import module_a

    set_lru = MagicMock()
    fake_torch = SimpleNamespace(
        _C=SimpleNamespace(_dynamo=SimpleNamespace(eval_frame=SimpleNamespace(_set_lru_cache=set_lru)))
    )
    original = sys.modules.get("torch")
    sys.modules["torch"] = fake_torch
    try:
        assert module_a.disable_dynamo_lru_cache() is True
        set_lru.assert_called_once_with(False)
    finally:
        if original is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = original


def test_logits_for_loss_unwraps_distributed_data_parallel() -> None:
    from types import SimpleNamespace

    import pytest

    torch = pytest.importorskip("torch")
    from online_mistake_detection.training.module_a import logits_for_loss

    class Inner:
        def __init__(self) -> None:
            self._lm_head = torch.nn.Linear(4, 8, bias=False)

        def get_output_embeddings(self):
            return self._lm_head

    wrapped = SimpleNamespace(module=Inner())
    vocab_logits = torch.randn(2, 3, 8)
    returned = logits_for_loss(
        wrapped,
        SimpleNamespace(logits=vocab_logits, hidden_states=None),
    )
    assert returned is vocab_logits

    projected = logits_for_loss(
        wrapped,
        SimpleNamespace(logits=torch.randn(2, 3, 4), hidden_states=None),
    )
    assert projected.shape == (2, 3, 8)


def test_response_token_loss_projects_only_labeled_tokens() -> None:
    from types import SimpleNamespace

    import pytest

    torch = pytest.importorskip("torch")
    from online_mistake_detection.training.module_a import response_token_loss

    class RecordingHead(torch.nn.Linear):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)
            self.last_input_shape: tuple[int, ...] | None = None

        def forward(self, hidden: torch.Tensor) -> torch.Tensor:
            self.last_input_shape = tuple(hidden.shape)
            return super().forward(hidden)

    class Inner:
        def __init__(self, head: torch.nn.Module) -> None:
            self._lm_head = head

        def get_output_embeddings(self):
            return self._lm_head

    torch.manual_seed(0)
    head = RecordingHead(4, 8, bias=False)
    hidden = torch.randn(1, 6, 4)
    labels = torch.full((1, 6), -100, dtype=torch.long)
    labels[0, 4] = 1
    labels[0, 5] = 2
    weights = torch.zeros(1, 6)
    weights[0, 4] = 1.0
    weights[0, 5] = 2.0
    hidden_states = [torch.randn(1, 6, 4), hidden]

    loss = response_token_loss(
        Inner(head),
        SimpleNamespace(logits=None, hidden_states=hidden_states),
        labels,
        token_weights=weights,
    )
    assert head.last_input_shape == (2, 4)
    assert hidden_states == [hidden]

    selected = hidden[:, :-1][labels[:, 1:] != -100]
    logits = torch.nn.functional.linear(selected, head.weight)
    expected_nll = torch.nn.functional.cross_entropy(
        logits.float(),
        torch.tensor([1, 2]),
        reduction="none",
    )
    expected = (expected_nll * torch.tensor([1.0, 2.0])).sum() / 3.0
    assert torch.allclose(loss, expected, atol=1e-5)


def test_response_token_loss_rejects_fully_truncated_response() -> None:
    from types import SimpleNamespace

    import pytest

    torch = pytest.importorskip("torch")
    from online_mistake_detection.training.module_a import response_token_loss

    class FakeModel:
        def __init__(self) -> None:
            self.head = torch.nn.Linear(4, 8, bias=False)

        def get_output_embeddings(self) -> torch.nn.Module:
            return self.head

    labels = torch.full((1, 6), -100, dtype=torch.long)
    outputs = SimpleNamespace(
        logits=None,
        hidden_states=[torch.randn(1, 6, 4)],
    )
    with pytest.raises(RuntimeError, match="right-truncated"):
        response_token_loss(FakeModel(), outputs, labels)


def test_candidate_labels_uses_all_instructions_not_prompt_text() -> None:
    """Scorer must enumerate every step letter from all_instructions."""
    from online_mistake_detection.training.module_a import (
        ModuleADetectionMetricsCallback,
    )

    callback = ModuleADetectionMetricsCallback.__new__(ModuleADetectionMetricsCallback)
    callback.label_score_eval = True
    callback.eval_examples = []

    # Three instructions → letters A (none), B (extra), C, D, E
    example = {
        "label_mode": "step_id",
        "all_instructions": ["step one", "step two", "step three"],
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Task: Test\n"
                            "Choices:\n"
                            "A: NONE\n"
                            "B: EXTRA\n"
                            "C: STEP_0 — step one\n"
                            "D: STEP_1 — step two\n"
                            "E: STEP_2 — step three\n"
                        ),
                    }
                ],
            }
        ],
    }
    candidates = callback._candidate_labels_for_example(example)
    assert candidates == ["A", "B", "C", "D", "E"], candidates

    # No instructions → only A, B
    example_no_steps = {**example, "all_instructions": []}
    assert callback._candidate_labels_for_example(example_no_steps) == ["A", "B"]


def test_label_scoring_uses_one_forward_for_all_candidates() -> None:
    from types import SimpleNamespace

    import pytest

    torch = pytest.importorskip("torch")
    from online_mistake_detection.training.module_a import (
        ModuleADetectionMetricsCallback,
    )

    class FakeTokenizer:
        token_ids = {"A": 10, "B": 11, "C": 12}

        def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
            assert not add_special_tokens
            return [self.token_ids[text]]

    class FakeCollator:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, examples: list[dict[str, object]]) -> dict[str, torch.Tensor]:
            self.calls += 1
            assistant = next(
                message
                for message in examples[0]["messages"]
                if message["role"] == "assistant"
            )
            assert assistant["content"][0]["text"] == "A"
            labels = torch.full((1, 6), -100, dtype=torch.long)
            labels[0, 4] = 10
            labels[0, 5] = 1
            return {
                "input_ids": torch.arange(6).unsqueeze(0),
                "labels": labels,
            }

    class FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lm_head = torch.nn.Linear(4, 20, bias=False)
            self.forward_calls = 0
            with torch.no_grad():
                self.lm_head.weight.zero_()
                self.lm_head.weight[10, 0] = 1.0
                self.lm_head.weight[11, 0] = 3.0
                self.lm_head.weight[12, 0] = 2.0

        def get_output_embeddings(self) -> torch.nn.Module:
            return self.lm_head

        def forward(self, **_: object) -> SimpleNamespace:
            self.forward_calls += 1
            hidden = torch.zeros(1, 6, 4)
            hidden[0, 3, 0] = 1.0
            return SimpleNamespace(hidden_states=[hidden], logits=None)

    callback = ModuleADetectionMetricsCallback.__new__(ModuleADetectionMetricsCallback)
    callback.trainer = None
    callback.tokenizer = FakeTokenizer()
    callback.data_collator = FakeCollator()
    example = {
        "label_mode": "step_id",
        "all_instructions": ["step one"],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "choices"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "C"}]},
        ],
    }
    model = FakeModel()

    scores = callback._score_label_nlls(model, example)

    assert callback.data_collator.calls == 1
    assert model.forward_calls == 1
    assert list(scores) == ["A", "B", "C"]
    assert min(scores, key=scores.__getitem__) == "B"


def test_step_id_letter_logits_reuse_existing_eval_outputs() -> None:
    from types import SimpleNamespace

    import pytest

    torch = pytest.importorskip("torch")
    from online_mistake_detection.training.module_a import (
        step_id_letter_logits_from_outputs,
    )

    class RecordingHead(torch.nn.Linear):
        def __init__(self) -> None:
            super().__init__(4, 32, bias=False)
            self.calls = 0

        def forward(self, hidden: torch.Tensor) -> torch.Tensor:
            self.calls += 1
            return super().forward(hidden)

    class FakeModel:
        def __init__(self) -> None:
            self.head = RecordingHead()

        def get_output_embeddings(self) -> torch.nn.Module:
            return self.head

    model = FakeModel()
    hidden = torch.zeros(2, 6, 4)
    hidden[0, 3, 0] = 1.0
    hidden[1, 2, 1] = 1.0
    outputs = SimpleNamespace(hidden_states=[hidden], logits=None)

    logits = step_id_letter_logits_from_outputs(
        model,
        outputs,
        torch.tensor([[4], [3]]),
        torch.arange(26).repeat(2, 1),
    )

    assert logits.shape == (2, 26)
    assert model.head.calls == 1


def test_reused_predictions_produce_step_id_metrics() -> None:
    from online_mistake_detection.training.module_a import (
        module_a_metrics_from_predictions,
    )

    examples = [
        {
            "target": "A",
            "task_id": "task",
            "all_instructions": ["first"],
            "label_mode": "step_id",
        },
        {
            "target": "C",
            "task_id": "task",
            "all_instructions": ["first"],
            "label_mode": "step_id",
        },
    ]

    metrics = module_a_metrics_from_predictions(
        examples,
        ["A", "C"],
        beta=2.0,
    )

    assert metrics["eval_detection/accuracy"] == 1.0
    # EXTRA remains in the confusion catalog but has no ground-truth support,
    # so it does not depress the supported-class semantic macro.
    assert metrics["eval_step_id/task_task/num_classes"] == 3.0
    assert metrics["eval_step_id/task_task/num_supported_classes"] == 2.0
    assert metrics["eval_step_id/task_task/f1_EXTRA"] == 0.0
    assert metrics["eval_step_id/total_macro_f1"] == 1.0


def test_resume_eval_runs_for_unevaluated_epoch_checkpoint(tmp_path: Path) -> None:
    import json

    from online_mistake_detection.training.module_a import (
        checkpoint_needs_resume_evaluation,
    )

    (tmp_path / "trainer_state.json").write_text(
        json.dumps({
            "epoch": 1.0,
            "global_step": 322,
            "log_history": [{"loss": 0.02, "step": 322}],
        }),
        encoding="utf-8",
    )
    assert checkpoint_needs_resume_evaluation(tmp_path) is True


def test_resume_eval_skips_mid_epoch_and_already_evaluated_checkpoints(
    tmp_path: Path,
) -> None:
    import json

    from online_mistake_detection.training.module_a import (
        checkpoint_needs_resume_evaluation,
    )

    state_path = tmp_path / "trainer_state.json"
    state_path.write_text(
        json.dumps({"epoch": 1.4, "global_step": 450, "log_history": []}),
        encoding="utf-8",
    )
    assert checkpoint_needs_resume_evaluation(tmp_path) is False

    state_path.write_text(
        json.dumps({
            "epoch": 1.0,
            "global_step": 322,
            "log_history": [{"eval_loss": 0.1, "step": 322}],
        }),
        encoding="utf-8",
    )
    assert checkpoint_needs_resume_evaluation(tmp_path) is False


def test_apply_letter_only_labels_keeps_the_answer_token() -> None:
    import pytest

    torch = pytest.importorskip("torch")
    from online_mistake_detection.training.module_a import apply_letter_only_labels

    labels = torch.tensor([[-100, -100, 7, 8, 9], [-100, 3, 4, 5, -100]])
    indices = torch.tensor([2, 2])
    masked = apply_letter_only_labels(labels, indices)
    assert masked.tolist() == [[-100, -100, 7, -100, -100], [-100, -100, 4, -100, -100]]


def test_apply_letter_only_labels_rejects_unlabeled_letter() -> None:
    import pytest

    torch = pytest.importorskip("torch")
    from online_mistake_detection.training.module_a import apply_letter_only_labels

    labels = torch.full((1, 4), -100)
    with pytest.raises(RuntimeError, match="labeled response"):
        apply_letter_only_labels(labels, torch.tensor([1]))


def test_answer_letter_index_from_tail_uses_suffix() -> None:
    import pytest

    torch = pytest.importorskip("torch")
    from online_mistake_detection.training.module_a import answer_letter_index_from_tail

    input_ids = torch.tensor([10, 11, 12, 7, 8, 9, 0])
    attention = torch.tensor([1, 1, 1, 1, 1, 1, 0])
    assert answer_letter_index_from_tail(
        input_ids,
        attention,
        [7, 8, 9],
        max_seq_length=16,
    ) == 3


def test_letter_only_loss_requires_step_id(monkeypatch) -> None:
    import pytest

    from online_mistake_detection.training.module_a import parse_args

    monkeypatch.delenv("LOAD_ADAPTER_FROM", raising=False)
    monkeypatch.delenv("INITIAL_RESUME_FROM_CHECKPOINT", raising=False)
    monkeypatch.setattr(
        "sys.argv",
        ["module_a", "--letter-only-loss", "--module-a-label-mode", "legacy"],
    )
    with pytest.raises(SystemExit):
        parse_args()


def test_zero_shot_eval_rejects_adapter_path(monkeypatch) -> None:
    import pytest

    from online_mistake_detection.training.module_a import parse_args

    monkeypatch.delenv("LOAD_ADAPTER_FROM", raising=False)
    monkeypatch.delenv("INITIAL_RESUME_FROM_CHECKPOINT", raising=False)
    monkeypatch.setattr(
        "sys.argv",
        [
            "module_a",
            "--zero-shot-eval",
            "--load-adapter-from",
            "/tmp/ckpt_ep1",
        ],
    )
    with pytest.raises(SystemExit):
        parse_args()


def test_load_adapter_rejects_full_trainer_resume(monkeypatch) -> None:
    import pytest

    from online_mistake_detection.training.module_a import parse_args

    monkeypatch.delenv("LOAD_ADAPTER_FROM", raising=False)
    monkeypatch.delenv("INITIAL_RESUME_FROM_CHECKPOINT", raising=False)
    monkeypatch.setattr(
        "sys.argv",
        [
            "module_a",
            "--load-adapter-from",
            "/tmp/ckpt_ep1",
            "--initial-resume-from-checkpoint",
            "/tmp/ckpt_ep1",
        ],
    )
    with pytest.raises(SystemExit):
        parse_args()


def test_dummy_train_rows_for_eval_only_pads_world_size() -> None:
    from online_mistake_detection.training.module_a import dummy_train_rows_for_eval_only

    rows = dummy_train_rows_for_eval_only([{"target": "A"}], world_size=8)
    assert len(rows) == 8
    assert rows[0]["target"] == "A"


def test_label_score_diagnostics_reports_gate_auc_and_nll_spread() -> None:
    from online_mistake_detection.metrics.calibration import label_score_diagnostics

    # Gate score is the non-A mass, so the two positives outrank the negatives.
    probabilities = [
        {"A": 0.9, "C": 0.1},
        {"A": 0.8, "C": 0.2},
        {"A": 0.3, "C": 0.7},
        {"A": 0.4, "C": 0.6},
    ]
    targets = ["A", "A", "C", "C"]
    metrics = label_score_diagnostics(probabilities, targets)

    assert metrics["eval_detection/auc"] == 1.0
    assert metrics["eval_step_id/gate/auc"] == metrics["eval_detection/auc"]
    assert "eval_step_id/macro_auc" in metrics
    assert metrics["eval_step_id/gate_positive_rate"] == 0.5
    # Median must be reported separately: eval_loss is a mean of an unbounded
    # quantity, so a saturated minority can dominate it.
    assert metrics["eval_step_id/target_nll_median"] > 0.0
    assert metrics["eval_step_id/target_nll_p90"] >= metrics["eval_step_id/target_nll_median"]
    assert metrics["eval_step_id/target_nll_saturated_fraction"] == 0.0


def test_label_score_diagnostics_flags_saturated_targets() -> None:
    from online_mistake_detection.metrics.calibration import label_score_diagnostics

    metrics = label_score_diagnostics(
        [{"A": 1.0, "C": 0.0}, {"A": 0.5, "C": 0.5}],
        ["C", "C"],
    )
    assert metrics["eval_step_id/target_nll_saturated_fraction"] == 0.5


def test_logits_to_probabilities_matches_nll_scores() -> None:
    from online_mistake_detection.metrics.calibration import (
        logits_to_probabilities,
        scores_to_probabilities,
    )

    logits = [2.0, -1.0, 0.5]
    labels = ["A", "B", "C"]
    from_logits = logits_to_probabilities(logits, labels)
    from_nlls = scores_to_probabilities(dict(zip(labels, (-value for value in logits))))

    assert abs(sum(from_logits.values()) - 1.0) < 1e-9
    for label in labels:
        assert abs(from_logits[label] - from_nlls[label]) < 1e-9
    assert max(from_logits, key=from_logits.get) == "A"


def test_eval_only_requires_adapter_and_rejects_zero_shot(monkeypatch) -> None:
    import pytest

    from online_mistake_detection.training.module_a import parse_args

    def run(*extra: str):
        monkeypatch.setattr(
            "sys.argv",
            ["module_a", "--dataset", "egoper", "--eval-only", *extra],
        )
        return parse_args()

    with pytest.raises(SystemExit):
        run()
    with pytest.raises(SystemExit):
        run("--load-adapter-from", "/tmp/ckpt", "--zero-shot-eval")
    with pytest.raises(SystemExit):
        run("--load-adapter-from", "/tmp/ckpt", "--dry-run")
    with pytest.raises(SystemExit):
        run("--load-adapter-from", "/tmp/ckpt", "--eval-generation-max-samples", "0")

    args = run("--load-adapter-from", "/tmp/ckpt")
    assert args.eval_only is True
    assert args.zero_shot_eval is False
    assert args.write_calibration_records is True


def test_estimated_window_frames_orders_windows_by_sequence_length() -> None:
    """The --group-by-length sort key."""
    from online_mistake_detection.training.module_a import estimated_window_frames

    def frames(span: float, *, max_frames: int = 32) -> int:
        return estimated_window_frames(
            100.0, 100.0 + span, fps=1.0, min_frames=2, max_frames=max_frames
        )

    assert frames(0.0) == 2, "a degenerate window still decodes min_frames"
    assert frames(1.5) == 2, "below min_frames the floor applies, not the span"
    assert frames(12.0) == 12
    assert frames(60.0) == 32, "long windows are capped, not extended"
    assert frames(60.0, max_frames=40) == 40

    spans = [0.0, 3.0, 12.0, 31.0, 52.0, 600.0]
    lengths = [frames(s) for s in spans]
    assert lengths == sorted(lengths), "sort key must be monotonic in duration"

    # A negative span cannot happen from real windowing, but the sampler must
    # never see a negative length if one is ever constructed by hand.
    assert frames(-5.0) == 2
