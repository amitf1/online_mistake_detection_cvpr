"""EgoPER-only one-pass A/B/C monolith training."""

from __future__ import annotations

import argparse
import copy
import inspect
import json
import math
import os
import re
from collections import Counter
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

os.environ["UNSLOTH_RETURN_LOGITS"] = "0"
os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "0"

from ..datasets import EgoPERProvider
from ..config import ModuleAConfig, ModuleBConfig, ModuleCConfig
from ..datasets.training import (
    build_module_b_training_dataset,
    build_module_c_training_dataset,
)
from ..models.vision import vision_resize_arg
from ..schemas import TrainingExample
from .distributed_eval import (
    distributed_rank_and_world,
    gather_round_robin_shards,
    shard_examples_for_rank,
    unwrap_trainer_model,
)
from .kernel_cache import configure_kernel_caches
from .module_a import (
    EpochCheckpointCallback,
    Qwen3VLMetadataVisionDataCollator,
    checkpoint_needs_resume_evaluation,
    estimated_window_frames,
    is_bf16_available,
    is_distributed_rank_zero,
    json_safe,
    last_hidden_states,
    load_unsloth_model,
    release_cuda_memory,
    save_outputs,
    unwrap_output_embeddings,
    validate_lora_target_configuration,
    validate_model_precision,
)
from .module_c import disable_unsloth_extra_outputs
from .preemption import (
    add_preemption_args,
    apply_resume_checkpoint,
    attach_cluster_checkpointing,
)

MONOLITH_CLASSES = (
    "CORRECT",
    "SLIP",
    "MODIFICATION",
    "CORRECTION",
    "ADDITION",
)
DEFAULT_SPLIT_FILE = "configs/splits/egoper_video_split_seed3407.json"
MONOLITH_METRIC = "eval_monolith/stage_composite"


def geometric_stage_composite(a_value: float, b_value: float, c_value: float) -> float:
    """Geometric mean used for monolith best-checkpoint selection."""
    values = (float(a_value), float(b_value), float(c_value))
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError(f"Stage metrics must be finite and non-negative, got {values!r}")
    if any(value == 0.0 for value in values):
        return 0.0
    return math.exp(sum(math.log(value) for value in values) / len(values))


def parse_monolith_response(
    text: str,
    *,
    dataset: str = "egoper",
) -> dict[str, Any] | None:
    """Strict fallback parser; the metrics module may expose a richer parser."""
    from ..prompts.abc_monolith import (
        classification_matches_step,
        execution_classes_for_dataset,
    )
    try:
        payload = json.loads(text.strip())
    except (json.JSONDecodeError, TypeError, AttributeError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload == {"status": "NOT_COMPLETED"}:
        return payload
    expected = {"step_id", "start", "end", "classification", "reasoning"}
    if set(payload) != expected:
        return None
    step_id = payload.get("step_id")
    classification = payload.get("classification")
    reasoning = payload.get("reasoning")
    start = payload.get("start")
    end = payload.get("end")
    if not isinstance(step_id, str) or (
        step_id != "EXTRA" and re.fullmatch(r"STEP_(0|[1-9]\d*)", step_id) is None
    ):
        return None
    if classification not in execution_classes_for_dataset(
        dataset,
    ):
        return None
    if not classification_matches_step(
        step_id,
        str(classification),
        dataset,
    ):
        return None
    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, (int, float))
        or not isinstance(end, (int, float))
        or not math.isfinite(float(start))
        or not math.isfinite(float(end))
        or float(start) < 0.0
        or float(end) <= float(start)
    ):
        return None
    if not isinstance(reasoning, str) or not reasoning.strip():
        return None
    return {
        **payload,
        "start": float(start),
        "end": float(end),
        "reasoning": reasoning.strip(),
    }


def _metrics_parser() -> Callable[[str], Any]:
    try:
        from ..metrics import abc_monolith as metric_module
    except ImportError:
        return parse_monolith_response
    for name in (
        "parse_abc_monolith_prediction",
        "parse_monolith_prediction",
        "parse_monolith_response",
    ):
        parser = getattr(metric_module, name, None)
        if callable(parser):
            return parser
    return parse_monolith_response


def _flatten_scalars(value: Any, *, prefix: str = "") -> dict[str, float]:
    result: dict[str, float] = {}
    if isinstance(value, dict):
        for key, nested in value.items():
            child = f"{prefix}/{key}" if prefix else str(key)
            result.update(_flatten_scalars(nested, prefix=child))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        result[prefix] = float(value)
    return result


def _metric_value(metrics: dict[str, float], candidates: Iterable[str]) -> float:
    for candidate in candidates:
        if candidate in metrics:
            return float(metrics[candidate])
    raise KeyError(
        "Monolith metric callback did not publish any of "
        f"{tuple(candidates)!r}; available={sorted(metrics)}"
    )


def add_stage_composite(metrics: dict[str, Any]) -> dict[str, Any]:
    """Publish canonical component names and the geometric composite."""
    flat = _flatten_scalars(metrics)
    a_value = _metric_value(
        flat,
        (
            "eval_monolith/a_total_macro_f1",
            "eval_monolith/a/total_macro_f1",
            "a_total_macro_f1",
            "a/total_macro_f1",
            "module_a/total_macro_f1",
        ),
    )
    b_value = _metric_value(
        flat,
        (
            "eval_monolith/b_mixed_raw_macro",
            "eval_monolith/b/mixed_raw_macro",
            "b_mixed_raw_macro",
            "b/mixed_raw_macro",
            "module_b/mixed_raw_macro",
        ),
    )
    c_value = _metric_value(
        flat,
        (
            "eval_monolith/c_all_class_macro_f1",
            "eval_monolith/c/all_class_macro_f1",
            "c_all_class_macro_f1",
            "c/all_class_macro_f1",
            "module_c/all_class_macro_f1",
        ),
    )
    flat["eval_monolith/a_total_macro_f1"] = a_value
    flat["eval_monolith/b_mixed_raw_macro"] = b_value
    flat["eval_monolith/c_all_class_macro_f1"] = c_value
    flat[MONOLITH_METRIC] = geometric_stage_composite(a_value, b_value, c_value)
    return flat


def _row_dataset(row: dict[str, Any]) -> str:
    from ..prompts.abc_monolith import normalize_monolith_dataset

    return normalize_monolith_dataset(
        str((row.get("metadata") or {}).get("dataset") or "egoper")
    )


def _row_prompt_contract(row: dict[str, Any]) -> str | None:
    value = (row.get("metadata") or {}).get("prompt_contract")
    if value is None or not str(value).strip():
        return None
    return str(value)


def compute_monolith_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Project fixed native cohorts and compute their existing stage metrics."""
    from ..datasets.egoper import (
        EGOPER_AGGREGATE_MAPPING,
        EGOPER_MODULE_C_CLASSES,
    )
    from ..metrics.abc_monolith import parse_monolith_prediction
    from ..metrics.abc_monolith import step_id_to_module_a_label
    from ..prompts.abc_monolith import resolve_monolith_contract
    from ..metrics.calibration import label_score_diagnostics
    from ..metrics.module_a import step_id_semantic_metrics
    from ..metrics.temporal import TemporalPrediction
    from .module_b import isolated_temporal_metrics, temporal_metrics
    from .module_c import module_c_multiclass_metrics, module_c_reasoning_metrics

    targets = [
        parse_monolith_prediction(
            str(row["target"]),
            clip_duration=float(row["window_end"]) - float(row["window_start"]),
            dataset=_row_dataset(row),
            prompt_contract=_row_prompt_contract(row),
        )
        for row in rows
    ]
    predictions = [
        (
            row.get("prediction")
            if hasattr(row.get("prediction"), "valid")
            else parse_monolith_prediction(
                str(row["raw_output"]),
                clip_duration=float(row["window_end"]) - float(row["window_start"]),
                dataset=_row_dataset(row),
                    prompt_contract=_row_prompt_contract(row),
            )
        )
        for row in rows
    ]
    parsed_rows = list(zip(rows, targets, predictions))
    a_rows = [
        (row, target, prediction)
        for row, target, prediction in parsed_rows
        if str((row.get("metadata") or {}).get("projection_stage")) == "A"
    ]
    a_metrics = step_id_semantic_metrics(
        task_ids=[str(row["task_id"]) for row, _target, _prediction in a_rows],
        all_instructions=[
            list((row.get("metadata") or {}).get("all_instructions") or ())
            for row, _target, _prediction in a_rows
        ],
        targets=[
            step_id_to_module_a_label(target.step_id if target.completed else None)
            for _row, target, _prediction in a_rows
        ],
        predictions=[
            step_id_to_module_a_label(
                prediction.step_id
                if prediction.valid and prediction.completed
                else None
            )
            for _row, _target, prediction in a_rows
        ],
        prefix="eval_step_id",
    )
    if a_rows and all(
        isinstance(row.get("step_probabilities"), dict)
        and bool(row["step_probabilities"])
        for row, _target, _prediction in a_rows
    ):
        a_metrics.update(
            label_score_diagnostics(
                [row["step_probabilities"] for row, _target, _prediction in a_rows],
                [
                    step_id_to_module_a_label(
                        target.step_id if target.completed else None
                    )
                    for _row, target, _prediction in a_rows
                ],
            )
        )

    b_rows = [
        (row, target, prediction)
        for row, target, prediction in parsed_rows
        if str((row.get("metadata") or {}).get("projection_stage")) == "B"
    ]
    temporal_rows: list[dict[str, Any]] = []
    for row, target, prediction in b_rows:
        if prediction.valid and prediction.completed:
            temporal_prediction = TemporalPrediction(
                prediction.start,
                prediction.end,
                source_format="abc_monolith",
                raw_start=prediction.start,
                raw_end=prediction.end,
            )
        elif prediction.valid:
            temporal_prediction = TemporalPrediction(
                None, None, "no_action", "abc_monolith"
            )
        else:
            temporal_prediction = TemporalPrediction(
                None, None, "malformed", "abc_monolith"
            )
        temporal_rows.append(
            {
                "label": (
                    "LOCALIZE" if target.valid and target.completed else "NO_ACTION"
                ),
                "late_trigger_augmented": bool(
                    (row.get("metadata") or {}).get("late_trigger_augmented")
                ),
                "target": (
                    (float(target.start), float(target.end))
                    if target.start is not None and target.end is not None
                    else (0.0, 0.0)
                ),
                "prediction": temporal_prediction,
            }
        )
    b_metrics = isolated_temporal_metrics(temporal_rows)
    for view in ("raw", "clamped"):
        prefix = f"eval_temporal/mixed/{view}"
        b_metrics.update(
            temporal_metrics(
                [row["target"] for row in temporal_rows],
                [row["prediction"] for row in temporal_rows],
                [str(row["label"]) for row in temporal_rows],
                prefix=prefix,
                view=view,
            )
        )
        b_metrics[f"{prefix}/macro_score"] = (
            b_metrics[f"eval_temporal/on_time/{view}/mean_iou"]
            + b_metrics[f"eval_temporal/late/{view}/mean_iou"]
            + b_metrics["eval_temporal/not_completed/accuracy"]
        ) / 3.0

    c_rows = [
        (row, target, prediction)
        for row, target, prediction in parsed_rows
        if str((row.get("metadata") or {}).get("projection_stage")) == "C"
    ]
    fine_classes = EGOPER_MODULE_C_CLASSES
    aggregate_mapping = EGOPER_AGGREGATE_MAPPING
    c_score_key = "eval_all/macro_f1"
    c_targets = [
        str(target.classification) if target.completed else "NOT_COMPLETED"
        for _row, target, _prediction in c_rows
    ]
    c_predictions = []
    for row, _target, prediction in c_rows:
        addition_is_not_a_step_class = (
            resolve_monolith_contract(
                _row_dataset(row),
                _row_prompt_contract(row),
            )
            == "egoper"
        )
        if (
            prediction.valid
            and prediction.completed
            and not (addition_is_not_a_step_class and prediction.classification == "ADDITION")
        ):
            c_predictions.append(str(prediction.classification))
        elif prediction.valid and not prediction.completed:
            c_predictions.append("NOT_COMPLETED")
        else:
            c_predictions.append(None)
    c_metrics = module_c_multiclass_metrics(
        c_targets,
        c_predictions,
        fine_classes=fine_classes,
        aggregate_mapping=aggregate_mapping,
    )
    reasoning_metrics, _ = module_c_reasoning_metrics(
        [
            str(target.reasoning or "")
            for _row, target, _prediction in c_rows
            if target.completed
        ],
        [
            prediction.reasoning if prediction.valid and prediction.completed else None
            for _row, target, prediction in c_rows
            if target.completed
        ],
        total_examples=sum(target.completed for _row, target, _prediction in c_rows),
    )
    valid_rate = (
        sum(bool(prediction.valid) for prediction in predictions) / len(predictions)
        if predictions
        else 0.0
    )
    return add_stage_composite(
        {
            **a_metrics,
            **b_metrics,
            **c_metrics,
            **reasoning_metrics,
            "eval_monolith/a_total_macro_f1": a_metrics[
                "eval_step_id/total_macro_f1"
            ],
            "eval_monolith/b_mixed_raw_macro": b_metrics[
                "eval_temporal/mixed/raw/macro_score"
            ],
            "eval_monolith/c_all_class_macro_f1": c_metrics[c_score_key],
            "eval_monolith/parse_valid_rate": valid_rate,
        }
    )


def restore_abc_monolith_training_state(model: Any) -> None:
    """Restore memory-critical state after inline cached generation."""
    from unsloth import FastVisionModel

    FastVisionModel.for_training(model, use_gradient_checkpointing="unsloth")
    if getattr(model, "is_gradient_checkpointing", True) is False:
        enable = getattr(model, "gradient_checkpointing_enable", None)
        if callable(enable):
            enable()
    model.train()
    for config in (
        getattr(model, "config", None),
        getattr(getattr(model, "config", None), "text_config", None),
    ):
        if config is not None and hasattr(config, "use_cache"):
            config.use_cache = False


class ABCMonolithUnslothEnvGuardCallback:
    """Reapply training state at the first forward after generation eval."""

    def __init__(self) -> None:
        from transformers import TrainerCallback

        self._callback_base = TrainerCallback()
        self.trainer: Any | None = None
        self._restore_training_state_at_next_step = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._callback_base, name)

    def on_evaluate(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        disable_unsloth_extra_outputs()
        self._restore_training_state_at_next_step = True
        return control

    def on_step_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        disable_unsloth_extra_outputs()
        if self._restore_training_state_at_next_step:
            model = kwargs.get("model")
            if model is not None:
                model = unwrap_trainer_model(self.trainer, model)
                restore_abc_monolith_training_state(model)
                release_cuda_memory()
            self._restore_training_state_at_next_step = False
        return control

    def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        disable_unsloth_extra_outputs()
        return control


class ABCMonolithMetricsCallback:
    """Full-generation oracle projection, sharded over every DDP rank."""

    def __init__(
        self,
        *,
        eval_examples: list[dict[str, Any]],
        data_collator: Qwen3VLMetadataVisionDataCollator,
        tokenizer: Any,
        max_samples: int,
        max_new_tokens: int,
        batch_size: int,
        use_cache: bool,
        enable_thinking: bool,
        candidate_scoring: bool,
    ) -> None:
        from transformers import TrainerCallback

        self._callback_base = TrainerCallback()
        self.trainer: Any | None = None
        self.eval_examples = (
            eval_examples if max_samples < 0 else eval_examples[:max_samples]
        )
        self.data_collator = data_collator
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.use_cache = use_cache
        self.enable_thinking = enable_thinking
        self.candidate_scoring = candidate_scoring

    def __getattr__(self, name: str) -> Any:
        return getattr(self._callback_base, name)

    def _generation_inputs(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        texts: list[str] = []
        images: list[Any] = []
        videos: list[Any] = []
        video_metadatas: list[Any] = []
        processor_kwargs: dict[str, Any] = {}
        for example in examples:
            messages = [
                copy.deepcopy(message)
                for message in example["messages"]
                if message.get("role") != "assistant"
            ]
            processor = self.data_collator._base.processor
            texts.append(
                processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=self.enable_thinking,
                )
            )
            image, video, video_kwargs, metadata = (
                self.data_collator._extract_images_videos_metadata(messages)
            )
            images.extend(image or [])
            videos.extend(video or [])
            for key, value in video_kwargs.items():
                if key != "fps":
                    processor_kwargs[key] = value
            video_metadatas.extend(metadata or [])
        kwargs: dict[str, Any] = {
            "text": texts,
            "padding": True,
            "return_tensors": "pt",
            "add_special_tokens": False,
        }
        if images:
            kwargs["images"] = images
        if videos:
            kwargs.update(processor_kwargs)
            kwargs["videos"] = videos
            kwargs["do_resize"] = False
            if video_metadatas:
                kwargs["video_metadata"] = video_metadatas
        processor = self.data_collator._base.processor
        processor_tokenizer = getattr(processor, "tokenizer", None)
        old_padding_side = getattr(processor_tokenizer, "padding_side", None)
        if processor_tokenizer is not None:
            processor_tokenizer.padding_side = "left"
        try:
            batch = processor(**kwargs)
        finally:
            if processor_tokenizer is not None and old_padding_side is not None:
                processor_tokenizer.padding_side = old_padding_side
        attention_mask = batch.get("attention_mask")
        if (
            attention_mask is not None
            and attention_mask.shape[0] > 1
            and not bool((attention_mask[:, -1] != 0).all())
        ):
            raise RuntimeError(
                "ABC monolith batched generation produced right-padded inputs; "
                "decoder-only generation requires left padding."
            )
        return batch

    def _generate(self, model: Any, examples: list[dict[str, Any]]) -> list[str]:
        inputs = self._generation_inputs(examples)
        device = next(model.parameters()).device
        inputs = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        input_length = inputs["input_ids"].shape[-1]
        kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": False,
            "use_cache": self.use_cache,
        }
        if getattr(self.tokenizer, "pad_token_id", None) is not None:
            kwargs["pad_token_id"] = self.tokenizer.pad_token_id
        try:
            output_ids = model.generate(**inputs, **kwargs)
            return self.tokenizer.batch_decode(
                output_ids[:, input_length:], skip_special_tokens=True
            )
        finally:
            del inputs
            if "output_ids" in locals():
                del output_ids

    def _score_targets(
        self,
        model: Any,
        example: dict[str, Any],
        targets: dict[str, str],
    ) -> dict[str, float]:
        """Length-normalized response NLL without full-sequence vocabulary logits."""
        import torch
        import torch.nn.functional as functional

        device = next(model.parameters()).device
        output_embeddings = unwrap_output_embeddings(model)
        scores: dict[str, float] = {}
        for label, target in targets.items():
            scored = copy.deepcopy(example)
            scored["target"] = target
            for message in scored["messages"]:
                if message.get("role") == "assistant":
                    message["content"] = [{"type": "text", "text": target}]
                    break
            batch = self.data_collator([scored])
            inputs = {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in batch.items()
            }
            labels = inputs.pop("labels", None)
            for key in (
                "response_loss_weight",
                "eval_row_index",
                "eval_answer_positions",
                "eval_candidate_token_ids",
            ):
                inputs.pop(key, None)
            if labels is None:
                raise RuntimeError("Monolith candidate scoring requires response labels.")
            outputs = None
            try:
                with torch.inference_mode():
                    outputs = model(
                        **inputs,
                        output_hidden_states=True,
                        return_dict=True,
                        use_cache=False,
                    )
                    hidden = last_hidden_states(outputs)[:, :-1]
                    shifted = labels[:, 1:]
                    selected = shifted != -100
                    if not bool(selected.any()):
                        raise RuntimeError(
                            f"No response tokens found for candidate {label!r}."
                        )
                    logits = output_embeddings(hidden[selected])
                    nll = functional.cross_entropy(
                        logits.float(),
                        shifted[selected].to(logits.device),
                        reduction="none",
                    )
                    scores[label] = float(nll.mean().detach().cpu())
            finally:
                del inputs
                if outputs is not None:
                    del outputs
                release_cuda_memory()
        return scores

    @staticmethod
    def _candidate_targets(
        example: dict[str, Any],
        prediction: Any,
    ) -> tuple[dict[str, str], dict[str, str]]:
        from ..prompts.abc_monolith import (
            execution_classes_for_dataset,
            format_monolith_target,
            resolve_monolith_contract,
        )
        from ..prompts.contracts import NOT_COMPLETED_TARGET

        metadata = example.get("metadata") or {}
        dataset = _row_dataset(example)
        contract = resolve_monolith_contract(
            dataset,
            metadata.get("prompt_contract"),
        )
        base = (
            prediction
            if getattr(prediction, "valid", False)
            and getattr(prediction, "completed", False)
            else None
        )
        clip_duration = float(example["window_end"]) - float(example["window_start"])
        start = float(getattr(base, "start", None) or 0.0)
        end = float(getattr(base, "end", None) or min(1.0, clip_duration))
        if end <= start:
            start, end = 0.0, min(1.0, clip_duration)
        reasoning = str(
            getattr(base, "reasoning", None)
            or "Canonical candidate used only for offline confidence scoring."
        )
        instructions = list((example.get("metadata") or {}).get("all_instructions") or ())
        step_targets = {"A": NOT_COMPLETED_TARGET}
        semantic_steps = ["EXTRA", *(f"STEP_{index}" for index in range(len(instructions)))]
        extra_classes = ("ADDITION",)
        for step_id in semantic_steps:
            letter = (
                "B"
                if step_id == "EXTRA"
                else chr(ord("C") + int(step_id.removeprefix("STEP_")))
            )
            step_class = extra_classes[0] if step_id == "EXTRA" else "CORRECT"
            step_targets[letter] = format_monolith_target(
                step_id=step_id,
                start=start,
                end=end,
                classification=step_class,
                reasoning=reasoning,
                dataset=contract,
            )
        step_id = str(getattr(base, "step_id", None) or "")
        class_targets: dict[str, str] = {}
        if step_id == "EXTRA":
            for class_name in extra_classes:
                class_targets[class_name] = format_monolith_target(
                    step_id="EXTRA",
                    start=start,
                    end=end,
                    classification=class_name,
                    reasoning=reasoning,
                    dataset=contract,
                )
        elif step_id.startswith("STEP_"):
            for class_name in execution_classes_for_dataset(
                contract,
            ):
                if contract == "egoper" and class_name == "ADDITION":
                    continue
                class_targets[class_name] = format_monolith_target(
                    step_id=step_id,
                    start=start,
                    end=end,
                    classification=class_name,
                    reasoning=reasoning,
                    dataset=contract,
                )
        return step_targets, class_targets

    @staticmethod
    def scoring_targets(
        example: dict[str, Any],
        prediction: Any,
    ) -> tuple[dict[str, str], dict[str, str]]:
        """Class NLLs only after a completed step."""
        _step_targets, class_targets = ABCMonolithMetricsCallback._candidate_targets(
            example, prediction
        )
        if not (
            getattr(prediction, "valid", False)
            and getattr(prediction, "completed", False)
        ):
            return {}, {}
        return {}, class_targets

    def on_evaluate(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        if self.trainer is None or not self.eval_examples:
            return control
        import torch
        from unsloth import FastVisionModel

        rank, world_size = distributed_rank_and_world()
        local_examples = shard_examples_for_rank(
            self.eval_examples, rank=rank, world_size=world_size
        )
        model = unwrap_trainer_model(self.trainer, kwargs["model"])
        parser = _metrics_parser()
        local_rows: list[dict[str, Any]] = []
        FastVisionModel.for_inference(model)
        try:
            with torch.inference_mode():
                for start in range(0, len(local_examples), self.batch_size):
                    examples = local_examples[start : start + self.batch_size]
                    generated = self._generate(model, examples)
                    for example, raw in zip(examples, generated):
                        metadata = example.get("metadata") or {}
                        try:
                            parsed = parser(
                                raw,
                                clip_duration=(
                                    float(example["window_end"])
                                    - float(example["window_start"])
                                ),
                                dataset=str(metadata.get("dataset") or "egoper"),
                                prompt_contract=metadata.get("prompt_contract"),
                            )
                        except TypeError:
                            parsed = parser(raw)
                        score_payload: dict[str, Any] = {}
                        if self.candidate_scoring:
                            step_targets, class_targets = self.scoring_targets(
                                example, parsed
                            )
                            from ..metrics.calibration import scores_to_probabilities

                            if step_targets:
                                step_nlls = self._score_targets(
                                    model, example, step_targets
                                )
                                score_payload["step_candidate_nlls"] = step_nlls
                                score_payload["step_probabilities"] = (
                                    scores_to_probabilities(step_nlls)
                                )
                                score_payload["extra_probability"] = score_payload[
                                    "step_probabilities"
                                ].get("B")
                            if class_targets:
                                class_nlls = self._score_targets(
                                    model, example, class_targets
                                )
                                class_probabilities = scores_to_probabilities(
                                    class_nlls
                                )
                                score_payload["class_candidate_nlls"] = class_nlls
                                score_payload["class_probabilities"] = (
                                    class_probabilities
                                )
                                score_payload["error_score"] = sum(
                                    value
                                    for key, value in class_probabilities.items()
                                    if key != "CORRECT"
                                )
                        local_rows.append(
                            {
                                **{
                                    key: example.get(key)
                                    for key in (
                                        "video_id",
                                        "task_id",
                                        "step_index",
                                        "window_start",
                                        "window_end",
                                        "gt_start",
                                        "gt_end",
                                        "target",
                                        "label",
                                        "metadata",
                                    )
                                },
                                "prediction": parsed,
                                "raw_output": raw,
                                **score_payload,
                            }
                        )
        finally:
            # Restore now for safety; the guard repeats this at next step_begin,
            # after all evaluation callbacks have completed.
            restore_abc_monolith_training_state(model)
            disable_unsloth_extra_outputs()
            release_cuda_memory()

        rows = gather_round_robin_shards(local_rows)
        metrics = compute_monolith_metrics(rows)
        callback_metrics = kwargs.get("metrics")
        if isinstance(callback_metrics, dict):
            callback_metrics.update(metrics)
        # Every rank computes and receives identical metrics before checkpoint
        # selection. Only rank zero emits duplicate-free logs and artifacts.
        if rank == 0:
            prediction_dir = Path(self.trainer.args.output_dir) / "eval_predictions"
            prediction_dir.mkdir(parents=True, exist_ok=True)
            path = prediction_dir / f"step-{int(state.global_step):08d}.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    serializable = dict(row)
                    if is_dataclass(serializable.get("prediction")):
                        serializable["prediction"] = asdict(serializable["prediction"])
                    handle.write(
                        json.dumps(json_safe(serializable), ensure_ascii=False) + "\n"
                    )
            self.trainer.log(metrics)
        return control


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the EgoPER one-pass A/B/C Qwen vision monolith."
    )
    parser.add_argument("--dataset", choices=("egoper",), default="egoper")
    parser.add_argument("--egoper-root", default=None)
    parser.add_argument("--model-name", default="unsloth/Qwen3.8-27B")
    parser.add_argument("--max-videos", type=int, default=389)
    parser.add_argument("--split-file", default=DEFAULT_SPLIT_FILE)
    parser.add_argument(
        "--load-adapter-from",
        default=None,
        help="Load these LoRA weights and start a fresh optimizer.",
    )
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--load-in-8bit", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--load-in-16bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--lora-dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--lora-target-modules", choices=("auto", "all-linear"), default="auto")
    parser.add_argument("--finetune-vision-layers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--finetune-language-layers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--finetune-attention-modules", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--finetune-mlp-modules", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-mode", choices=("steps", "epochs"), default="epochs")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--num-train-epochs", type=float, default=6.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--expected-world-size", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--lr-scheduler-type", default="linear")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--optim", default="adamw_torch_fused")
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--checkpoint-epochs", type=int, default=1)
    parser.add_argument("--eval-epochs", type=int, default=1)
    parser.add_argument("--eval-on-resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--keep-last-checkpoints", type=int, default=4)
    parser.add_argument("--keep-best-checkpoints", type=int, default=4)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--early-stopping-threshold", type=float, default=0.0)
    parser.add_argument("--metric-for-best-model", default=MONOLITH_METRIC)
    parser.add_argument("--greater-is-better", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--min-frames", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=54)
    parser.add_argument("--eval-max-frames", type=int, default=54)
    parser.add_argument("--vision-resize", type=vision_resize_arg, default=1024)
    parser.add_argument("--max-seq-length", type=int, default=20480)
    parser.add_argument("--eval-generation-max-samples", type=int, default=-1)
    parser.add_argument("--eval-generation-max-new-tokens", type=int, default=192)
    parser.add_argument("--eval-generation-batch-size", type=int, default=2)
    parser.add_argument("--eval-generation-use-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--generation-enable-thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--monolith-candidate-scoring", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--history-max-events", type=int, default=12)
    parser.add_argument("--history-dropout-probability", type=float, default=0.25)
    parser.add_argument("--late-trigger-ratio", type=float, default=0.10)
    parser.add_argument("--max-natural-not-completed-per-episode", type=int, default=2)
    parser.add_argument("--non-correct-oversample-factor", type=float, default=4.0)
    parser.add_argument(
        "--not-completed-ratio",
        type=float,
        default=None,
        help=(
            "After non-correct upsampling, keep this many NOT_COMPLETED rows "
            "per completed row. Unset keeps the per-episode cap only."
        ),
    )
    parser.add_argument("--output-dir", default="outputs/abc_monolith/runs/default")
    parser.add_argument("--report-to", default="tensorboard")
    parser.add_argument("--dataset-num-proc", type=int, default=1)
    parser.add_argument("--torch-empty-cache-steps", type=int, default=100)
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument("--dataloader-prefetch-factor", type=int, default=None)
    parser.add_argument("--group-by-length", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ddp-find-unused-parameters", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ddp-timeout", type=int, default=7200)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-token-budget", action="store_true")
    parser.add_argument("--memory-preflight", action="store_true")
    parser.add_argument("--memory-preflight-only", action="store_true")
    parser.add_argument("--token-budget-safety-margin", type=int, default=256)
    parser.add_argument("--print-examples", type=int, default=2)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--wandb-log-best-checkpoints", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wandb-artifact-prefix", default="abc-monolith")
    add_preemption_args(parser)
    args = parser.parse_args(argv)
    args.report_to = [item.strip() for item in args.report_to.split(",") if item.strip()]
    validate_args(args, parser.error)
    return args


def validate_args(args: argparse.Namespace, fail: Callable[[str], Any]) -> None:
    if args.memory_preflight_only:
        args.memory_preflight = True
    if args.memory_preflight and not args.validate_token_budget:
        fail("--memory-preflight requires --validate-token-budget to select the exact worst row.")
    if args.metric_for_best_model != MONOLITH_METRIC:
        fail(f"--metric-for-best-model must be {MONOLITH_METRIC!r}.")
    if args.eval_generation_max_new_tokens != 192:
        fail("The approved monolith profile requires 192 generated tokens.")
    if args.expected_world_size * args.per_device_train_batch_size * args.gradient_accumulation_steps != 16:
        fail("Expected world size × train batch × accumulation must equal global batch 16.")
    if not all(
        (
            args.finetune_vision_layers,
            args.finetune_language_layers,
            args.finetune_attention_modules,
            args.finetune_mlp_modules,
        )
    ):
        fail("The monolith requires vision, language, attention, and MLP LoRA.")
    if args.lora_dropout != 0.0:
        fail("The approved LoRA dropout is 0.")
    if not 0.0 <= args.history_dropout_probability <= 1.0:
        fail("--history-dropout-probability must be in [0, 1].")
    not_completed_ratio = getattr(args, "not_completed_ratio", None)
    if not_completed_ratio is not None and not 0.0 <= not_completed_ratio <= 1.0:
        fail("--not-completed-ratio must be in [0, 1].")
    if args.eval_generation_max_samples == 0:
        fail("Full generation validation is required for composite checkpoint selection.")
    dataset = str(getattr(args, "dataset", "egoper") or "egoper")
    if dataset != "egoper":
        fail(f"Unsupported monolith dataset {dataset!r}.")
    if not args.egoper_root:
        fail("--egoper-root is required for the EgoPER monolith.")
    if Path(args.split_file).resolve() != Path(DEFAULT_SPLIT_FILE).resolve():
        fail(f"--split-file must be the fixed EgoPER split {DEFAULT_SPLIT_FILE!r}.")
    if args.max_frames != 54 or args.eval_max_frames != 54:
        fail("The approved monolith profile requires 54 train/eval frames.")
    if args.vision_resize != 1024:
        fail("The approved monolith profile requires --vision-resize 1024.")
    if args.max_seq_length < 20480:
        fail("--max-seq-length must be at least the provisional 20480-token budget.")
    if args.lora_r != 16 or args.lora_alpha != 32:
        fail("The approved EgoPER LoRA profile is r=16, alpha=32.")
    if args.train_mode == "epochs" and args.num_train_epochs != 6:
        fail("The approved EgoPER epoch cap is 6.")
    validate_model_precision(args, fail)
    validate_lora_target_configuration(args, fail)


def _load_fixed_split(args: argparse.Namespace, records: list[Any]) -> dict[str, Any]:
    path = Path(args.split_file)
    if not path.is_file():
        raise FileNotFoundError(f"Fixed EgoPER split not found: {path}")
    split = json.loads(path.read_text(encoding="utf-8"))
    known = {record.video_id for record in records}
    train = set(split.get("train_video_ids", ()))
    val = set(split.get("val_video_ids", ()))
    if not train or not val or train & val:
        raise ValueError("Fixed split must contain non-empty disjoint train/val video lists.")
    missing = (train | val) - known
    if missing:
        raise ValueError(f"Fixed split references unavailable videos: {sorted(missing)[:5]}")
    return split


def _build_examples(
    records: list[Any],
    args: argparse.Namespace,
    *,
    training: bool,
    sampled_incomplete_examples: list[TrainingExample] | None = None,
) -> list[TrainingExample]:
    from ..windowing.abc_monolith import build_abc_monolith_examples

    values = {
        "records": records,
        "seed": args.seed,
        "training": training,
        "max_history": args.history_max_events,
        "history_max_events": args.history_max_events,
        "history_dropout_probability": (
            args.history_dropout_probability if training else 0.0
        ),
        "late_trigger_probability": args.late_trigger_ratio if training else 0.0,
        "late_trigger_ratio": args.late_trigger_ratio if training else 0.0,
        "max_natural_not_completed_per_episode": (
            args.max_natural_not_completed_per_episode if training else None
        ),
        "mistake_repeat_factor": (
            int(args.non_correct_oversample_factor) if training else 1
        ),
        "not_completed_ratio": (
            getattr(args, "not_completed_ratio", None) if training else None
        ),
        "non_correct_oversample_factor": (
            args.non_correct_oversample_factor if training else 1.0
        ),
        "reset_jitter_seconds": 0.0,
        "sampled_incomplete_examples": sampled_incomplete_examples or (),
    }
    signature = inspect.signature(build_abc_monolith_examples)
    kwargs = {
        key: value
        for key, value in values.items()
        if key != "records" and key in signature.parameters
    }
    return list(build_abc_monolith_examples(records, **kwargs))


def _system_prompt(
    dataset: str = "egoper",
    *,
    prompt_contract: str | None = None,
) -> str:
    from ..prompts.abc_monolith import system_prompt_for_dataset

    return system_prompt_for_dataset(
        dataset,
        prompt_contract=prompt_contract,
    )


def training_example_to_conversation(
    example: TrainingExample,
    *,
    fps: float,
    min_frames: int,
    max_frames: int,
) -> dict[str, Any]:
    return {
        "messages": [
            {
                "role": "system",
                "content": [{
                    "type": "text",
                    "text": _system_prompt(
                        example.source_dataset,
                        prompt_contract=example.metadata.get("prompt_contract"),
                    ),
                }],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": str(example.video_path),
                        "video_start": example.window_start,
                        "video_end": example.window_end,
                        "fps": fps,
                        "min_frames": min_frames,
                        "max_frames": max_frames,
                    },
                    {"type": "text", "text": example.prompt_text},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": example.target_text}],
            },
        ],
        "video_id": example.video_id,
        "task_id": example.task_id,
        "step_index": example.step_index,
        "window_start": example.window_start,
        "window_end": example.window_end,
        "gt_start": example.gt_start,
        "gt_end": example.gt_end,
        "target": example.target_text,
        "label": example.label,
        "metadata": example.metadata,
    }


def token_budget_preflight(
    conversations: list[dict[str, Any]],
    *,
    collator: Any,
    max_seq_length: int,
    safety_margin: int,
) -> dict[str, int]:
    """Exhaustively validate processor lengths and answer truncation."""
    maximum = 0
    maximum_index = -1
    for index, conversation in enumerate(conversations):
        batch = collator([conversation])
        labels = batch.get("labels")
        input_ids = batch.get("input_ids")
        if labels is None or input_ids is None:
            raise RuntimeError("Token preflight requires collator input_ids and labels.")
        length = int(input_ids.shape[-1])
        if length > maximum:
            maximum = length
            maximum_index = index
        if not bool((labels != -100).any()):
            raise RuntimeError(f"Answer was truncated from preflight row {index}.")
    available = max_seq_length - maximum
    if available < safety_margin:
        raise RuntimeError(
            f"Token preflight margin {available} is below required {safety_margin}; "
            "move to the next aligned sequence length."
        )
    return {
        "max_tokens": maximum,
        "safety_margin": available,
        "max_row_index": maximum_index,
    }


def forward_backward_memory_preflight(
    *,
    model: Any,
    collator: Any,
    metrics_callback: "ABCMonolithMetricsCallback",
    train_example: dict[str, Any],
    eval_example: dict[str, Any],
) -> dict[str, Any]:
    """Exercise train -> inline generation -> restored train forward/backward."""
    import time

    import torch
    from unsloth import FastVisionModel

    device = next(model.parameters()).device

    def train_backward(example: dict[str, Any]) -> float:
        restore_abc_monolith_training_state(model)
        batch = collator([example])
        inputs = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in batch.items()
        }
        model.zero_grad(set_to_none=True)
        output = None
        try:
            output = model(**inputs)
            loss = output.loss
            if loss is None or not bool(torch.isfinite(loss).all()):
                raise RuntimeError("Memory preflight produced a non-finite training loss.")
            loss.backward()
            return float(loss.detach().cpu())
        finally:
            model.zero_grad(set_to_none=True)
            del inputs
            if output is not None:
                del output
            release_cuda_memory()

    before_loss = train_backward(train_example)
    FastVisionModel.for_inference(model)
    generation_started = time.perf_counter()
    with torch.inference_mode():
        generated = metrics_callback._generate(model, [eval_example])[0]
    generation_seconds = time.perf_counter() - generation_started
    restore_abc_monolith_training_state(model)
    release_cuda_memory()
    after_loss = train_backward(train_example)
    return {
        "train_loss_before_generation": before_loss,
        "generated_characters": len(generated),
        "generation_seconds": generation_seconds,
        "train_loss_after_generation": after_loss,
        "training_mode_restored": bool(model.training),
        "use_cache_restored": not bool(getattr(model.config, "use_cache", False)),
    }


def build_trainer(
    *,
    model: Any,
    tokenizer: Any,
    train_dataset: list[dict[str, Any]],
    eval_dataset: list[dict[str, Any]],
    args: argparse.Namespace,
) -> Any:
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastVisionModel

    FastVisionModel.for_training(model, use_gradient_checkpointing="unsloth")

    def with_length(row: dict[str, Any], max_frames: int) -> dict[str, Any]:
        if not args.group_by_length:
            return row
        return {
            **row,
            "length": estimated_window_frames(
                row["window_start"],
                row["window_end"],
                fps=args.fps,
                min_frames=args.min_frames,
                max_frames=max_frames,
            ),
        }

    hf_train = Dataset.from_list([with_length(row, args.max_frames) for row in train_dataset])
    hf_eval = Dataset.from_list([with_length(row, args.eval_max_frames) for row in eval_dataset])
    eval_on_start = bool(
        args.eval_on_resume
        and args.resume_from_checkpoint
        and checkpoint_needs_resume_evaluation(args.resume_from_checkpoint)
    )
    trainer_args: dict[str, Any] = {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "warmup_steps": args.warmup_steps,
        "learning_rate": args.learning_rate,
        "logging_steps": args.logging_steps,
        "eval_strategy": "no",
        "save_strategy": "no",
        "load_best_model_at_end": False,
        "metric_for_best_model": args.metric_for_best_model,
        "greater_is_better": args.greater_is_better,
        "optim": args.optim,
        "weight_decay": args.weight_decay,
        "lr_scheduler_type": args.lr_scheduler_type,
        "max_grad_norm": args.max_grad_norm,
        "seed": args.seed,
        "report_to": args.report_to,
        "dataset_num_proc": args.dataset_num_proc,
        "dataloader_num_workers": args.dataloader_num_workers,
        "dataloader_prefetch_factor": args.dataloader_prefetch_factor,
        "dataloader_pin_memory": True,
        "max_seq_length": args.max_seq_length,
        "torch_empty_cache_steps": args.torch_empty_cache_steps or None,
        "ddp_find_unused_parameters": args.ddp_find_unused_parameters,
        "ddp_broadcast_buffers": False,
        "ddp_timeout": args.ddp_timeout,
        "fp16": not is_bf16_available(),
        "bf16": is_bf16_available(),
        "gradient_checkpointing": True,
        "remove_unused_columns": False,
        "dataset_text_field": "",
        "dataset_kwargs": {"skip_prepare_dataset": True},
        "train_sampling_strategy": "group_by_length" if args.group_by_length else "random",
        "length_column_name": "length",
        "eval_on_start": eval_on_start,
    }
    if args.train_mode == "epochs":
        trainer_args["num_train_epochs"] = args.num_train_epochs
    else:
        trainer_args["max_steps"] = args.max_steps
    collator = Qwen3VLMetadataVisionDataCollator(
        model,
        tokenizer,
        max_seq_length=args.max_seq_length,
        resize=args.vision_resize,
        train_on_responses_only=True,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
        completion_only_loss=True,
    )
    preflight_summary: dict[str, int] | None = None
    if args.validate_token_budget:
        preflight_summary = token_budget_preflight(
            train_dataset + eval_dataset,
            collator=collator,
            max_seq_length=args.max_seq_length,
            safety_margin=args.token_budget_safety_margin,
        )
        print(
            f"[preflight] {json.dumps(preflight_summary, sort_keys=True)}",
            flush=True,
        )
    metrics_callback = ABCMonolithMetricsCallback(
        eval_examples=eval_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
        max_samples=args.eval_generation_max_samples,
        max_new_tokens=args.eval_generation_max_new_tokens,
        batch_size=args.eval_generation_batch_size,
        use_cache=args.eval_generation_use_cache,
        enable_thinking=args.generation_enable_thinking,
        candidate_scoring=args.monolith_candidate_scoring,
    )
    guard = ABCMonolithUnslothEnvGuardCallback()
    checkpoint = EpochCheckpointCallback(
        output_dir=args.output_dir,
        eval_epochs=args.eval_epochs,
        checkpoint_epochs=args.checkpoint_epochs,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.greater_is_better,
        keep_last_checkpoints=args.keep_last_checkpoints,
        keep_best_checkpoints=args.keep_best_checkpoints,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_threshold=args.early_stopping_threshold,
        wandb_log_best_checkpoints=args.wandb_log_best_checkpoints,
        wandb_artifact_prefix=args.wandb_artifact_prefix,
        generation_eval_args=argparse.Namespace(
            eval_generation_max_samples=args.eval_generation_max_samples,
            generation_eval_mode="inline",
        ),
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=hf_train,
        eval_dataset=hf_eval,
        data_collator=collator,
        args=SFTConfig(**trainer_args),
        callbacks=[metrics_callback, checkpoint, guard],
    )
    metrics_callback.trainer = trainer
    checkpoint.trainer = trainer
    guard.trainer = trainer
    trainer.abc_monolith_metrics_callback = metrics_callback
    trainer.abc_monolith_token_preflight = preflight_summary
    attach_cluster_checkpointing(trainer, args)
    return trainer


def _summary(examples: list[TrainingExample]) -> dict[str, Any]:
    return {
        "num_examples": len(examples),
        "num_videos": len({example.video_id for example in examples}),
        "labels": dict(sorted(Counter(str(example.label) for example in examples).items())),
        "families": dict(
            sorted(
                Counter(
                    str(example.metadata.get("cohort_family", "unspecified"))
                    for example in examples
                ).items()
            )
        ),
    }


def main() -> None:
    configure_kernel_caches()
    disable_unsloth_extra_outputs()
    args = parse_args()
    provider = EgoPERProvider(args.egoper_root, require_existing_videos=True)
    records = list(provider.iter_video_records())[: args.max_videos]
    split = _load_fixed_split(args, records)
    train_ids = set(split["train_video_ids"])
    val_ids = set(split["val_video_ids"])
    native_module_a_config = ModuleAConfig(
        stride_seconds=5.0,
        reset_jitter_seconds=0.0,
        seed=args.seed,
        label_mode="step_id",
    )
    native_b = build_module_b_training_dataset(
        provider,
        config=ModuleBConfig(
            late_trigger_probability=0.10,
            include_incomplete_negatives=True,
            negative_ratio=0.10,
            seed=args.seed,
        ),
        module_a_config=native_module_a_config,
    ).examples
    native_c = build_module_c_training_dataset(
        provider,
        config=ModuleCConfig(
            min_duration_seconds=0.5,
            jitter_ratio_min=0.05,
            jitter_ratio_max=0.10,
            include_not_completed_examples=True,
            not_completed_ratio=0.10,
            history_max_events=args.history_max_events,
            seed=args.seed,
        ),
    ).examples
    sampled_incomplete = [
        example
        for example in [*native_b, *native_c]
        if (
            str(example.label) in {"NO_ACTION", "NOT_COMPLETED"}
            or bool(example.metadata.get("not_completed"))
        )
    ]
    train_records = [record for record in records if record.video_id in train_ids]
    train_examples = _build_examples(
        train_records,
        args,
        training=True,
        sampled_incomplete_examples=[
            example for example in sampled_incomplete if example.video_id in train_ids
        ],
    )
    pre_sampling_examples = _build_examples(
        train_records,
        args,
        training=False,
        sampled_incomplete_examples=[
            example for example in sampled_incomplete if example.video_id in train_ids
        ],
    )
    val_records = [record for record in records if record.video_id in val_ids]
    from ..windowing.abc_monolith import (
        build_oracle_projection_examples,
        materialize_validation_manifest,
    )

    val_examples = build_oracle_projection_examples(
        val_records,
        provider=provider,
        seed=args.seed,
        max_history=args.history_max_events,
        native_b=native_b,
        native_c=native_c,
    )
    if not train_examples or not val_examples:
        raise ValueError("Fixed split produced an empty monolith train or validation set.")
    train_rows = [
        training_example_to_conversation(
            example, fps=args.fps, min_frames=args.min_frames, max_frames=args.max_frames
        )
        for example in train_examples
    ]
    val_rows = [
        training_example_to_conversation(
            example,
            fps=args.fps,
            min_frames=args.min_frames,
            max_frames=args.eval_max_frames,
        )
        for example in val_examples
    ]
    summary = {
        "split_file": args.split_file,
        "train": _summary(train_examples),
        "train_pre_sampling": _summary(pre_sampling_examples),
        "validation": _summary(val_examples),
        "validation_cohorts": materialize_validation_manifest(val_examples),
        "args": json_safe(vars(args)),
    }
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if args.dry_run:
        for row in train_rows[: max(0, args.print_examples)]:
            print(json.dumps(json_safe(row), indent=2, ensure_ascii=False))
        return
    if is_distributed_rank_zero():
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "run_config.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (output / "oracle_validation_cohorts.json").write_text(
            json.dumps(summary["validation_cohorts"], indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
    apply_resume_checkpoint(args)
    model, tokenizer = load_unsloth_model(args)
    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_rows,
        eval_dataset=val_rows,
        args=args,
    )
    if args.memory_preflight:
        all_rows = train_rows + val_rows
        token_audit = trainer.abc_monolith_token_preflight
        if not isinstance(token_audit, dict):
            raise RuntimeError("Memory preflight requires the processor token audit.")
        worst_index = int(token_audit["max_row_index"])
        if not 0 <= worst_index < len(all_rows):
            raise RuntimeError("Processor token audit returned an invalid worst-row index.")
        worst_case_row = all_rows[worst_index]
        longest_eval = max(
            val_rows,
            key=lambda row: float(row["window_end"]) - float(row["window_start"]),
        )
        audit = forward_backward_memory_preflight(
            model=model,
            collator=trainer.data_collator,
            metrics_callback=trainer.abc_monolith_metrics_callback,
            train_example=worst_case_row,
            eval_example=longest_eval,
        )
        if is_distributed_rank_zero():
            audit_path = Path(args.output_dir) / "memory_preflight.json"
            audit_path.write_text(
                json.dumps(json_safe(audit), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(f"[preflight] {json.dumps(audit, sort_keys=True)}", flush=True)
        if args.memory_preflight_only:
            return
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    if trainer.is_world_process_zero():
        save_outputs(model, tokenizer, args.output_dir)


if __name__ == "__main__":
    main()
