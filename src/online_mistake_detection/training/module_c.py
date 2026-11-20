from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import math
import os
import re
import random
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

os.environ["UNSLOTH_RETURN_LOGITS"] = "0"
os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "1"

from ..config import ModuleCConfig
from ..datasets import (
    EgoPERProvider,
    ModuleTrainingDataset,
    build_module_c_training_dataset,
)
from ..datasets.egoper import EGOPER_AGGREGATE_MAPPING, EGOPER_MODULE_C_CLASSES
from ..models.vision import vision_resize_arg
from ..prompts.contracts import NOT_COMPLETED_STATUS
from ..prompts.module_c import (
    MODULE_C_SYSTEM_PROMPT,
    append_module_c_history,
)
from ..metrics.classification import (
    module_c_metrics as multiclass_metrics,
    parse_module_c_prediction as parse_classification_prediction,
)
from ..paths import egoper_root
from ..schemas import TrainingExample

from .distributed_eval import (
    distributed_rank_and_world,
    gather_round_robin_shards,
    shard_examples_for_rank,
    unwrap_trainer_model,
)
from .kernel_cache import configure_kernel_caches
from .preemption import add_preemption_args, apply_resume_checkpoint, attach_cluster_checkpointing
from .module_a import (
    EpochCheckpointCallback,
    Qwen3VLMetadataVisionDataCollator,
    apply_video_split_subset,
    collapse_fps,
    default_split_file as shared_default_split_file,
    estimated_window_frames,
    grouped_video_ids,
    inference_runtime_metrics,
    dummy_train_rows_for_eval_only,
    is_distributed_rank_zero,
    json_safe,
    last_hidden_states,
    load_generation_eval_model,
    load_unsloth_model,
    scalar_metrics_for_logging,
    make_video_split,
    release_cuda_memory,
    response_token_loss,
    save_outputs,
    split_file_lock,
    unwrap_output_embeddings,
    validate_lora_target_configuration,
    validate_model_precision,
    validate_video_split,
)

MODULE_C_SPLIT_VERSION = "module_c_video_split_v1"


def module_c_candidate_target(class_name: str) -> str:
    """Canonical short response used for deterministic forced-choice scoring."""

    return json.dumps(
        {"classification": class_name, "reasoning": ""},
        separators=(",", ":"),
    )


def module_c_system_prompt(dataset_name: str) -> str:
    return MODULE_C_SYSTEM_PROMPT


def taxonomy_for_dataset(dataset_name: str) -> tuple[tuple[str, ...], dict[str, Any]]:
    return EGOPER_MODULE_C_CLASSES, EGOPER_AGGREGATE_MAPPING


def oversample_non_correct_examples(
    examples: list[TrainingExample],
    *,
    factor: float,
    seed: int,
) -> list[TrainingExample]:
    """Repeat completed non-CORRECT outcomes in training only."""
    if factor <= 1.0:
        return list(examples)
    candidates = [
        example
        for example in examples
        if example.fine_class not in {None, "CORRECT", NOT_COMPLETED_STATUS}
    ]
    if not candidates:
        return list(examples)

    whole_repeats = int(math.floor(factor)) - 1
    fractional_count = round((factor - math.floor(factor)) * len(candidates))
    extras = [
        copy.deepcopy(example)
        for _ in range(whole_repeats)
        for example in candidates
    ]
    if fractional_count:
        rng = random.Random(seed)
        extras.extend(
            copy.deepcopy(example)
            for example in rng.sample(candidates, fractional_count)
        )
    random.Random(seed + 1).shuffle(extras)
    return list(examples) + extras


def apply_module_c_history_dropout(
    examples: list[TrainingExample],
    *,
    probability: float,
    max_events: int,
    seed: int,
) -> list[TrainingExample]:
    """Drop the complete prior-action history from training examples only."""
    if max_events <= 0 or probability <= 0.0:
        return list(examples)
    rng = random.Random(seed)
    result: list[TrainingExample] = []
    for example in examples:
        if example.source_dataset != "egoper":
            result.append(example)
            continue
        metadata = dict(example.metadata)
        base_prompt = metadata.get("module_c_prompt_without_history")
        if not isinstance(base_prompt, str):
            result.append(example)
            continue
        dropped = rng.random() < probability
        if not dropped:
            result.append(example)
            continue
        metadata["module_c_history_mode"] = "dropped"
        result.append(
            replace(
                example,
                prompt_text=append_module_c_history(
                    base_prompt,
                    None,
                    max_events=max_events,
                ),
                metadata=metadata,
            )
        )
    return result


def disable_unsloth_extra_outputs() -> None:
    was_logits = os.environ.get("UNSLOTH_RETURN_LOGITS")
    was_hidden = os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES")
    if was_logits == "1" or was_hidden == "1":
        print(
            "Module C disabled leaked Unsloth extra outputs: "
            f"UNSLOTH_RETURN_LOGITS={was_logits}, "
            f"UNSLOTH_RETURN_HIDDEN_STATES={was_hidden}"
        )
    os.environ["UNSLOTH_RETURN_LOGITS"] = "0"
    os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "0"


def enable_module_c_training_hidden_states() -> None:
    """Return final hidden states, never full-sequence vocabulary logits."""
    os.environ["UNSLOTH_RETURN_LOGITS"] = "0"
    os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "1"


def module_c_response_only_loss(
    model: Any,
    inputs: dict[str, Any],
    *,
    trainer: Any | None = None,
) -> Any:
    """Compute CE after selecting response positions, avoiding full-sequence logits."""
    for key in (
        "response_loss_weight",
        "eval_row_index",
        "eval_answer_positions",
        "eval_candidate_token_ids",
    ):
        inputs.pop(key, None)
    labels = inputs.pop("labels", None)
    if labels is None:
        raise RuntimeError("Module C response-only loss requires labels.")
    outputs = model(
        **inputs,
        output_hidden_states=True,
        return_dict=True,
        use_cache=False,
    )
    return response_token_loss(
        model,
        outputs,
        labels,
        trainer=trainer,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Qwen VL for Module C mistake reasoning.")
    parser.add_argument("--dataset", choices=("egoper",), default="egoper")
    parser.add_argument("--egoper-root", default=str(egoper_root()))
    parser.add_argument("--video-ids", nargs="*", default=None)
    parser.add_argument("--task-ids", nargs="*", default=None)
    parser.add_argument("--max-videos", type=int, default=50)
    parser.add_argument("--model-name", default="unsloth/Qwen3.5-2B")
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--load-in-8bit", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--load-in-16bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--lora-dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--finetune-vision-layers", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--finetune-language-layers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--finetune-attention-modules", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--finetune-mlp-modules", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument(
        "--lora-target-modules",
        choices=("auto", "all-linear"),
        default="auto",
        help=(
            "LoRA target selection. 'auto' respects finetune_* flags. "
            "'all-linear' targets every linear layer and requires all finetune_* flags to be true."
        ),
    )
    parser.add_argument("--train-mode", choices=["steps", "epochs"], default="steps")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--lr-scheduler-type", default="linear")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--optim",
        default="adamw_torch_fused",
        help="HF Trainer optimizer. Default fused AdamW; use adamw_8bit only if optimizer memory is tight.",
    )
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--checkpoint-epochs", type=int, default=1)
    parser.add_argument("--eval-epochs", type=int, default=1)
    parser.add_argument("--keep-last-checkpoints", type=int, default=4)
    parser.add_argument("--keep-best-checkpoints", type=int, default=4)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--early-stopping-threshold", type=float, default=0.0)
    parser.add_argument("--metric-for-best-model", default="eval_fine/macro_f1")
    parser.add_argument(
        "--legacy-binary-parser",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Temporary compatibility parser for pre-multiclass checkpoints.",
    )
    parser.add_argument("--greater-is-better", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument(
        "--train-videos-per-task",
        type=int,
        default=None,
        help="Optional deterministic cap per task from the official training split.",
    )
    parser.add_argument(
        "--train-validation-erroneous-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="For capped EgoPER smoke runs, choose train/validation videos only from erroneous videos.",
    )
    parser.add_argument("--val-videos-per-task", type=int, default=2)
    parser.add_argument("--split-file", default=None)
    parser.add_argument("--regenerate-split", action="store_true")
    parser.add_argument("--module-c-min-duration-seconds", type=float, default=0.5)
    parser.add_argument("--module-c-jitter-ratio-min", type=float, default=0.05)
    parser.add_argument("--module-c-jitter-ratio-max", type=float, default=0.10)
    parser.add_argument(
        "--include-not-completed-examples",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--module-c-not-completed-ratio", type=float, default=0.10)
    parser.add_argument(
        "--module-c-history-max-events",
        type=int,
        default=12,
        help="Maximum prior completed actions included as text in each Module C prompt.",
    )
    parser.add_argument(
        "--module-c-history-dropout-probability",
        type=float,
        default=0.25,
        help="Training-only probability of replacing the complete action history with unavailable.",
    )
    parser.add_argument(
        "--module-c-non-correct-oversample-factor",
        type=float,
        default=1.0,
        help=(
            "Training-only multiplicity for completed outcomes other than CORRECT. "
            "Does not oversample NOT_COMPLETED and never changes validation/test."
        ),
    )
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--min-frames", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--eval-max-frames", type=int, default=None)
    parser.add_argument("--vision-resize", type=vision_resize_arg, default=512)
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--eval-generation-max-samples", type=int, default=-1)
    parser.add_argument("--eval-generation-max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--eval-generation-batch-size",
        type=int,
        default=1,
        help="Examples generated together per rank during classification evaluation.",
    )
    parser.add_argument(
        "--eval-generation-use-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the model cache instead of recomputing the video prompt for every output token.",
    )
    parser.add_argument(
        "--generation-enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow reasoning before the required JSON response.",
    )
    parser.add_argument(
        "--zero-shot-eval",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Run the same generation classification validation as training eval on "
            "the base model (identity LoRA, no optimizer steps), then exit."
        ),
    )
    parser.add_argument(
        "--generation-eval-only",
        action="store_true",
        help="Skip training and run Module C generation metrics on the validation split only.",
    )
    parser.add_argument(
        "--eval-checkpoint",
        default=None,
        help="LoRA checkpoint directory for --generation-eval-only.",
    )
    parser.add_argument(
        "--generation-eval-output-json",
        default=None,
        help="Optional path to write generation-eval-only metrics JSON.",
    )
    parser.add_argument(
        "--bertscore-model-type",
        default="microsoft/deberta-xlarge-mnli",
        help=(
            "English encoder used for BERTScore in zero-shot and "
            "generation-eval-only runs."
        ),
    )
    parser.add_argument(
        "--bertscore-batch-size",
        type=int,
        default=16,
        help="Text-pair batch size for eval-only BERTScore computation.",
    )
    parser.add_argument(
        "--bertscore-device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device used by BERTScore; auto uses CUDA when available.",
    )
    parser.add_argument("--output-dir", default="outputs/module_c_qwen35_lora_reasoning")
    parser.add_argument("--report-to", default="tensorboard")
    parser.add_argument("--dataset-num-proc", type=int, default=1)
    parser.add_argument(
        "--torch-empty-cache-steps",
        type=int,
        default=1,
        help=(
            "Call torch.cuda.empty_cache() every N optimizer steps; 0 disables "
            "it. Every step was the default when the 35B MoE ran at the memory "
            "ceiling, but it synchronizes and drops the caching allocator's "
            "blocks, so it costs real throughput once memory is not scarce."
        ),
    )
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument(
        "--dataloader-prefetch-factor",
        type=int,
        default=None,
        help="Batches prefetched per worker. None uses the HuggingFace/PyTorch default.",
    )
    parser.add_argument(
        "--group-by-length",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Group similar frame-count examples to reduce DDP rank stragglers.",
    )
    parser.add_argument(
        "--ddp-find-unused-parameters",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Ask DDP to traverse and retain model outputs while discovering unused "
            "parameters. Keep disabled for Module C LoRA: the DDP output sink otherwise "
            "clones the full [sequence, vocabulary] logits tensor."
        ),
    )
    parser.add_argument("--ddp-timeout", type=int, default=7200)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-examples", type=int, default=2)
    parser.add_argument("--resume-from-checkpoint", default=None)
    add_preemption_args(parser)
    parser.add_argument("--wandb-log-best-checkpoints", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wandb-artifact-prefix", default="module-c")
    args = parser.parse_args()
    args.report_to = [item.strip() for item in args.report_to.split(",") if item.strip()]
    if args.max_videos <= 0:
        parser.error("--max-videos must be positive.")
    if args.eval_max_frames is None:
        args.eval_max_frames = args.max_frames
    if args.max_frames < args.min_frames or args.eval_max_frames < args.min_frames:
        parser.error("--max-frames and --eval-max-frames must be >= --min-frames.")
    if args.module_c_min_duration_seconds <= 0:
        parser.error("--module-c-min-duration-seconds must be positive.")
    if args.module_c_jitter_ratio_min < 0 or args.module_c_jitter_ratio_max < args.module_c_jitter_ratio_min:
        parser.error("Module C jitter ratios must be non-negative and min <= max.")
    if args.module_c_non_correct_oversample_factor < 1.0:
        parser.error("--module-c-non-correct-oversample-factor must be >= 1.")
    if args.module_c_history_max_events < 0:
        parser.error("--module-c-history-max-events must be non-negative.")
    if not 0.0 <= args.module_c_history_dropout_probability <= 1.0:
        parser.error("--module-c-history-dropout-probability must be between 0 and 1.")
    if args.keep_last_checkpoints < 0 or args.keep_best_checkpoints < 0:
        parser.error("checkpoint retention counts cannot be negative.")
    if args.eval_generation_max_samples < -1:
        parser.error("--eval-generation-max-samples must be -1, 0, or positive.")
    if args.eval_generation_batch_size < 1:
        parser.error("--eval-generation-batch-size must be positive.")
    if args.bertscore_batch_size < 1:
        parser.error("--bertscore-batch-size must be positive.")
    if args.max_grad_norm <= 0:
        parser.error("--max-grad-norm must be positive.")
    if args.zero_shot_eval and args.generation_eval_only:
        parser.error("--zero-shot-eval cannot be combined with --generation-eval-only.")
    if args.zero_shot_eval and args.dry_run:
        parser.error("--zero-shot-eval cannot be combined with --dry-run.")
    if args.zero_shot_eval and args.eval_generation_max_samples == 0:
        parser.error("--zero-shot-eval requires --eval-generation-max-samples != 0.")
    if args.generation_eval_only and args.dry_run:
        parser.error("--generation-eval-only cannot be combined with --dry-run.")
    validate_generation_metric_is_reachable(args, parser.error)
    validate_model_precision(args, parser.error)
    validate_lora_target_configuration(args, parser.error)
    return args


MODULE_C_GENERATION_METRIC_PREFIXES = (
    "eval_all/",
    "eval_fine/",
    "eval_aggregate/",
    "eval_completed/",
    "eval_completed_conditional/",
    "eval_not_completed/",
    "eval_reasoning/",
    "deprecated_eval_mistake/",
)


def validate_generation_metric_is_reachable(
    args: argparse.Namespace,
    fail: Any,
) -> None:
    """Reject a --metric-for-best-model the run can never log."""
    metric = str(args.metric_for_best_model)
    if metric.endswith("loss"):
        return
    if not metric.startswith(MODULE_C_GENERATION_METRIC_PREFIXES):
        fail(
            f"--metric-for-best-model {metric!r} is never logged by Module C. "
            f"Use eval_loss or a key starting with one of: "
            f"{', '.join(MODULE_C_GENERATION_METRIC_PREFIXES)}. "
            "eval_fine/present_class_macro_f1 averages only over classes "
            "that actually appear in validation."
        )
    if args.eval_generation_max_samples == 0:
        fail(
            f"--metric-for-best-model {metric!r} comes from the generation eval, "
            "but --eval-generation-max-samples is 0, which disables it. The run "
            "would train to completion and never save a best checkpoint. Pass -1 "
            "to score every validation example, or select eval_loss."
        )


def build_module_c_dataset(args: argparse.Namespace) -> ModuleTrainingDataset:
    config = ModuleCConfig(
        min_duration_seconds=args.module_c_min_duration_seconds,
        jitter_ratio_min=args.module_c_jitter_ratio_min,
        jitter_ratio_max=args.module_c_jitter_ratio_max,
        include_not_completed_examples=args.include_not_completed_examples,
        not_completed_ratio=args.module_c_not_completed_ratio,
        history_max_events=args.module_c_history_max_events,
        seed=args.seed,
    )
    provider = EgoPERProvider(
        args.egoper_root,
        video_ids=set(args.video_ids) if args.video_ids else None,
        task_ids=set(args.task_ids) if args.task_ids else None,
        require_existing_videos=True,
    )
    return build_module_c_training_dataset(
        provider,
        config=config,
        max_videos=args.max_videos,
    )


def default_split_file(args: argparse.Namespace) -> Path:
    return shared_default_split_file(args)


def split_policy(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "type": "per_task_video_holdout",
        "val_fraction": args.val_fraction,
        "val_videos_per_task": args.val_videos_per_task,
    }


def data_config_for_split(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "max_videos": args.max_videos,
        "task_ids": sorted(args.task_ids or []),
        "video_ids": sorted(args.video_ids or []),
        "module_c_min_duration_seconds": args.module_c_min_duration_seconds,
        "module_c_jitter_ratio_min": args.module_c_jitter_ratio_min,
        "module_c_jitter_ratio_max": args.module_c_jitter_ratio_max,
        "include_not_completed_examples": args.include_not_completed_examples,
        "module_c_not_completed_ratio": args.module_c_not_completed_ratio,
        "module_c_history_max_events": args.module_c_history_max_events,
    }


def make_module_c_video_split(dataset: ModuleTrainingDataset, args: argparse.Namespace) -> dict[str, Any]:
    grouped = grouped_video_ids(dataset)
    if not grouped:
        raise ValueError("Cannot create a split from an empty dataset.")
    if any(record.split in {"train", "validation"} for record in dataset.records):
        return make_video_split(dataset, args)

    import random

    rng = random.Random(args.seed)
    tasks: dict[str, dict[str, list[str]]] = {}
    train_video_ids: list[str] = []
    val_video_ids: list[str] = []
    for task_id, video_ids in grouped.items():
        shuffled = list(video_ids)
        rng.shuffle(shuffled)
        if len(shuffled) < 2:
            raise ValueError(f"Task {task_id!r} has too few videos for validation splitting.")
        val_count = min(args.val_videos_per_task, len(shuffled) - 1)
        if len(grouped) != 5 or len(shuffled) != 10:
            val_count = max(1, round(len(shuffled) * args.val_fraction))
            val_count = min(val_count, len(shuffled) - 1)
        val = sorted(shuffled[:val_count])
        train = sorted(shuffled[val_count:])
        tasks[task_id] = {"train": train, "val": val}
        train_video_ids.extend(train)
        val_video_ids.extend(val)

    return {
        "version": MODULE_C_SPLIT_VERSION,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "seed": args.seed,
        "split_policy": split_policy(args),
        "data_config": data_config_for_split(args),
        "train_video_ids": sorted(train_video_ids),
        "val_video_ids": sorted(val_video_ids),
        "tasks": tasks,
    }


def load_or_create_video_split(dataset: ModuleTrainingDataset, args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    split_path = Path(args.split_file) if args.split_file else default_split_file(args)
    with split_file_lock(split_path):
        if split_path.exists() and not args.regenerate_split:
            with open(split_path, "r", encoding="utf-8") as file:
                split = json.load(file)
            validate_video_split(split, dataset)
            return apply_video_split_subset(split, dataset, args), split_path
        split = make_module_c_video_split(dataset, args)
        validate_video_split(split, dataset)
        with open(split_path, "w", encoding="utf-8") as file:
            json.dump(split, file, indent=2, sort_keys=True)
            file.write("\n")
    return apply_video_split_subset(split, dataset, args), split_path


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
                "content": [
                    {
                        "type": "text",
                        "text": module_c_system_prompt(example.source_dataset),
                    }
                ],
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
            {"role": "assistant", "content": [{"type": "text", "text": example.target_text}]},
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
        "mistake": example.mistake,
        "reasoning": example.reasoning,
        "metadata": example.metadata,
    }


def conversations_for_video_ids(
    dataset: ModuleTrainingDataset,
    video_ids: set[str],
    *,
    fps: float,
    min_frames: int,
    max_frames: int,
) -> list[dict[str, Any]]:
    return [
        training_example_to_conversation(example, fps=fps, min_frames=min_frames, max_frames=max_frames)
        for example in dataset
        if example.video_id in video_ids
    ]


def summarize_dataset(dataset: ModuleTrainingDataset) -> dict[str, Any]:
    labels = Counter(example.label for example in dataset)
    return {
        "num_examples": len(dataset),
        "num_videos": len({record.video_id for record in dataset.records}),
        "label_counts": dict(sorted(labels.items())),
        "tasks": {
            task_id: len(video_ids)
            for task_id, video_ids in grouped_video_ids(dataset).items()
        },
    }


def summarize_conversations(conversations: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "num_examples": len(conversations),
        "num_videos": len({item["video_id"] for item in conversations}),
        "label_counts": dict(sorted(Counter(str(item.get("label")) for item in conversations).items())),
    }


def save_run_config(
    *,
    args: argparse.Namespace,
    dataset: ModuleTrainingDataset,
    train_conversations: list[dict[str, Any]],
    val_conversations: list[dict[str, Any]],
    split: dict[str, Any],
    split_path: Path,
) -> None:
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    payload = {
        "args": json_safe(vars(args)),
        "dataset_summary": summarize_dataset(dataset),
        "num_conversations": len(train_conversations) + len(val_conversations),
        "split_file": str(split_path),
        "split": json_safe(split),
        "train_summary": summarize_conversations(train_conversations),
        "validation_summary": summarize_conversations(val_conversations),
    }
    config_path = output_path / "run_config.json"
    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True, ensure_ascii=False)
        file.write("\n")
    print(f"Saved run config to: {config_path}")


def parse_bool_value(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1", "mistake"}:
            return True
        if normalized in {"false", "no", "0", "correct"}:
            return False
    return None


def parse_module_c_prediction(
    text: str,
    *,
    allow_legacy_binary: bool = False,
    allowed_classes: Any = EGOPER_MODULE_C_CLASSES,
    aggregate_mapping: Any = EGOPER_AGGREGATE_MAPPING,
) -> dict[str, Any] | None:
    parsed = parse_classification_prediction(
        text,
        allowed_classes=allowed_classes,
        allow_legacy_binary=False,
    )
    if parsed.valid:
        classification = str(parsed.classification)
        if classification == NOT_COMPLETED_STATUS:
            return {
                "classification": classification,
                "mistake": False,
                "reasoning": parsed.reasoning,
                "legacy": False,
            }
        return {
            "classification": classification,
            "mistake": aggregate_mapping[classification] == "MISTAKE",
            "reasoning": parsed.reasoning,
            "legacy": False,
        }
    if not allow_legacy_binary:
        return None
    match = re.search(r"\{.*\}", text.strip(), flags=re.DOTALL)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    mistake = parse_bool_value(payload.get("mistake"))
    reasoning = payload.get("reasoning")
    if mistake is None or not isinstance(reasoning, str) or not reasoning.strip():
        return None
    return {
        "classification": "MISTAKE" if mistake else "CORRECT",
        "mistake": mistake,
        "reasoning": reasoning.strip(),
        "legacy": True,
    }


def module_c_multiclass_metrics(
    targets: list[str],
    predictions: list[str | None],
    *,
    fine_classes: Any = EGOPER_MODULE_C_CLASSES,
    aggregate_mapping: Any = EGOPER_AGGREGATE_MAPPING,
) -> dict[str, Any]:
    nested = multiclass_metrics(
        targets,
        predictions,
        fine_classes=fine_classes,
        aggregate_mapping=aggregate_mapping,
    )
    flat: dict[str, Any] = {
        "eval_all/macro_f1": nested["all"]["macro_f1"],
        "eval_all/present_class_macro_f1": nested["all"][
            "present_class_macro_f1"
        ],
        "eval_all/weighted_f1": nested["all"]["weighted_f1"],
        "eval_all/invalid_rate": nested["all"]["invalid_rate"],
        "eval_all/confusion_matrix": nested["all"]["confusion_matrix"],
        "eval_fine/macro_f1": nested["fine"]["macro_f1"],
        "eval_fine/present_class_macro_f1": nested["fine"][
            "present_class_macro_f1"
        ],
        "eval_fine/weighted_f1": nested["fine"]["weighted_f1"],
        "eval_fine/invalid_rate": nested["fine"]["invalid_rate"],
        "eval_fine/confusion_matrix": nested["fine"]["confusion_matrix"],
        "eval_aggregate/macro_f1": nested["aggregate"]["macro_f1"],
        "eval_aggregate/weighted_f1": nested["aggregate"]["weighted_f1"],
        "eval_aggregate/confusion_matrix": nested["aggregate"]["confusion_matrix"],
        "deprecated_eval_mistake/f1": nested["deprecated_binary"]["f1"],
        "eval_not_completed/precision": nested["not_completed"]["precision"],
        "eval_not_completed/recall": nested["not_completed"]["recall"],
        "eval_not_completed/f1": nested["not_completed"]["f1"],
        "eval_completed/macro_f1": nested["completed"]["macro_f1"],
        "eval_completed_conditional/macro_f1": nested["completed_conditional"]["macro_f1"],
    }
    for class_name, values in nested["all"]["per_class"].items():
        for metric_name, value in values.items():
            flat[f"eval_all/{class_name}/{metric_name}"] = value
    for class_name, values in nested["fine"]["per_class"].items():
        for metric_name, value in values.items():
            flat[f"eval_fine/{class_name}/{metric_name}"] = value
    for class_name, values in nested["aggregate"]["per_class"].items():
        for metric_name, value in values.items():
            flat[f"eval_aggregate/{class_name}/{metric_name}"] = value
    return flat


def _token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def lexical_overlap(reference: str, prediction: str) -> float:
    reference_tokens = _token_set(reference)
    prediction_tokens = _token_set(prediction)
    if not reference_tokens and not prediction_tokens:
        return 1.0
    if not reference_tokens or not prediction_tokens:
        return 0.0
    return len(reference_tokens & prediction_tokens) / len(reference_tokens | prediction_tokens)


def module_c_reasoning_metrics(
    target_reasonings: list[str],
    predicted_reasonings: list[str | None],
    *,
    total_examples: int | None = None,
    compute_bertscore: bool = False,
    bertscore_model_type: str = "microsoft/deberta-xlarge-mnli",
    bertscore_batch_size: int = 16,
    bertscore_device: str = "auto",
) -> tuple[dict[str, float], str | None]:
    total = len(target_reasonings) if total_examples is None else total_examples
    nonempty_predictions = sum(
        prediction is not None and bool(prediction.strip())
        for prediction in predicted_reasonings
    )
    valid_reasonings = [
        (target.strip(), prediction.strip())
        for target, prediction in zip(target_reasonings, predicted_reasonings)
        if target.strip() and prediction is not None and prediction.strip()
    ]
    metrics = {
        "eval_reasoning/nonempty_rate": (
            nonempty_predictions / total if total else 0.0
        ),
        "eval_reasoning/lexical_overlap": (
            sum(
                lexical_overlap(target, prediction)
                for target, prediction in valid_reasonings
            )
            / len(valid_reasonings)
            if valid_reasonings
            else 0.0
        ),
    }
    if not compute_bertscore:
        return metrics, None

    metrics["eval_reasoning/bertscore_num_scored"] = float(len(valid_reasonings))
    if not valid_reasonings:
        metrics.update({
            "eval_reasoning/bertscore_precision": 0.0,
            "eval_reasoning/bertscore_recall": 0.0,
            "eval_reasoning/bertscore_f1": 0.0,
        })
        return metrics, None

    try:
        from bert_score import score as bert_score
    except ImportError as exc:
        raise RuntimeError(
            "BERTScore evaluation requires bert-score==0.3.13. "
            "Rebuild the evaluation image or install the project requirements."
        ) from exc

    references = [target for target, _ in valid_reasonings]
    candidates = [prediction for _, prediction in valid_reasonings]
    device = None if bertscore_device == "auto" else bertscore_device
    (precision, recall, f1), score_hash = bert_score(
        candidates,
        references,
        model_type=bertscore_model_type,
        batch_size=bertscore_batch_size,
        device=device,
        verbose=False,
        return_hash=True,
    )
    metrics.update({
        "eval_reasoning/bertscore_precision": float(precision.mean().item()),
        "eval_reasoning/bertscore_recall": float(recall.mean().item()),
        "eval_reasoning/bertscore_f1": float(f1.mean().item()),
    })
    return metrics, str(score_hash)


def module_c_metrics(
    targets: list[bool],
    predictions: list[bool | None],
    *,
    target_reasonings: list[str] | None = None,
    predicted_reasonings: list[str | None] | None = None,
) -> dict[str, float]:
    tp = fp = tn = fn = invalid = 0
    for target, prediction in zip(targets, predictions):
        if prediction is None:
            invalid += 1
            if target:
                fn += 1
            else:
                fp += 1
        elif target and prediction:
            tp += 1
        elif not target and prediction:
            fp += 1
        elif not target and not prediction:
            tn += 1
        else:
            fn += 1

    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    accuracy = (tp + tn) / total if total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / denominator if denominator else 0.0
    metrics = {
        "eval_mistake/accuracy": accuracy,
        "eval_mistake/precision": precision,
        "eval_mistake/recall": recall,
        "eval_mistake/f1": f1,
        "eval_mistake/specificity": specificity,
        "eval_mistake/balanced_accuracy": (recall + specificity) / 2.0,
        "eval_mistake/mcc": mcc,
        "eval_mistake/false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "eval_mistake/false_negative_rate": fn / (fn + tp) if fn + tp else 0.0,
        "eval_mistake/invalid_json_rate": invalid / total if total else 0.0,
        "eval_mistake/tp": float(tp),
        "eval_mistake/fp": float(fp),
        "eval_mistake/tn": float(tn),
        "eval_mistake/fn": float(fn),
        "eval_mistake/num_samples": float(total),
    }
    if target_reasonings is not None and predicted_reasonings is not None:
        reasoning_metrics, _ = module_c_reasoning_metrics(
            target_reasonings,
            predicted_reasonings,
            total_examples=total,
        )
        metrics.update(reasoning_metrics)
    return metrics


def _broadcast_rank_zero_object(value: Any, *, rank: int) -> Any:
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        return value
    values = [value if rank == 0 else None]
    dist.broadcast_object_list(values, src=0)
    return values[0]


class ModuleCMistakeMetricsCallback:
    def __init__(
        self,
        *,
        eval_examples: list[dict[str, Any]],
        data_collator: Qwen3VLMetadataVisionDataCollator,
        tokenizer: Any,
        max_samples: int,
        max_new_tokens: int,
        enable_thinking: bool = False,
        batch_size: int = 1,
        use_cache: bool = True,
        allow_legacy_binary: bool = False,
        fine_classes: Any = EGOPER_MODULE_C_CLASSES,
        aggregate_mapping: Any = EGOPER_AGGREGATE_MAPPING,
        report_runtime: bool = False,
        compute_bertscore: bool = False,
        bertscore_model_type: str = "microsoft/deberta-xlarge-mnli",
        bertscore_batch_size: int = 16,
        bertscore_device: str = "auto",
    ) -> None:
        from transformers import TrainerCallback

        self._callback_base = TrainerCallback()
        self.trainer: Any | None = None
        self.eval_examples = eval_examples if max_samples < 0 else eval_examples[:max_samples]
        self.data_collator = data_collator
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.enable_thinking = enable_thinking
        self.batch_size = batch_size
        self.use_cache = use_cache
        self.allow_legacy_binary = allow_legacy_binary
        self.fine_classes = fine_classes
        self.aggregate_mapping = aggregate_mapping
        self.report_runtime = report_runtime
        self.compute_bertscore = compute_bertscore
        self.bertscore_model_type = bertscore_model_type
        self.bertscore_batch_size = bertscore_batch_size
        self.bertscore_device = bertscore_device
        self.bertscore_hash: str | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._callback_base, name)

    def on_evaluate(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        if self.trainer is None or not self.eval_examples:
            return control
        rank, world_size = distributed_rank_and_world()
        local_examples = shard_examples_for_rank(
            self.eval_examples,
            rank=rank,
            world_size=world_size,
        )
        if rank == 0:
            print(
                "Module C inline generation eval: "
                f"{len(self.eval_examples)} examples sharded across {world_size} GPU(s)."
            )
        model = unwrap_trainer_model(self.trainer, kwargs["model"])
        was_training = model.training
        local_rows: list[dict[str, Any]] = []

        import torch
        from unsloth import FastVisionModel

        FastVisionModel.for_inference(model)
        try:
            with torch.inference_mode():
                for start in range(0, len(local_examples), self.batch_size):
                    batch_examples = local_examples[start : start + self.batch_size]
                    if self.report_runtime and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    inference_started = (
                        time.perf_counter() if self.report_runtime else 0.0
                    )
                    generated_texts = self._generate_jsons(model, batch_examples)
                    if self.report_runtime and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    batch_response_seconds = (
                        time.perf_counter() - inference_started
                        if self.report_runtime
                        else 0.0
                    )
                    amortized_seconds = batch_response_seconds / len(batch_examples)
                    for example, generated_text in zip(batch_examples, generated_texts):
                        parsed = parse_module_c_prediction(
                            generated_text,
                            allow_legacy_binary=self.allow_legacy_binary,
                            allowed_classes=self.fine_classes,
                            aggregate_mapping=self.aggregate_mapping,
                        )
                        local_rows.append({
                            "prediction": None if parsed is None else str(parsed["classification"]),
                            "predicted_reasoning": (
                                None
                                if parsed is None or parsed["reasoning"] is None
                                else str(parsed["reasoning"])
                            ),
                            "raw_output": generated_text,
                            "target": str(example.get("label")),
                            "target_reasoning": str(example.get("reasoning") or ""),
                            "video_id": str(example.get("video_id", "")),
                            "task_id": str(example.get("task_id", "")),
                            "step_index": int(example.get("step_index", -1)),
                            "window_start": float(example.get("window_start", 0.0)),
                            "window_end": float(example.get("window_end", 0.0)),
                            "response_seconds": batch_response_seconds,
                            "amortized_inference_seconds": amortized_seconds,
                        })
        finally:
            if was_training:
                restore_module_c_training_state(model)
            disable_unsloth_extra_outputs()
            release_cuda_memory()

        rows = gather_round_robin_shards(local_rows)
        if not rows:
            return control
        if rank == 0:
            prediction_dir = Path(self.trainer.args.output_dir) / "eval_predictions"
            prediction_dir.mkdir(parents=True, exist_ok=True)
            prediction_path = prediction_dir / (
                f"step-{int(getattr(state, 'global_step', 0)):08d}.jsonl"
            )
            with open(prediction_path, "w", encoding="utf-8") as file:
                for row in rows:
                    file.write(json.dumps({
                        "video_id": row["video_id"],
                        "task_id": row["task_id"],
                        "step_index": row["step_index"],
                        "window_start": row["window_start"],
                        "window_end": row["window_end"],
                        "target": row["target"],
                        "target_reasoning": row["target_reasoning"],
                        "generated_text": row["raw_output"],
                        "parsed_classification": row["prediction"],
                        "parsed_reasoning": row["predicted_reasoning"],
                    }, ensure_ascii=False) + "\n")
            print(f"Wrote Module C eval predictions to: {prediction_path}", flush=True)
        predictions = [row["prediction"] for row in rows]
        targets = [str(row["target"]) for row in rows]
        predicted_reasonings = [row["predicted_reasoning"] for row in rows]
        target_reasonings = [str(row["target_reasoning"]) for row in rows]

        metrics = module_c_multiclass_metrics(
            targets,
            predictions,
            fine_classes=self.fine_classes,
            aggregate_mapping=self.aggregate_mapping,
        )
        reasoning_metrics, bertscore_hash = module_c_reasoning_metrics(
            target_reasonings,
            predicted_reasonings,
            total_examples=len(targets),
            compute_bertscore=self.compute_bertscore and rank == 0,
            bertscore_model_type=self.bertscore_model_type,
            bertscore_batch_size=self.bertscore_batch_size,
            bertscore_device=self.bertscore_device,
        )
        if self.compute_bertscore:
            reasoning_metrics, bertscore_hash = _broadcast_rank_zero_object(
                (reasoning_metrics, bertscore_hash),
                rank=rank,
            )
        metrics.update(reasoning_metrics)
        self.bertscore_hash = bertscore_hash
        if self.report_runtime:
            metrics.update(
                inference_runtime_metrics(
                    [float(row["response_seconds"]) for row in rows],
                    amortized_seconds=[
                        float(row["amortized_inference_seconds"])
                        for row in rows
                    ],
                )
            )
        callback_metrics = kwargs.get("metrics")
        if isinstance(callback_metrics, dict):
            callback_metrics.update(metrics)
        if rank == 0:
            self.trainer.log(metrics)
        return control

    def _generation_inputs(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        texts: list[str] = []
        images: list[Any] = []
        videos: list[Any] = []
        video_metadatas: list[Any] = []
        video_processor_kwargs: dict[str, Any] = {}
        fps_values: list[float] = []
        for example in examples:
            messages = [
                copy.deepcopy(message)
                for message in example["messages"]
                if message.get("role") != "assistant"
            ]
            texts.append(
                self.data_collator._base.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=self.enable_thinking,
                )
            )
            image, video, video_kwargs, video_metadata = (
                self.data_collator._extract_images_videos_metadata(messages)
            )
            if image:
                images.extend(image)
            if video:
                videos.extend(video)
                for key, value in video_kwargs.items():
                    if key == "fps":
                        fps_values.extend(value)
                    else:
                        video_processor_kwargs[key] = value
                if video_metadata:
                    video_metadatas.extend(video_metadata)
        proc_kwargs: dict[str, Any] = {
            "text": texts,
            "padding": True,
            "return_tensors": "pt",
            "add_special_tokens": False,
        }
        if images:
            proc_kwargs["images"] = images
        if videos:
            proc_kwargs["videos"] = videos
            proc_kwargs["do_resize"] = False
            proc_kwargs.update(video_processor_kwargs)
            if video_metadatas:
                proc_kwargs["video_metadata"] = video_metadatas
            else:
                collapsed_fps = collapse_fps(fps_values)
                if collapsed_fps is not None:
                    proc_kwargs["fps"] = collapsed_fps
        processor = self.data_collator._base.processor
        processor_tokenizer = getattr(processor, "tokenizer", None)
        previous_padding_side = getattr(processor_tokenizer, "padding_side", None)
        if processor_tokenizer is not None:
            processor_tokenizer.padding_side = "left"
        try:
            batch = processor(**proc_kwargs)
        finally:
            if processor_tokenizer is not None and previous_padding_side is not None:
                processor_tokenizer.padding_side = previous_padding_side
        attention_mask = batch.get("attention_mask")
        if (
            attention_mask is not None
            and attention_mask.shape[0] > 1
            and not bool((attention_mask[:, -1] != 0).all())
        ):
            raise RuntimeError(
                "Module C batched generation produced right-padded inputs. "
                "Decoder-only generation requires left padding."
            )
        return batch

    def _generate_jsons(
        self,
        model: Any,
        examples: list[dict[str, Any]],
    ) -> list[str]:
        inputs = self._generation_inputs(examples)
        device = next(model.parameters()).device
        inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
        input_length = inputs["input_ids"].shape[-1]
        generate_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": False,
            "use_cache": self.use_cache,
        }
        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_token_id is not None:
            generate_kwargs["pad_token_id"] = pad_token_id
        try:
            output_ids = model.generate(**inputs, **generate_kwargs)
            generated_ids = output_ids[:, input_length:]
            return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
        finally:
            del inputs
            if "output_ids" in locals():
                del output_ids
            if "generated_ids" in locals():
                del generated_ids

    def _generate_json(self, model: Any, example: dict[str, Any]) -> str:
        return self._generate_jsons(model, [example])[0]

    def _score_class_nlls(
        self,
        model: Any,
        example: dict[str, Any],
        class_names: tuple[str, ...] | list[str],
    ) -> dict[str, float]:
        """Length-normalized response NLL for each canonical class answer."""

        import torch
        import torch.nn.functional as functional

        device = next(model.parameters()).device
        output_embeddings = unwrap_output_embeddings(model)
        scores: dict[str, float] = {}
        for class_name in class_names:
            scored_example = copy.deepcopy(example)
            target = module_c_candidate_target(str(class_name))
            scored_example["target"] = target
            for message in scored_example["messages"]:
                if message.get("role") == "assistant":
                    message["content"] = [{"type": "text", "text": target}]
                    break
            batch = self.data_collator([scored_example])
            inputs = {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in batch.items()
            }
            labels = inputs.pop("labels", None)
            inputs.pop("response_loss_weight", None)
            inputs.pop("eval_row_index", None)
            inputs.pop("eval_answer_positions", None)
            inputs.pop("eval_candidate_token_ids", None)
            if labels is None:
                raise RuntimeError(
                    "Module C class scoring requires response-only labels."
                )
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
                    shifted_labels = labels[:, 1:]
                    selected = shifted_labels != -100
                    if not bool(selected.any()):
                        raise RuntimeError(
                            f"No response tokens found while scoring {class_name!r}."
                        )
                    response_hidden = hidden[selected]
                    response_targets = shifted_labels[selected]
                    full_logits = output_embeddings(response_hidden)
                    token_nlls = functional.cross_entropy(
                        full_logits.float(),
                        response_targets.to(full_logits.device),
                        reduction="none",
                    )
                    scores[str(class_name)] = float(
                        token_nlls.mean().detach().cpu()
                    )
            finally:
                del inputs
                if outputs is not None:
                    del outputs
                release_cuda_memory()
        return scores


class ModuleCUnslothEnvGuardCallback:
    def __init__(self) -> None:
        from transformers import TrainerCallback

        self._callback_base = TrainerCallback()
        self.trainer: Any | None = None
        self._restore_training_state_at_next_step = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._callback_base, name)

    def on_step_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        enable_module_c_training_hidden_states()
        if self._restore_training_state_at_next_step:
            model = kwargs.get("model")
            if model is not None:
                model = unwrap_trainer_model(self.trainer, model)
                restore_module_c_training_state(model)
                release_cuda_memory()
                if is_distributed_rank_zero():
                    print(
                        "[memory] Restored Module C training state after generation "
                        f"eval; gradient_checkpointing="
                        f"{getattr(model, 'is_gradient_checkpointing', 'unknown')}.",
                        flush=True,
                    )
            self._restore_training_state_at_next_step = False
        return control

    def on_evaluate(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        disable_unsloth_extra_outputs()
        self._restore_training_state_at_next_step = True
        return control

    def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        disable_unsloth_extra_outputs()
        return control


def restore_module_c_training_state(model: Any) -> None:
    """Restore memory-critical training flags after inline generation."""
    from unsloth import FastVisionModel

    enable_module_c_training_hidden_states()
    FastVisionModel.for_training(
        model,
        use_gradient_checkpointing="unsloth",
    )
    if getattr(model, "is_gradient_checkpointing", True) is False:
        enable_checkpointing = getattr(model, "gradient_checkpointing_enable", None)
        if callable(enable_checkpointing):
            enable_checkpointing()
    model.train()
    for config in (
        getattr(model, "config", None),
        getattr(getattr(model, "config", None), "text_config", None),
    ):
        if config is not None and hasattr(config, "use_cache"):
            config.use_cache = False


def build_trainer(
    *,
    model: Any,
    tokenizer: Any,
    train_dataset: list[dict[str, Any]],
    eval_dataset: list[dict[str, Any]],
    args: argparse.Namespace,
) -> Any:
    from datasets import Dataset
    from transformers import TrainerCallback
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastVisionModel

    class ModuleCSFTTrainer(SFTTrainer):
        def compute_loss(
            self,
            model: Any,
            inputs: dict[str, Any],
            return_outputs: bool = False,
            num_items_in_batch: Any | None = None,
        ) -> Any:
            del num_items_in_batch
            loss = module_c_response_only_loss(
                model,
                inputs,
                trainer=self,
            )
            # Evaluation only needs the scalar loss. Returning the hidden-state
            # output would make Trainer retain a native-video tensor.
            return (loss, {"loss": loss.detach()}) if return_outputs else loss

    FastVisionModel.for_training(
        model,
        use_gradient_checkpointing="unsloth",
    )
    def with_length(
        example: dict[str, Any],
        *,
        max_frames: int,
    ) -> dict[str, Any]:
        if not args.group_by_length:
            return example
        return {
            **example,
            "length": estimated_window_frames(
                example["window_start"],
                example["window_end"],
                fps=args.fps,
                min_frames=args.min_frames,
                max_frames=max_frames,
            ),
        }

    hf_train_dataset = Dataset.from_list([
        with_length(example, max_frames=args.max_frames)
        for example in train_dataset
    ])
    hf_eval_dataset = Dataset.from_list([
        with_length(example, max_frames=args.eval_max_frames)
        for example in eval_dataset
    ])
    callbacks: list[TrainerCallback] = []
    trainer_args = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        eval_strategy="no",
        save_strategy="no",
        load_best_model_at_end=False,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.greater_is_better,
        optim=args.optim,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        report_to=args.report_to,
        dataset_num_proc=args.dataset_num_proc,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_prefetch_factor=args.dataloader_prefetch_factor,
        dataloader_pin_memory=True,
        max_seq_length=args.max_seq_length,
        torch_empty_cache_steps=args.torch_empty_cache_steps or None,
        ddp_find_unused_parameters=args.ddp_find_unused_parameters,
        ddp_broadcast_buffers=False,
        ddp_timeout=args.ddp_timeout,
        fp16=False,
        bf16=False,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        train_sampling_strategy=(
            "group_by_length" if args.group_by_length else "random"
        ),
        length_column_name="length",
    )
    if args.train_mode == "epochs":
        trainer_args["num_train_epochs"] = args.num_train_epochs
    else:
        trainer_args["max_steps"] = args.max_steps

    data_collator = Qwen3VLMetadataVisionDataCollator(
        model,
        tokenizer,
        max_seq_length=args.max_seq_length,
        resize=args.vision_resize,
        train_on_responses_only=True,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
        completion_only_loss=True,
    )
    mistake_callback = ModuleCMistakeMetricsCallback(
        eval_examples=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        max_samples=args.eval_generation_max_samples,
        max_new_tokens=args.eval_generation_max_new_tokens,
        enable_thinking=args.generation_enable_thinking,
        batch_size=args.eval_generation_batch_size,
        use_cache=args.eval_generation_use_cache,
        allow_legacy_binary=args.legacy_binary_parser,
        fine_classes=taxonomy_for_dataset(args.dataset)[0],
        aggregate_mapping=taxonomy_for_dataset(args.dataset)[1],
        report_runtime=bool(args.zero_shot_eval),
        compute_bertscore=bool(args.zero_shot_eval),
        bertscore_model_type=args.bertscore_model_type,
        bertscore_batch_size=args.bertscore_batch_size,
        bertscore_device=args.bertscore_device,
    )
    callbacks.append(mistake_callback)
    checkpoint_callback = EpochCheckpointCallback(
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
            eval_generation_max_samples=0,
            generation_eval_mode="inline",
        ),
    )
    if not getattr(args, "zero_shot_eval", False):
        callbacks.append(checkpoint_callback)
    env_guard_callback = ModuleCUnslothEnvGuardCallback()
    callbacks.append(env_guard_callback)
    trainer = ModuleCSFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=hf_train_dataset,
        eval_dataset=hf_eval_dataset,
        data_collator=data_collator,
        args=SFTConfig(**trainer_args),
        callbacks=callbacks,
    )
    mistake_callback.trainer = trainer
    env_guard_callback.trainer = trainer
    if not getattr(args, "zero_shot_eval", False):
        checkpoint_callback.trainer = trainer
        attach_cluster_checkpointing(trainer, args)
    return trainer


def run_generation_eval_only(
    *,
    args: argparse.Namespace,
    val_conversations: list[dict[str, Any]],
) -> None:
    import torch
    from unsloth import FastVisionModel

    if args.eval_generation_max_samples == 0:
        metrics: dict[str, float] = {}
        predictions: list[dict[str, Any]] = []
    else:
        model, tokenizer = load_generation_eval_model(args)
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
        generator = ModuleCMistakeMetricsCallback(
            eval_examples=val_conversations,
            data_collator=collator,
            tokenizer=tokenizer,
            max_samples=args.eval_generation_max_samples,
            max_new_tokens=args.eval_generation_max_new_tokens,
            enable_thinking=args.generation_enable_thinking,
            batch_size=args.eval_generation_batch_size,
            use_cache=args.eval_generation_use_cache,
            allow_legacy_binary=args.legacy_binary_parser,
            fine_classes=taxonomy_for_dataset(args.dataset)[0],
            aggregate_mapping=taxonomy_for_dataset(args.dataset)[1],
        )
        predictions = []
        pred_labels: list[str | None] = []
        target_labels: list[str] = []
        predicted_reasonings: list[str | None] = []
        target_reasonings: list[str] = []
        response_seconds: list[float] = []
        amortized_seconds: list[float] = []
        FastVisionModel.for_inference(model)
        try:
            with torch.inference_mode():
                for start in range(0, len(generator.eval_examples), generator.batch_size):
                    batch_examples = generator.eval_examples[
                        start : start + generator.batch_size
                    ]
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    inference_started = time.perf_counter()
                    generated_texts = generator._generate_jsons(
                        model,
                        batch_examples,
                    )
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    batch_seconds = time.perf_counter() - inference_started
                    for example, generated_text in zip(
                        batch_examples,
                        generated_texts,
                    ):
                        response_seconds.append(batch_seconds)
                        amortized_seconds.append(
                            batch_seconds / len(batch_examples)
                        )
                        parsed = parse_module_c_prediction(
                            generated_text,
                            allow_legacy_binary=generator.allow_legacy_binary,
                            allowed_classes=generator.fine_classes,
                            aggregate_mapping=generator.aggregate_mapping,
                        )
                        pred_class = (
                            None if parsed is None else str(parsed["classification"])
                        )
                        pred_mistake = (
                            None if parsed is None else bool(parsed["mistake"])
                        )
                        pred_reasoning = (
                            None
                            if parsed is None or parsed["reasoning"] is None
                            else str(parsed["reasoning"])
                        )
                        pred_labels.append(pred_class)
                        predicted_reasonings.append(pred_reasoning)
                        target_labels.append(str(example.get("label")))
                        target_reasonings.append(str(example.get("reasoning") or ""))
                        predictions.append(
                            {
                                "video_id": example.get("video_id"),
                                "task_id": example.get("task_id"),
                                "step_index": example.get("step_index"),
                                "window_start": example.get("window_start"),
                                "window_end": example.get("window_end"),
                                "gt_mistake": bool(example.get("mistake")),
                                "gt_classification": example.get("label"),
                                "gt_reasoning": example.get("reasoning"),
                                "pred_mistake": pred_mistake,
                                "pred_classification": pred_class,
                                "pred_reasoning": pred_reasoning,
                                "raw": generated_text,
                            }
                        )
        finally:
            del model
            disable_unsloth_extra_outputs()
            release_cuda_memory()
        metrics = module_c_multiclass_metrics(
            target_labels,
            pred_labels,
            fine_classes=generator.fine_classes,
            aggregate_mapping=generator.aggregate_mapping,
        )
        reasoning_metrics, bertscore_hash = module_c_reasoning_metrics(
            target_reasonings,
            predicted_reasonings,
            total_examples=len(target_labels),
            compute_bertscore=True,
            bertscore_model_type=args.bertscore_model_type,
            bertscore_batch_size=args.bertscore_batch_size,
            bertscore_device=args.bertscore_device,
        )
        metrics.update(reasoning_metrics)
        metrics.update(
            inference_runtime_metrics(
                response_seconds,
                amortized_seconds=amortized_seconds,
            )
        )
    if args.eval_generation_max_samples == 0:
        bertscore_hash = None

    payload = {
        "checkpoint": args.eval_checkpoint,
        "zero_shot_eval": not bool(args.eval_checkpoint),
        "model_name": args.model_name,
        "num_validation_examples": len(val_conversations),
        "num_generation_examples": (
            len(val_conversations)
            if args.eval_generation_max_samples < 0
            else min(args.eval_generation_max_samples, len(val_conversations))
        ),
        "eval_max_frames": args.eval_max_frames,
        "vision_resize": args.vision_resize,
        "max_seq_length": args.max_seq_length,
        "reasoning_metric_config": {
            "lexical_overlap": "lowercase_token_set_jaccard",
            "bertscore": {
                "model_type": args.bertscore_model_type,
                "batch_size": args.bertscore_batch_size,
                "device": args.bertscore_device,
                "hash": bertscore_hash,
                "idf": False,
                "rescale_with_baseline": False,
            },
        },
        "metrics": metrics,
        "predictions": predictions,
    }
    if args.generation_eval_output_json:
        output_path = Path(args.generation_eval_output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(json_safe(payload), file, indent=2, sort_keys=True, ensure_ascii=False)
            file.write("\n")
        print(f"Saved Module C generation eval to: {output_path}")
    print(json.dumps(json_safe({key: value for key, value in payload.items() if key != "predictions"}), indent=2, sort_keys=True))


def run_zero_shot_eval(
    *,
    args: argparse.Namespace,
    val_conversations: list[dict[str, Any]],
) -> None:
    print(
        "[eval] ZERO-SHOT START base_model="
        f"{args.model_name} examples={len(val_conversations)} "
        "generation_classification=True",
        flush=True,
    )
    args.resume_from_checkpoint = None
    model, tokenizer = load_unsloth_model(args)
    _, world_size = distributed_rank_and_world()
    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dummy_train_rows_for_eval_only(val_conversations, world_size),
        eval_dataset=val_conversations,
        args=args,
    )
    metrics = trainer.evaluate()
    bertscore_hash = next(
        (
            callback.bertscore_hash
            for callback in trainer.callback_handler.callbacks
            if isinstance(callback, ModuleCMistakeMetricsCallback)
        ),
        None,
    )
    payload = {
        "checkpoint": None,
        "zero_shot_eval": True,
        "model_name": args.model_name,
        "num_validation_examples": len(val_conversations),
        "eval_max_frames": args.eval_max_frames,
        "vision_resize": args.vision_resize,
        "max_seq_length": args.max_seq_length,
        "reasoning_metric_config": {
            "lexical_overlap": "lowercase_token_set_jaccard",
            "bertscore": {
                "model_type": args.bertscore_model_type,
                "batch_size": args.bertscore_batch_size,
                "device": args.bertscore_device,
                "hash": bertscore_hash,
                "idf": False,
                "rescale_with_baseline": False,
            },
        },
        "metrics": scalar_metrics_for_logging(metrics) if isinstance(metrics, dict) else {},
    }
    output_json = args.generation_eval_output_json
    if not output_json:
        output_json = str(Path(args.output_dir) / "zero_shot_eval.json")
    if trainer.is_world_process_zero():
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(json_safe(payload), file, indent=2, sort_keys=True, ensure_ascii=False)
            file.write("\n")
        print(f"Wrote zero-shot eval metrics to: {output_path}", flush=True)
        print(json.dumps(json_safe(payload), indent=2, sort_keys=True), flush=True)


def print_dry_run(
    dataset: ModuleTrainingDataset,
    train_conversations: list[dict[str, Any]],
    val_conversations: list[dict[str, Any]],
    split: dict[str, Any],
    split_path: Path,
    limit: int,
) -> None:
    print("Module C full dataset summary:")
    print(json.dumps(summarize_dataset(dataset), indent=2, ensure_ascii=False))
    print(f"Module C split file: {split_path}")
    print("Module C split video summary:")
    print(json.dumps({
        "train_video_ids": split["train_video_ids"],
        "val_video_ids": split["val_video_ids"],
        "tasks": split["tasks"],
    }, indent=2, ensure_ascii=False))
    print("Module C train summary:")
    print(json.dumps(summarize_conversations(train_conversations), indent=2, ensure_ascii=False))
    print("Module C validation summary:")
    print(json.dumps(summarize_conversations(val_conversations), indent=2, ensure_ascii=False))
    for index, conversation in enumerate(train_conversations[: max(0, limit)]):
        print(f"\nExample {index}:")
        print(json.dumps(conversation, indent=2, ensure_ascii=False))


def main() -> None:
    configure_kernel_caches()
    args = parse_args()
    print(
        "Module C Unsloth env: "
        f"UNSLOTH_RETURN_LOGITS={os.environ.get('UNSLOTH_RETURN_LOGITS')}, "
        f"UNSLOTH_RETURN_HIDDEN_STATES={os.environ.get('UNSLOTH_RETURN_HIDDEN_STATES')}"
    )
    dataset = build_module_c_dataset(args)
    split, split_path = load_or_create_video_split(dataset, args)
    train_examples = [
        example
        for example in dataset
        if example.video_id in set(split["train_video_ids"])
    ]
    train_examples = oversample_non_correct_examples(
        train_examples,
        factor=args.module_c_non_correct_oversample_factor,
        seed=args.seed,
    )
    train_examples = apply_module_c_history_dropout(
        train_examples,
        probability=args.module_c_history_dropout_probability,
        max_events=args.module_c_history_max_events,
        seed=args.seed + 2,
    )
    train_conversations = [
        training_example_to_conversation(
            example,
            fps=args.fps,
            min_frames=args.min_frames,
            max_frames=args.max_frames,
        )
        for example in train_examples
    ]
    val_conversations = conversations_for_video_ids(
        dataset,
        set(split["val_video_ids"]),
        fps=args.fps,
        min_frames=args.min_frames,
        max_frames=args.eval_max_frames,
    )
    if not val_conversations:
        raise ValueError("Train/validation split produced an empty validation set.")
    if not args.zero_shot_eval and not args.generation_eval_only and not train_conversations:
        raise ValueError("Train/validation split produced an empty conversation split.")
    if args.generation_eval_only:
        print("Module C full dataset summary:")
        print(json.dumps(summarize_dataset(dataset), indent=2, ensure_ascii=False))
        print(f"Module C split file: {split_path}")
        print("Module C validation summary:")
        print(json.dumps(summarize_conversations(val_conversations), indent=2, ensure_ascii=False))
        run_generation_eval_only(args=args, val_conversations=val_conversations)
        return
    if args.dry_run:
        print_dry_run(dataset, train_conversations, val_conversations, split, split_path, args.print_examples)
        return
    print("Module C full dataset summary:")
    print(json.dumps(summarize_dataset(dataset), indent=2, ensure_ascii=False))
    print(f"Module C split file: {split_path}")
    print("Module C train summary:")
    print(json.dumps(summarize_conversations(train_conversations), indent=2, ensure_ascii=False))
    print("Module C validation summary:")
    print(json.dumps(summarize_conversations(val_conversations), indent=2, ensure_ascii=False))

    if is_distributed_rank_zero():
        save_run_config(
            args=args,
            dataset=dataset,
            train_conversations=train_conversations,
            val_conversations=val_conversations,
            split=split,
            split_path=split_path,
        )
    if args.zero_shot_eval:
        run_zero_shot_eval(args=args, val_conversations=val_conversations)
        return
    apply_resume_checkpoint(args)
    model, tokenizer = load_unsloth_model(args)
    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_conversations,
        eval_dataset=val_conversations,
        args=args,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    if trainer.is_world_process_zero():
        save_outputs(model, tokenizer, args.output_dir)


if __name__ == "__main__":
    main()
