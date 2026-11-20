from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

os.environ["UNSLOTH_RETURN_LOGITS"] = "0"
os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "0"

from ..config import ModuleBConfig, module_a_config_from_args
from ..models.vision import vision_resize_arg
from ..datasets import (
    EgoPERProvider,
    ModuleTrainingDataset,
    build_module_b_training_dataset,
)
from ..metrics.temporal import (
    TemporalPrediction,
    parse_temporal_prediction as parse_bounded_temporal_prediction,
)
from ..paths import egoper_root
from ..prompts.module_b import MODULE_B_SYSTEM_PROMPT
from ..schemas import TrainingExample

from .kernel_cache import configure_kernel_caches
from .preemption import add_preemption_args, apply_resume_checkpoint, attach_cluster_checkpointing
from .distributed_eval import (
    distributed_rank_and_world,
    gather_round_robin_shards,
    shard_examples_for_rank,
    unwrap_trainer_model,
)
from .module_a import (
    EpochCheckpointCallback,
    Qwen3VLMetadataVisionDataCollator,
    apply_video_split_subset,
    collapse_fps,
    default_split_file as shared_default_split_file,
    estimated_window_frames,
    grouped_video_ids,
    inference_runtime_metrics,
    is_bf16_available,
    dummy_train_rows_for_eval_only,
    is_distributed_rank_zero,
    json_safe,
    load_unsloth_model,
    make_video_split,
    scalar_metrics_for_logging,
    release_cuda_memory,
    save_outputs,
    split_file_lock,
    validate_lora_target_configuration,
    validate_model_precision,
    validate_video_split,
)

def disable_unsloth_extra_outputs() -> None:
    was_logits = os.environ.get("UNSLOTH_RETURN_LOGITS")
    was_hidden = os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES")
    if was_logits == "1" or was_hidden == "1":
        print(
            "Module B disabled leaked Unsloth extra outputs: "
            f"UNSLOTH_RETURN_LOGITS={was_logits}, "
            f"UNSLOTH_RETURN_HIDDEN_STATES={was_hidden}"
        )
    os.environ["UNSLOTH_RETURN_LOGITS"] = "0"
    os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Qwen VL for Module B temporal grounding.")
    parser.add_argument(
        "--dataset",
        choices=("egoper",),
        default="egoper",
    )
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
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
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
    parser.add_argument("--bias", default="none")
    parser.add_argument("--use-rslora", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--loftq-config", default=None)
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
    parser.add_argument(
        "--metric-for-best-model",
        default="eval_temporal/mixed/raw/macro_score",
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
    parser.add_argument(
        "--stride-seconds",
        type=float,
        default=5.0,
        help="Module A polling stride used to reconstruct the clips passed to Module B.",
    )
    parser.add_argument("--completion-margin", type=float, default=0.25)
    parser.add_argument("--negative-to-positive-ratio", type=int, default=2)
    parser.add_argument("--keep-last-wait-windows", type=int, default=2)
    parser.add_argument("--module-b-pre-pad-ratio-min", type=float, default=0.25)
    parser.add_argument("--module-b-pre-pad-ratio-max", type=float, default=0.75)
    parser.add_argument("--module-b-post-pad-ratio-min", type=float, default=0.25)
    parser.add_argument("--module-b-post-pad-ratio-max", type=float, default=0.75)
    parser.add_argument(
        "--module-b-late-trigger-probability",
        type=float,
        default=0.10,
        help="Fraction of training clips extended by one Module A stride to simulate a late trigger.",
    )
    parser.add_argument("--include-incomplete-negatives", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--module-b-negative-ratio", type=float, default=0.10)
    parser.add_argument("--reset-jitter-seconds", type=float, default=0.5)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--min-frames", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--eval-max-frames", type=int, default=None)
    parser.add_argument("--vision-resize", type=vision_resize_arg, default=512)
    parser.add_argument("--max-seq-length", type=int, default=6144)
    parser.add_argument("--eval-generation-max-samples", type=int, default=-1)
    parser.add_argument("--eval-generation-max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--eval-generation-batch-size",
        type=int,
        default=1,
        help="Examples generated together per rank during temporal IoU evaluation.",
    )
    parser.add_argument(
        "--eval-generation-use-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use the model KV/state cache during autoregressive evaluation. "
            "Disabling it recomputes the full video prompt for every output token."
        ),
    )
    parser.add_argument(
        "--generation-enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Allow the model chat template to request reasoning before generation. "
            "Disabled by default because Module B requires short JSON-only output; "
            "thinking can consume the token budget before a temporal span is emitted."
        ),
    )
    parser.add_argument(
        "--zero-shot-eval",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Run the same generation IoU validation as training eval on the base "
            "model (identity LoRA, no optimizer steps), then exit."
        ),
    )
    parser.add_argument(
        "--generation-eval-output-json",
        default=None,
        help="Optional path to write zero-shot / generation-eval metrics JSON.",
    )
    parser.add_argument("--output-dir", default="outputs/module_b_qwen35_lora_grounding")
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
        default=True,
    )
    parser.add_argument("--ddp-timeout", type=int, default=7200)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-examples", type=int, default=2)
    parser.add_argument("--resume-from-checkpoint", default=None)
    add_preemption_args(parser)
    parser.add_argument("--wandb-log-best-checkpoints", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wandb-artifact-prefix", default="module-b")
    args = parser.parse_args()
    args.report_to = [item.strip() for item in args.report_to.split(",") if item.strip()]
    if args.max_videos <= 0:
        parser.error("--max-videos must be positive.")
    if args.eval_max_frames is None:
        args.eval_max_frames = args.max_frames
    if args.max_frames < args.min_frames or args.eval_max_frames < args.min_frames:
        parser.error("--max-frames and --eval-max-frames must be >= --min-frames.")
    if args.keep_last_checkpoints < 0 or args.keep_best_checkpoints < 0:
        parser.error("checkpoint retention counts cannot be negative.")
    if not 0.0 <= args.module_b_late_trigger_probability <= 1.0:
        parser.error("--module-b-late-trigger-probability must be between 0 and 1.")
    if args.eval_generation_max_samples < -1:
        parser.error("--eval-generation-max-samples must be -1, 0, or positive.")
    if args.eval_generation_batch_size < 1:
        parser.error("--eval-generation-batch-size must be positive.")
    if args.max_grad_norm <= 0:
        parser.error("--max-grad-norm must be positive.")
    if args.zero_shot_eval and args.dry_run:
        parser.error("--zero-shot-eval cannot be combined with --dry-run.")
    if args.zero_shot_eval and args.eval_generation_max_samples == 0:
        parser.error("--zero-shot-eval requires --eval-generation-max-samples != 0.")
    if args.zero_shot_eval and "/mixed/" in str(args.metric_for_best_model):
        parser.error(
            "--zero-shot-eval logs separated cohorts only; choose a metric such as "
            "eval_temporal/on_time/raw/mean_iou instead of a mixed metric."
        )
    validate_generation_metric_is_reachable(args, parser.error)
    validate_model_precision(args, parser.error)
    validate_lora_target_configuration(args, parser.error)
    return args


# IoU is parsed out of generated timestamps, so eval_temporal/* is the complete
# set of eval keys Module B produces outside of eval_loss.
MODULE_B_GENERATION_METRIC_PREFIX = "eval_temporal/"


def validate_generation_metric_is_reachable(
    args: argparse.Namespace,
    fail: Any,
) -> None:
    """Reject a --metric-for-best-model the run can never log."""
    metric = str(args.metric_for_best_model)
    if metric.endswith("loss"):
        return
    if not metric.startswith(MODULE_B_GENERATION_METRIC_PREFIX):
        fail(
            f"--metric-for-best-model {metric!r} is never logged by Module B. "
            f"Use eval_loss or a key starting with {MODULE_B_GENERATION_METRIC_PREFIX!r}."
        )
    if args.eval_generation_max_samples == 0:
        fail(
            f"--metric-for-best-model {metric!r} comes from the generation eval, "
            "but --eval-generation-max-samples is 0, which disables it. The run "
            "would train to completion and never save a best checkpoint. Pass -1 "
            "to score every validation example, or select eval_loss."
        )


def build_module_b_dataset(args: argparse.Namespace) -> ModuleTrainingDataset:
    module_a_config = module_a_config_from_args(
        args,
        label_mode="step_id",
        reset_jitter_seconds=0.0,
    )
    module_b_config = ModuleBConfig(
        late_trigger_probability=args.module_b_late_trigger_probability,
        pre_pad_ratio_min=args.module_b_pre_pad_ratio_min,
        pre_pad_ratio_max=args.module_b_pre_pad_ratio_max,
        post_pad_ratio_min=args.module_b_post_pad_ratio_min,
        post_pad_ratio_max=args.module_b_post_pad_ratio_max,
        include_incomplete_negatives=args.include_incomplete_negatives,
        negative_ratio=args.module_b_negative_ratio,
        seed=args.seed,
    )
    provider = EgoPERProvider(
        args.egoper_root,
        video_ids=set(args.video_ids) if args.video_ids else None,
        task_ids=set(args.task_ids) if args.task_ids else None,
        require_existing_videos=True,
    )
    return build_module_b_training_dataset(
        provider,
        config=module_b_config,
        module_a_config=module_a_config,
        max_videos=args.max_videos,
    )


def default_split_file(args: argparse.Namespace) -> Path:
    return shared_default_split_file(args)


def load_or_create_video_split(dataset: ModuleTrainingDataset, args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    split_path = Path(args.split_file) if args.split_file else default_split_file(args)
    with split_file_lock(split_path):
        if split_path.exists() and not args.regenerate_split:
            with open(split_path, "r", encoding="utf-8") as file:
                split = json.load(file)
            validate_video_split(split, dataset)
            return apply_video_split_subset(split, dataset, args), split_path
        split = make_video_split(dataset, args)
        with open(split_path, "w", encoding="utf-8") as file:
            json.dump(split, file, indent=2, sort_keys=True)
            file.write("\n")
    return apply_video_split_subset(split, dataset, args), split_path


def to_unsloth_conversations(
    dataset: ModuleTrainingDataset,
    *,
    fps: float,
    min_frames: int,
    max_frames: int,
) -> list[dict[str, Any]]:
    return [
        training_example_to_conversation(example, fps=fps, min_frames=min_frames, max_frames=max_frames)
        for example in dataset
    ]


def training_example_to_conversation(
    example: TrainingExample,
    *,
    fps: float,
    min_frames: int,
    max_frames: int,
) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": MODULE_B_SYSTEM_PROMPT}]},
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
        "metadata": example.metadata,
    }


def split_conversations(
    conversations: list[dict[str, Any]],
    split: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_ids = set(split["train_video_ids"])
    val_ids = set(split["val_video_ids"])
    train = [item for item in conversations if item["video_id"] in train_ids]
    val = [item for item in conversations if item["video_id"] in val_ids]
    if not train or not val:
        raise ValueError("Train/validation split produced an empty conversation split.")
    return train, val


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
    conversations: list[dict[str, Any]],
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
        "num_conversations": len(conversations),
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


def parse_temporal_prediction(
    text: str,
    *,
    clip_duration: float | None = None,
) -> tuple[float, float] | None:
    return parse_bounded_temporal_prediction(
        text,
        clip_duration=clip_duration,
    ).span


def normalize_span(start: Any, end: Any) -> tuple[float, float] | None:
    try:
        parsed_start = float(start)
        parsed_end = float(end)
    except (TypeError, ValueError):
        return None
    if parsed_start < 0 or parsed_end < parsed_start:
        return None
    return parsed_start, parsed_end


def temporal_iou(pred: tuple[float, float], target: tuple[float, float]) -> float:
    start = max(pred[0], target[0])
    end = min(pred[1], target[1])
    intersection = max(0.0, end - start)
    union = max(pred[1], target[1]) - min(pred[0], target[0])
    return intersection / union if union > 0 else 0.0


def temporal_metrics(
    targets: list[tuple[float, float]],
    predictions: list[TemporalPrediction | tuple[float, float] | None],
    labels: list[str] | None = None,
    *,
    prefix: str = "eval_temporal",
    view: str = "clamped",
) -> dict[str, float]:
    labels = labels or ["LOCALIZE"] * len(targets)
    spans = [
        (
            prediction.span if view == "clamped" else prediction.raw_span
        )
        if isinstance(prediction, TemporalPrediction)
        else prediction
        for prediction in predictions
    ]
    positives = [
        (target, pred)
        for target, pred, label in zip(targets, spans, labels)
        if label != "NO_ACTION"
    ]
    negatives = [
        pred
        for pred, label in zip(spans, labels)
        if label == "NO_ACTION"
    ]
    valid = [(target, pred) for target, pred in positives if pred is not None]
    invalid_count = len(positives) - len(valid)
    ious = [temporal_iou(pred, target) for target, pred in valid]
    clamped_count = sum(
        isinstance(prediction, TemporalPrediction) and prediction.was_clamped
        for prediction in predictions
    )
    metrics: dict[str, float] = {
        f"{prefix}/num_samples": float(len(targets)),
        f"{prefix}/num_positive_samples": float(len(positives)),
        f"{prefix}/num_no_action_samples": float(len(negatives)),
        f"{prefix}/invalid_rate": invalid_count / len(positives) if positives else 0.0,
        f"{prefix}/clamped_rate": clamped_count / len(targets) if targets else 0.0,
        f"{prefix}/mean_iou": sum(ious) / len(positives) if positives else 0.0,
    }
    if negatives:
        no_action_correct = sum(
            (
                prediction.rejection_reason == "no_action"
                if isinstance(prediction, TemporalPrediction)
                else prediction is None
            )
            for prediction, label in zip(predictions, labels)
            if label == "NO_ACTION"
        )
        metrics[f"{prefix}/no_action_accuracy"] = no_action_correct / len(negatives)
        metrics[f"{prefix}/no_action_false_positive_rate"] = 1.0 - metrics[f"{prefix}/no_action_accuracy"]
        true_not_completed = [label == "NO_ACTION" for label in labels]
        pred_not_completed = []
        for prediction, span in zip(predictions, spans):
            if isinstance(prediction, TemporalPrediction):
                pred_not_completed.append(prediction.rejection_reason == "no_action")
            else:
                pred_not_completed.append(span is None)
        not_completed_tp = sum(
            truth and predicted
            for truth, predicted in zip(true_not_completed, pred_not_completed)
        )
        not_completed_fp = sum(
            (not truth) and predicted
            for truth, predicted in zip(true_not_completed, pred_not_completed)
        )
        not_completed_fn = sum(
            truth and (not predicted)
            for truth, predicted in zip(true_not_completed, pred_not_completed)
        )
        not_completed_precision = (
            not_completed_tp / (not_completed_tp + not_completed_fp)
            if not_completed_tp + not_completed_fp
            else 0.0
        )
        not_completed_recall = (
            not_completed_tp / (not_completed_tp + not_completed_fn)
            if not_completed_tp + not_completed_fn
            else 0.0
        )
        metrics[f"{prefix}/not_completed/precision"] = not_completed_precision
        metrics[f"{prefix}/not_completed/recall"] = not_completed_recall
        metrics[f"{prefix}/not_completed/f1"] = (
            2 * not_completed_precision * not_completed_recall
            / (not_completed_precision + not_completed_recall)
            if not_completed_precision + not_completed_recall
            else 0.0
        )
    completed_valid = len(valid)
    metrics[f"{prefix}/completed/num_samples"] = float(len(positives))
    metrics[f"{prefix}/completed/valid_rate"] = (
        completed_valid / len(positives) if positives else 0.0
    )
    for threshold in (0.1, 0.3, 0.5, 0.7, 0.9):
        hits = sum(1 for value in ious if value >= threshold)
        recall = hits / len(positives) if positives else 0.0
        precision = hits / len(valid) if valid else 0.0
        metrics[f"{prefix}/recall_at_{threshold:.1f}"] = recall
        metrics[f"{prefix}/precision_at_{threshold:.1f}"] = precision
        metrics[f"{prefix}/f1_at_{threshold:.1f}"] = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
    if valid:
        start_errors = [abs(pred[0] - target[0]) for target, pred in valid]
        end_errors = [abs(pred[1] - target[1]) for target, pred in valid]
        center_errors = [abs(((pred[0] + pred[1]) / 2) - ((target[0] + target[1]) / 2)) for target, pred in valid]
        duration_errors = [abs((pred[1] - pred[0]) - (target[1] - target[0])) for target, pred in valid]
        metrics.update({
            f"{prefix}/start_mae": sum(start_errors) / len(start_errors),
            f"{prefix}/end_mae": sum(end_errors) / len(end_errors),
            f"{prefix}/center_mae": sum(center_errors) / len(center_errors),
            f"{prefix}/duration_mae": sum(duration_errors) / len(duration_errors),
        })
        for tolerance in (0.5, 1.0, 2.0):
            within = sum(
                1 for start_error, end_error in zip(start_errors, end_errors)
                if start_error <= tolerance and end_error <= tolerance
            )
            metrics[f"{prefix}/boundary_within_{tolerance:.1f}s"] = within / len(positives)
    else:
        metrics.update({
            f"{prefix}/start_mae": 0.0,
            f"{prefix}/end_mae": 0.0,
            f"{prefix}/center_mae": 0.0,
            f"{prefix}/duration_mae": 0.0,
            f"{prefix}/boundary_within_0.5s": 0.0,
            f"{prefix}/boundary_within_1.0s": 0.0,
            f"{prefix}/boundary_within_2.0s": 0.0,
        })
    return metrics


def isolated_temporal_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Score disjoint oracle-trigger, late-trigger, and no-action cohorts."""

    positive_cohorts = {
        "on_time": [
            row
            for row in rows
            if row["label"] != "NO_ACTION" and not row["late_trigger_augmented"]
        ],
        "late": [
            row
            for row in rows
            if row["label"] != "NO_ACTION" and row["late_trigger_augmented"]
        ],
    }
    metrics: dict[str, float] = {}
    for cohort_name, cohort_rows in positive_cohorts.items():
        predictions = [row["prediction"] for row in cohort_rows]
        targets = [row["target"] for row in cohort_rows]
        for view in ("raw", "clamped"):
            metrics.update(
                temporal_metrics(
                    targets,
                    predictions,
                    prefix=f"eval_temporal/{cohort_name}/{view}",
                    view=view,
                )
            )

    no_action_rows = [row for row in rows if row["label"] == "NO_ACTION"]
    count = len(no_action_rows)
    explicit_no_action = sum(
        isinstance(row["prediction"], TemporalPrediction)
        and row["prediction"].rejection_reason == "no_action"
        for row in no_action_rows
    )
    predicted_spans = sum(
        isinstance(row["prediction"], TemporalPrediction)
        and row["prediction"].raw_span is not None
        for row in no_action_rows
    )
    malformed = count - explicit_no_action - predicted_spans
    prefix = "eval_temporal/not_completed"
    metrics.update({
        f"{prefix}/num_samples": float(count),
        f"{prefix}/accuracy": explicit_no_action / count if count else 0.0,
        f"{prefix}/predicted_span_rate": predicted_spans / count if count else 0.0,
        f"{prefix}/malformed_rate": malformed / count if count else 0.0,
    })
    return metrics


class ModuleBTemporalMetricsCallback:
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
        include_mixed_metrics: bool = False,
        report_runtime: bool = False,
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
        self.include_mixed_metrics = include_mixed_metrics
        self.report_runtime = report_runtime

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
                "Module B inline generation eval: "
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
                    generated_texts = self._generate_spans(model, batch_examples)
                    if self.report_runtime and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    batch_response_seconds = (
                        time.perf_counter() - inference_started
                        if self.report_runtime
                        else 0.0
                    )
                    amortized_seconds = (
                        batch_response_seconds / len(batch_examples)
                    )
                    for example, generated_text in zip(batch_examples, generated_texts):
                        parsed = parse_bounded_temporal_prediction(
                            generated_text,
                            clip_duration=example["window_end"] - example["window_start"],
                        )
                        offset = float(example["window_start"])
                        if parsed.start is not None and parsed.end is not None:
                            parsed = TemporalPrediction(
                                start=offset + parsed.start,
                                end=offset + parsed.end,
                                rejection_reason=parsed.rejection_reason,
                                source_format=parsed.source_format,
                                raw_start=(
                                    None if parsed.raw_start is None else offset + parsed.raw_start
                                ),
                                raw_end=(
                                    None if parsed.raw_end is None else offset + parsed.raw_end
                                ),
                                was_clamped=parsed.was_clamped,
                                clamp_reason=parsed.clamp_reason,
                            )
                        local_rows.append({
                            "prediction": parsed,
                            "raw_output": generated_text,
                            "target": (float(example["gt_start"]), float(example["gt_end"])),
                            "label": str(example.get("label", "LOCALIZE")),
                            "video_id": str(example.get("video_id", "")),
                            "task_id": str(example.get("task_id", "")),
                            "step_index": int(example.get("step_index", -1)),
                            "window_start": float(example["window_start"]),
                            "window_end": float(example["window_end"]),
                            "late_trigger_augmented": bool(
                                example.get("metadata", {}).get(
                                    "late_trigger_augmented",
                                    False,
                                )
                            ),
                            "response_seconds": batch_response_seconds,
                            "amortized_inference_seconds": amortized_seconds,
                        })
        finally:
            if was_training:
                restore_module_b_training_state(model)
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
                    prediction = row["prediction"]
                    file.write(json.dumps({
                        "video_id": row["video_id"],
                        "task_id": row["task_id"],
                        "step_index": row["step_index"],
                        "window_start": row["window_start"],
                        "window_end": row["window_end"],
                        "label": row["label"],
                        "late_trigger_augmented": row["late_trigger_augmented"],
                        "target_start": row["target"][0],
                        "target_end": row["target"][1],
                        "generated_text": row["raw_output"],
                        "parsed_start": prediction.start,
                        "parsed_end": prediction.end,
                        "raw_start": prediction.raw_start,
                        "raw_end": prediction.raw_end,
                        "rejection_reason": prediction.rejection_reason,
                        "source_format": prediction.source_format,
                        "was_clamped": prediction.was_clamped,
                    }, ensure_ascii=False) + "\n")
            print(f"Wrote Module B eval predictions to: {prediction_path}", flush=True)
        metrics = isolated_temporal_metrics(rows)
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
        if self.include_mixed_metrics:
            predictions = [row["prediction"] for row in rows]
            targets = [row["target"] for row in rows]
            labels = [str(row["label"]) for row in rows]
            for view in ("raw", "clamped"):
                prefix = f"eval_temporal/mixed/{view}"
                metrics.update(
                    temporal_metrics(
                        targets,
                        predictions,
                        labels,
                        prefix=prefix,
                        view=view,
                    )
                )
                metrics[f"{prefix}/macro_score"] = (
                    metrics[f"eval_temporal/on_time/{view}/mean_iou"]
                    + metrics[f"eval_temporal/late/{view}/mean_iou"]
                    + metrics["eval_temporal/not_completed/accuracy"]
                ) / 3.0

        requested_metric = str(
            getattr(self.trainer.args, "metric_for_best_model", "")
        )
        if self.include_mixed_metrics:
            legacy_metric_aliases = {
                "eval_temporal/mean_iou": (
                    "eval_temporal/mixed/raw/macro_score"
                ),
                "eval_temporal/clamped/mean_iou": (
                    "eval_temporal/mixed/clamped/macro_score"
                ),
            }
        else:
            legacy_metric_aliases = {
                "eval_temporal/mean_iou": "eval_temporal/on_time/raw/mean_iou",
                "eval_temporal/clamped/mean_iou": (
                    "eval_temporal/on_time/clamped/mean_iou"
                ),
            }
        if requested_metric in legacy_metric_aliases:
            metrics[requested_metric] = metrics[
                legacy_metric_aliases[requested_metric]
            ]
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
                dict(message)
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
        previous_padding_side = getattr(
            processor_tokenizer,
            "padding_side",
            None,
        )
        if processor_tokenizer is not None:
            processor_tokenizer.padding_side = "left"
        try:
            batch = processor(**proc_kwargs)
        finally:
            if (
                processor_tokenizer is not None
                and previous_padding_side is not None
            ):
                processor_tokenizer.padding_side = previous_padding_side

        attention_mask = batch.get("attention_mask")
        if (
            attention_mask is not None
            and attention_mask.shape[0] > 1
            and not bool((attention_mask[:, -1] != 0).all())
        ):
            raise RuntimeError(
                "Module B batched generation produced right-padded inputs. "
                "Decoder-only generation requires left padding."
            )
        return batch

    def _generate_spans(
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
            return self.tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=False,
            )
        finally:
            del inputs
            if "output_ids" in locals():
                del output_ids
            if "generated_ids" in locals():
                del generated_ids

    def _generate_span(self, model: Any, example: dict[str, Any]) -> str:
        """Compatibility wrapper for single-example pipeline inference."""

        return self._generate_spans(model, [example])[0]


class ModuleBUnslothEnvGuardCallback:
    def __init__(self) -> None:
        from transformers import TrainerCallback

        self._callback_base = TrainerCallback()
        self.trainer: Any | None = None
        self._restore_training_state_at_next_step = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._callback_base, name)

    def on_step_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        disable_unsloth_extra_outputs()
        if self._restore_training_state_at_next_step:
            model = kwargs.get("model")
            if model is not None:
                model = unwrap_trainer_model(self.trainer, model)
                restore_module_b_training_state(model)
                release_cuda_memory()
                if is_distributed_rank_zero():
                    print(
                        "[memory] Restored Module B training state after generation "
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


def restore_module_b_training_state(model: Any) -> None:
    """Restore memory-critical training flags after inline generation."""
    from unsloth import FastVisionModel

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

    FastVisionModel.for_training(
        model,
        use_gradient_checkpointing="unsloth",
    )
    bf16 = is_bf16_available()
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

    hf_train_dataset = Dataset.from_list(
        [
            with_length(example, max_frames=args.max_frames)
            for example in train_dataset
        ]
    )
    hf_eval_dataset = Dataset.from_list(
        [
            with_length(example, max_frames=args.eval_max_frames)
            for example in eval_dataset
        ]
    )
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
        fp16=not bf16,
        bf16=bf16,
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
    temporal_callback = ModuleBTemporalMetricsCallback(
        eval_examples=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        max_samples=args.eval_generation_max_samples,
        max_new_tokens=args.eval_generation_max_new_tokens,
        enable_thinking=args.generation_enable_thinking,
        batch_size=args.eval_generation_batch_size,
        use_cache=args.eval_generation_use_cache,
        include_mixed_metrics=not args.zero_shot_eval,
        report_runtime=bool(args.zero_shot_eval),
    )
    callbacks.append(temporal_callback)
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
    env_guard_callback = ModuleBUnslothEnvGuardCallback()
    callbacks.append(env_guard_callback)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=hf_train_dataset,
        eval_dataset=hf_eval_dataset,
        data_collator=data_collator,
        args=SFTConfig(**trainer_args),
        callbacks=callbacks,
    )
    temporal_callback.trainer = trainer
    env_guard_callback.trainer = trainer
    if not getattr(args, "zero_shot_eval", False):
        checkpoint_callback.trainer = trainer
        attach_cluster_checkpointing(trainer, args)
    return trainer


def print_dry_run(
    dataset: ModuleTrainingDataset,
    train_conversations: list[dict[str, Any]],
    val_conversations: list[dict[str, Any]],
    split: dict[str, Any],
    split_path: Path,
    limit: int,
) -> None:
    print("Module B full dataset summary:")
    print(json.dumps(summarize_dataset(dataset), indent=2, ensure_ascii=False))
    print(f"Module B split file: {split_path}")
    print("Module B split video summary:")
    print(json.dumps({
        "train_video_ids": split["train_video_ids"],
        "val_video_ids": split["val_video_ids"],
        "tasks": split["tasks"],
    }, indent=2, ensure_ascii=False))
    print("Module B train summary:")
    print(json.dumps(summarize_conversations(train_conversations), indent=2, ensure_ascii=False))
    print("Module B validation summary:")
    print(json.dumps(summarize_conversations(val_conversations), indent=2, ensure_ascii=False))
    for index, conversation in enumerate(train_conversations[: max(0, limit)]):
        print(f"\nExample {index}:")
        print(json.dumps(conversation, indent=2, ensure_ascii=False))


def run_zero_shot_eval(
    *,
    args: argparse.Namespace,
    val_conversations: list[dict[str, Any]],
) -> None:
    print(
        "[eval] ZERO-SHOT START base_model="
        f"{args.model_name} examples={len(val_conversations)} "
        "generation_iou=True",
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
    payload = {
        "checkpoint": None,
        "zero_shot_eval": True,
        "model_name": args.model_name,
        "num_validation_examples": len(val_conversations),
        "eval_max_frames": args.eval_max_frames,
        "vision_resize": args.vision_resize,
        "max_seq_length": args.max_seq_length,
        "metrics": scalar_metrics_for_logging(metrics) if isinstance(metrics, dict) else {},
    }
    output_json = args.generation_eval_output_json
    if not output_json:
        output_json = str(Path(args.output_dir) / "zero_shot_eval.json")
    if trainer.is_world_process_zero():
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(json_safe(payload), file, indent=2, sort_keys=True)
            file.write("\n")
        print(f"Wrote zero-shot eval metrics to: {output_path}", flush=True)
        print(json.dumps(json_safe(payload), indent=2, sort_keys=True), flush=True)


def main() -> None:
    configure_kernel_caches()
    args = parse_args()
    print(
        "Module B Unsloth env: "
        f"UNSLOTH_RETURN_LOGITS={os.environ.get('UNSLOTH_RETURN_LOGITS')}, "
        f"UNSLOTH_RETURN_HIDDEN_STATES={os.environ.get('UNSLOTH_RETURN_HIDDEN_STATES')}"
    )
    dataset = build_module_b_dataset(args)
    split, split_path = load_or_create_video_split(dataset, args)
    train_conversations = conversations_for_video_ids(
        dataset,
        set(split["train_video_ids"]),
        fps=args.fps,
        min_frames=args.min_frames,
        max_frames=args.max_frames,
    )
    val_conversations = conversations_for_video_ids(
        dataset,
        set(split["val_video_ids"]),
        fps=args.fps,
        min_frames=args.min_frames,
        max_frames=args.eval_max_frames,
    )
    conversations = train_conversations + val_conversations
    if not conversations:
        raise SystemExit("Module B dataset produced no conversations.")
    if not val_conversations:
        raise ValueError("Train/validation split produced an empty validation set.")
    if not args.zero_shot_eval and not train_conversations:
        raise ValueError("Train/validation split produced an empty conversation split.")
    if args.dry_run:
        print_dry_run(dataset, train_conversations, val_conversations, split, split_path, args.print_examples)
        return
    if is_distributed_rank_zero():
        save_run_config(
            args=args,
            dataset=dataset,
            conversations=conversations,
            train_conversations=train_conversations,
            val_conversations=val_conversations,
            split=split,
            split_path=split_path,
        )
    print("Module B full dataset summary:")
    print(json.dumps(summarize_dataset(dataset), indent=2, ensure_ascii=False))
    print(f"Module B split file: {split_path}")
    print("Module B train summary:")
    print(json.dumps(summarize_conversations(train_conversations), indent=2, ensure_ascii=False))
    print("Module B validation summary:")
    print(json.dumps(summarize_conversations(val_conversations), indent=2, ensure_ascii=False))

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
