from __future__ import annotations

import argparse
import csv
import json
import math
import os
import signal
import time
from dataclasses import asdict, fields, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]

from ..config import ModuleAConfig, module_a_config_from_args
from ..datasets import (
    EgoPERProvider,
    ModuleTrainingDataset,
    build_module_a_training_dataset,
)
from ..datasets.egoper import EGOPER_AGGREGATE_MAPPING, EGOPER_FINE_CLASSES, EGOPER_MODULE_C_CLASSES
from ..datasets.labels import (
    EXTRA_STEP_INSTRUCTION,
    is_module_a_completion_label,
    module_a_instruction_for_step_id_letter,
    reasoning_for_segment,
)
from ..prompts.module_b import build_module_b_prompt
from ..metrics.classification import (
    module_c_metrics as classification_metrics,
)
from ..metrics.calibration import label_score_diagnostics, scores_to_probabilities
from ..metrics.module_a import (
    binary_detection_metrics,
    multiclass_letter_metrics,
)
from ..paths import egoper_root
from .procedural_graph import (
    DependencyGraphPipeline,
    DetectedOccurrence,
    StructuralAlert,
    structural_metrics,
)
from ..prompts.contracts import NOT_COMPLETED_STATUS, is_not_completed_text
from ..prompts.module_a import parse_completion_set
from ..prompts.module_c import (
    append_module_c_history,
    build_module_c_prompt,
)
from ..metrics.temporal import (
    TemporalPrediction,
    parse_temporal_prediction as parse_bounded_temporal_prediction,
)
from ..schemas import TrainingExample, VideoRecord, WindowSpec
from ..models.unsloth import (
    ModuleADetectionMetricsCallback,
    ModuleBTemporalMetricsCallback,
    ModuleCMistakeMetricsCallback,
    Qwen3VLMetadataVisionDataCollator,
    extract_module_a_label,
    json_safe,
    load_generation_eval_model,
    load_or_create_video_split,
    parse_module_c_prediction,
    release_cuda_memory,
    lexical_overlap,
    module_a_to_conversation,
    module_b_to_conversation,
    module_c_to_conversation,
    temporal_iou,
    temporal_metrics,
)
from ..models.video import video_duration_seconds
from .events import GroundTruthEvent, PipelineEvent, predicted_step_index
from .shards import (
    claim_next_video,
    completed_shard_dirs,
    discover_shard_artifacts,
    mark_shard_done,
    merge_pipeline_artifacts,
    shard_output_dir,
    shard_video_ids,
    try_acquire_merge_lock,
    window_counts_by_video,
)
from .adapters import LoraAdapterBank, check_adapter_configs, load_adapter_weights
from .causal import CausalVideoResult, run_causal_video
from .video_checkpoints import (
    enqueue_video,
    events_checkpoint_path,
    list_module_a_done,
    list_pipeline_done,
    load_causal_ab_events,
    load_event_rows,
    load_ground_truth_rows,
    load_module_a_rows,
    load_run_manifest,
    refresh_remaining_videos,
    write_causal_ab_events,
    write_events_checkpoint,
    write_module_a_checkpoint,
    write_progress,
    write_run_manifest,
)

MODULE_C_CLIPS = ("span", "window")


class RunControl:
    stop_requested = False


def install_stop_handler() -> None:
    def _handle(signum: int, _frame: Any) -> None:
        RunControl.stop_requested = True
        print(f"Signal {signum}: finishing the current video, then exiting.", flush=True)

    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)

IOU_THRESHOLDS = (0.1, 0.3, 0.5, 0.7, 0.9)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run end-to-end online Module A -> B -> C pipeline evaluation.")
    parser.add_argument("--module-a-checkpoint", default=None)
    parser.add_argument("--module-b-checkpoint", default=None)
    parser.add_argument("--module-c-checkpoint", default=None)
    parser.add_argument(
        "--reuse-events-json",
        default=None,
        help=(
            "Reuse saved pipeline events (A/B windows) and only re-run Module C. "
            "Skips Module A/B inference. Useful after fixing Module C frame/resize settings."
        ),
    )
    parser.add_argument("--model-name", default="unsloth/Qwen3.5-2B")
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--load-in-8bit", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--load-in-16bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--dataset",
        choices=("egoper",),
        default="egoper",
    )
    parser.add_argument("--egoper-root", default=str(egoper_root()))
    parser.add_argument("--max-videos", type=int, default=50)
    parser.add_argument("--task-ids", nargs="*", default=None)
    parser.add_argument("--video-ids", nargs="*", default=None)
    parser.add_argument("--split-file", default=None)
    parser.add_argument("--regenerate-split", action="store_true")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--val-videos-per-task", type=int, default=2)
    parser.add_argument("--train-videos-per-task", type=int, default=None)
    parser.add_argument(
        "--train-validation-erroneous-only",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--stride-seconds", type=float, default=5.0)
    parser.add_argument("--completion-margin", type=float, default=0.10)
    parser.add_argument("--negative-to-positive-ratio", type=int, default=2)
    parser.add_argument("--keep-last-wait-windows", type=int, default=2)
    parser.add_argument("--max-negative-windows-per-episode", type=int, default=3)
    parser.add_argument("--reset-jitter-seconds", type=float, default=0.0)
    parser.add_argument(
        "--module-a-label-mode",
        choices=("legacy", "step_id", "completion_set"),
        default="step_id",
        help="Module A label/prompt mode. step_id uses A/B/C+ letters; legacy uses WAIT/COMPLETE.",
    )
    parser.add_argument(
        "--online-reset-from-b",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "In step_id mode, after each Module A trigger run Module B and advance the stream "
            "start to B's predicted end for subsequent Module A windows on that video."
        ),
    )
    parser.add_argument(
        "--causal-stream",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Build Module A clips online from the pipeline's own cursor and "
            "predicted step history instead of the gold-cut windows. Loads A "
            "and B together, then C."
        ),
    )
    parser.add_argument(
        "--causal-max-videos",
        type=int,
        default=-1,
        help="With --causal-stream, stop each worker after this many videos (dry runs).",
    )
    parser.add_argument(
        "--causal-module-c-in-loop",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "With --causal-stream, run Module C right after each placed step on one "
            "shared base model. A Module C NOT_COMPLETED keeps the cursor, like "
            "Module A's and Module B's."
        ),
    )
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--min-frames", type=int, default=2)
    parser.add_argument("--max-frames-a", type=int, default=32)
    parser.add_argument("--max-frames-b", type=int, default=24)
    parser.add_argument("--max-frames-c", type=int, default=16)
    parser.add_argument("--vision-resize-a", type=int, default=384)
    parser.add_argument("--vision-resize-b", type=int, default=512)
    parser.add_argument("--vision-resize-c", type=int, default=384)
    parser.add_argument("--max-seq-length-a", type=int, default=6144)
    parser.add_argument("--max-seq-length-b", type=int, default=8192)
    parser.add_argument("--max-seq-length-c", type=int, default=3072)
    parser.add_argument("--module-a-max-new-tokens", type=int, default=8)
    parser.add_argument("--module-b-max-new-tokens", type=int, default=64)
    parser.add_argument("--module-c-max-new-tokens", type=int, default=128)
    parser.add_argument("--module-c-history-max-events", type=int, default=12)
    parser.add_argument(
        "--module-c-clip",
        choices=MODULE_C_CLIPS,
        default="span",
        help=(
            "Video Module C sees: Module B's span, or the whole clip Module B "
            "searched. The next Module A clip starts at Module B's end either way."
        ),
    )
    parser.add_argument(
        "--module-c-candidate-scoring",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Compute deterministic forced-choice Module C class probabilities "
            "from length-normalized candidate NLLs and save module_c_error_score."
        ),
    )
    parser.add_argument(
        "--module-c-score-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "With --reuse-events-json, preserve saved Module C generations and "
            "only add forced-choice class probabilities."
        ),
    )
    parser.add_argument("--label-score-eval", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--zero-shot-eval",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Load the base --model-name for A, B, and C instead of LoRA checkpoints.",
    )
    parser.add_argument("--fbeta-beta", type=float, default=2.0)
    parser.add_argument("--max-samples", type=int, default=-1, help="Cap validation windows/events for smoke tests.")
    parser.add_argument(
        "--output-dir",
        default=str(Path(os.environ.get("OUTPUT_ROOT", "outputs")) / "online_pipeline_eval"),
    )
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-jsonl", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument(
        "--defer-metrics",
        action="store_true",
        help=(
            "Run inference and save a self-contained artifact without matching or "
            "calculating metrics. Use evaluate/online_pipeline/evaluate.py offline."
        ),
    )
    parser.add_argument("--skip-inference", action="store_true", help="Build GT/split only; useful for debugging setup.")
    parser.add_argument(
        "--video-shard-rank",
        type=int,
        default=0,
        help="This process's video-id shard index. Used with --video-shard-world-size.",
    )
    parser.add_argument(
        "--video-shard-world-size",
        type=int,
        default=1,
        help="Number of independent video-id shards. 1 disables sharding.",
    )
    parser.add_argument(
        "--merge-shards-dir",
        default=None,
        help=(
            "Merge shards/*/pipeline_artifacts.json under this directory into one "
            "pipeline_artifacts.json. Does not load A/B/C models."
        ),
    )
    parser.add_argument(
        "--video-queue-dir",
        default=None,
        help=(
            "Shared work queue of evaluation videos. Workers claim the next video "
            "after finishing the previous one, then exit when the queue is empty."
        ),
    )
    parser.add_argument(
        "--video-checkpoint-dir",
        default=None,
        help="Per-video Module A / event checkpoints so a later submit can resume.",
    )
    parser.add_argument(
        "--queue-session-id",
        default=None,
        help="Job-level id used to refresh the work queue once per allocation.",
    )
    parser.add_argument(
        "--eval-split",
        choices=("val", "test"),
        default="val",
        help="Which split of --split-file to evaluate. Trained EgoPER uses test.",
    )
    parser.add_argument(
        "--merge-if-complete",
        default=None,
        help=(
            "If every eval video under this directory has a checkpoint, write "
            "pipeline_artifacts.json. Otherwise write progress.json and exit."
        ),
    )
    args = parser.parse_args()
    if args.merge_shards_dir or args.merge_if_complete:
        return args
    if args.max_videos <= 0:
        parser.error("--max-videos must be positive.")
    if args.stride_seconds <= 0:
        parser.error("--stride-seconds must be positive.")
    if args.max_samples < -1:
        parser.error("--max-samples must be -1 or non-negative.")
    if args.module_c_history_max_events < 0:
        parser.error("--module-c-history-max-events must be non-negative.")
    if args.module_c_score_only and not args.module_c_candidate_scoring:
        parser.error("--module-c-score-only requires --module-c-candidate-scoring.")
    if args.module_c_score_only and not args.reuse_events_json:
        parser.error("--module-c-score-only requires --reuse-events-json.")
    if args.causal_stream and args.module_a_label_mode != "step_id":
        parser.error("--causal-stream requires --module-a-label-mode step_id.")
    if args.causal_stream and args.reuse_events_json:
        parser.error("--causal-stream cannot be combined with --reuse-events-json.")
    if args.causal_module_c_in_loop and not args.causal_stream:
        parser.error("--causal-module-c-in-loop requires --causal-stream.")
    if args.causal_module_c_in_loop and args.module_c_score_only:
        parser.error("--causal-module-c-in-loop needs Module C generation; drop --module-c-score-only.")
    if sum((args.load_in_4bit, args.load_in_8bit, args.load_in_16bit)) != 1:
        parser.error("Exactly one model loading precision must be enabled.")
    if args.zero_shot_eval and args.skip_inference:
        parser.error("--zero-shot-eval cannot be combined with --skip-inference.")
    if args.video_shard_world_size < 1:
        parser.error("--video-shard-world-size must be >= 1.")
    if args.video_shard_rank < 0 or args.video_shard_rank >= args.video_shard_world_size:
        parser.error("--video-shard-rank must be in [0, --video-shard-world-size).")
    if args.reuse_events_json:
        if args.skip_inference:
            parser.error("--reuse-events-json cannot be combined with --skip-inference.")
        if not args.zero_shot_eval and not args.module_c_checkpoint:
            parser.error("--module-c-checkpoint is required unless --zero-shot-eval is set.")
    elif not args.skip_inference:
        missing_abc = not args.module_a_checkpoint or not args.module_b_checkpoint or not args.module_c_checkpoint
        if missing_abc and not args.zero_shot_eval:
            parser.error(
                "--module-a-checkpoint, --module-b-checkpoint, and --module-c-checkpoint "
                "are required unless --zero-shot-eval or --reuse-events-json is set."
            )
    return args


def make_provider(args: argparse.Namespace) -> Any:
    return EgoPERProvider(
        args.egoper_root,
        video_ids=set(args.video_ids) if args.video_ids else None,
        task_ids=set(args.task_ids) if args.task_ids else None,
        require_existing_videos=True,
    )


def module_a_config(args: argparse.Namespace) -> ModuleAConfig:
    return module_a_config_from_args(args, reset_jitter_seconds=0.0)


def gt_events_for_records(
    records: list[VideoRecord],
    val_video_ids: set[str],
    *,
    label_mode: str = "step_id",
) -> list[GroundTruthEvent]:
    events: list[GroundTruthEvent] = []
    for record in records:
        if record.video_id not in val_video_ids:
            continue
        for segment in record.segments:
            if label_mode == "step_id":
                eligible = segment.duration >= 0.5
            else:
                eligible = segment.instruction_index >= 0 and segment.duration >= 0.5
            events.append(
                GroundTruthEvent(
                    video_id=record.video_id,
                    task_id=record.task_id,
                    step_index=segment.instruction_index,
                    instruction=segment.instruction,
                    gt_start=segment.start,
                    gt_end=segment.end,
                    gt_mistake=segment.is_mistake,
                    gt_reasoning=reasoning_for_segment(segment),
                    eligible=eligible,
                    gt_fine_class=segment.fine_class,
                    gt_aggregate_class=segment.aggregate_class,
                    occurrence_id=segment.occurrence_id,
                    annotation_position=segment.annotation_position,
                    gt_fine_classes=segment.resolved_fine_classes,
                    gt_structural_labels=segment.structural_labels,
                )
            )
    return events


def eval_video_ids_from_split(split: dict[str, Any], eval_split: str) -> set[str]:
    key = "test_video_ids" if eval_split == "test" else "val_video_ids"
    ids = split.get(key)
    if not ids:
        raise ValueError(f"Split file has no {key}.")
    return {str(video_id) for video_id in ids}


def val_windows(dataset: ModuleTrainingDataset, val_video_ids: set[str], max_samples: int) -> list[WindowSpec]:
    windows = [window for window in dataset.windows if window.video_id in val_video_ids]
    windows.sort(key=lambda item: (item.video_id, item.window_end, item.step_index, item.window_start))
    if max_samples >= 0:
        return windows[:max_samples]
    return windows


def window_from_payload(payload: dict[str, Any]) -> WindowSpec:
    window_payload = dict(payload)
    window_payload["video_path"] = Path(window_payload["video_path"])
    for key in (
        "completed_steps",
        "pending_steps",
        "all_instructions",
        "completed_step_ids",
        "fine_classes",
    ):
        if key in window_payload:
            window_payload[key] = tuple(window_payload[key])
    allowed = {field.name for field in fields(WindowSpec)}
    return WindowSpec(**{key: value for key, value in window_payload.items() if key in allowed})


def module_a_prediction_from_row(row: dict[str, Any]) -> tuple[WindowSpec, str, str, float]:
    return (
        window_from_payload(row["window"]),
        str(row["prediction"]),
        str(row.get("raw_output") or ""),
        float(row.get("prediction_seconds") or 0.0),
    )


def serialize_module_a_prediction(
    prediction: tuple[WindowSpec, str, str, float],
) -> dict[str, Any]:
    window, parsed_prediction, raw_output, prediction_seconds = prediction
    return {
        "window": json_safe(asdict(window)),
        "prediction": parsed_prediction,
        "raw_output": raw_output,
        "prediction_seconds": prediction_seconds,
    }


def window_to_example(window: WindowSpec, dataset: ModuleTrainingDataset) -> TrainingExample:
    return dataset._window_to_example(window)


def resolved_instruction_for_prediction(
    *,
    window: WindowSpec,
    prediction: str,
    label_mode: str,
) -> str:
    if label_mode == "completion_set":
        steps = parse_completion_set(prediction)
        if len(steps) == 1 and steps[0].startswith("STEP_"):
            index = int(steps[0].removeprefix("STEP_"))
            instructions = window.all_instructions or ()
            if 0 <= index < len(instructions):
                return instructions[index]
        if prediction.startswith("STEP_"):
            index = int(prediction.removeprefix("STEP_"))
            instructions = window.all_instructions or ()
            if 0 <= index < len(instructions):
                return instructions[index]
        return window.current_step
    if label_mode != "step_id":
        return window.current_step
    mapped = module_a_instruction_for_step_id_letter(prediction, window.all_instructions or ())
    if mapped is None:
        return EXTRA_STEP_INSTRUCTION if prediction.upper().startswith("B") else window.current_step
    return mapped


def is_extra_module_a_prediction(prediction: str, *, label_mode: str) -> bool:
    cleaned = prediction.strip().upper()
    if label_mode == "step_id":
        return cleaned.startswith("B")
    if label_mode == "completion_set":
        return "EXTRA" in parse_completion_set(prediction)
    return False


def make_model_args(args: argparse.Namespace, checkpoint: str | None, max_seq_length: int, resize: int) -> argparse.Namespace:
    return argparse.Namespace(
        eval_checkpoint=(checkpoint or None),
        model_name=args.model_name,
        max_seq_length=max_seq_length,
        vision_resize=resize,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        load_in_16bit=args.load_in_16bit,
    )


def make_collator(model: Any, tokenizer: Any, max_seq_length: int, resize: int) -> Qwen3VLMetadataVisionDataCollator:
    return Qwen3VLMetadataVisionDataCollator(
        model,
        tokenizer,
        max_seq_length=max_seq_length,
        resize=resize,
        train_on_responses_only=True,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
        completion_only_loss=True,
    )


def _score_module_a_window(
    *,
    args: argparse.Namespace,
    model: Any,
    generator: ModuleADetectionMetricsCallback,
    dataset: ModuleTrainingDataset,
    window: WindowSpec,
) -> tuple[WindowSpec, str, str, float]:
    return _score_module_a_example(
        args=args,
        model=model,
        generator=generator,
        window=window,
        training_example=window_to_example(window, dataset),
    )


def _score_module_a_example(
    *,
    args: argparse.Namespace,
    model: Any,
    generator: ModuleADetectionMetricsCallback,
    window: WindowSpec,
    training_example: TrainingExample,
) -> tuple[WindowSpec, str, str, float]:
    label_mode = args.module_a_label_mode
    example = module_a_to_conversation(
        training_example,
        fps=args.fps,
        min_frames=args.min_frames,
        max_frames=args.max_frames_a,
    )
    if args.label_score_eval:
        started = time.perf_counter()
        candidate_nlls = generator._score_label_nlls(model, example)
        elapsed = time.perf_counter() - started
        prediction = min(candidate_nlls, key=candidate_nlls.get)
        raw = json.dumps(
            {
                "prediction": prediction,
                "candidate_nlls": candidate_nlls,
                "messages": json_safe(example.get("messages", [])),
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return window, prediction, raw, elapsed
    started = time.perf_counter()
    generated_text = generator._generate_label(model, example)
    elapsed = time.perf_counter() - started
    prediction = extract_module_a_label(
        generated_text,
        label_mode=label_mode,
    )
    raw = json.dumps(
        {
            "prediction": prediction,
            "generated_text": generated_text,
            "messages": json_safe(example.get("messages", [])),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return window, prediction, raw, elapsed


def run_module_a(
    args: argparse.Namespace,
    windows: list[WindowSpec],
    dataset: ModuleTrainingDataset,
    *,
    claimed_video_ids: list[str] | None = None,
) -> list[tuple[WindowSpec, str, str, float]]:
    queue_dir = getattr(args, "video_queue_dir", None)
    if not windows and not queue_dir:
        return []
    model, tokenizer = load_generation_eval_model(
        make_model_args(args, args.module_a_checkpoint, args.max_seq_length_a, args.vision_resize_a)
    )
    collator = make_collator(model, tokenizer, args.max_seq_length_a, args.vision_resize_a)
    generator = ModuleADetectionMetricsCallback(
        eval_examples=[],
        data_collator=collator,
        tokenizer=tokenizer,
        max_samples=-1,
        max_new_tokens=args.module_a_max_new_tokens,
        beta=args.fbeta_beta,
        label_score_eval=args.label_score_eval,
    )
    predictions: list[tuple[WindowSpec, str, str, float]] = []
    try:
        if queue_dir:
            windows_by_video: dict[str, list[WindowSpec]] = {}
            for window in windows:
                windows_by_video.setdefault(window.video_id, []).append(window)
            while True:
                if args.max_samples >= 0 and len(predictions) >= args.max_samples:
                    break
                video_id = claim_next_video(queue_dir, rank=args.video_shard_rank)
                if video_id is None:
                    break
                if claimed_video_ids is not None:
                    claimed_video_ids.append(video_id)
                video_windows = windows_by_video.get(video_id, [])
                if args.max_samples >= 0:
                    video_windows = video_windows[: args.max_samples - len(predictions)]
                if RunControl.stop_requested:
                    break
                print(
                    f"Claimed video {video_id} "
                    f"({len(video_windows)} windows, {len(predictions)} done)",
                    flush=True,
                )
                video_predictions = [
                    _score_module_a_window(
                        args=args,
                        model=model,
                        generator=generator,
                        dataset=dataset,
                        window=window,
                    )
                    for window in video_windows
                ]
                predictions.extend(video_predictions)
                checkpoint_root = getattr(args, "video_checkpoint_dir", None)
                if checkpoint_root:
                    write_module_a_checkpoint(
                        checkpoint_root,
                        video_id,
                        module_a_rows=[
                            serialize_module_a_prediction(item) for item in video_predictions
                        ],
                        ground_truth_rows=[
                            json_safe(asdict(event))
                            for event in gt_events_for_records(
                                dataset.records,
                                {video_id},
                                label_mode=args.module_a_label_mode,
                            )
                        ],
                    )
                    if queue_dir:
                        enqueue_video(queue_dir, video_id)
                    print(f"Checkpointed Module A for {video_id}", flush=True)
        else:
            for window in windows:
                predictions.append(
                    _score_module_a_window(
                        args=args,
                        model=model,
                        generator=generator,
                        dataset=dataset,
                        window=window,
                    )
                )
    finally:
        del model
        release_cuda_memory()
    return predictions


def explode_module_a_trigger(
    window: WindowSpec,
    prediction: str,
    raw: str,
    elapsed: float,
    *,
    label_mode: str,
) -> list[tuple[WindowSpec, str, str, float]]:
    if label_mode != "completion_set":
        return [(window, prediction, raw, elapsed)]
    step_ids = parse_completion_set(prediction)
    if not step_ids:
        return []
    occurrences = list(window.dataset_metadata.get("completed_occurrences") or [])
    exploded: list[tuple[WindowSpec, str, str, float]] = []
    for step_id in step_ids:
        index = int(step_id.removeprefix("STEP_")) if step_id.startswith("STEP_") else -1
        instruction = (
            window.all_instructions[index]
            if 0 <= index < len(window.all_instructions)
            else window.current_step
        )
        match = next((item for item in occurrences if item.get("step_id") == step_id), None)
        exploded.append(
            (
                replace(
                    window,
                    step_index=index if match is None else int(match.get("instruction_index", index)),
                    current_step=str(match.get("instruction") if match else instruction),
                    occurrence_id=str(match.get("occurrence_id") if match else window.occurrence_id),
                    annotation_position=int(
                        match.get("annotation_position") if match else window.annotation_position
                    ),
                    gt_start=None if match is None else float(match["gt_start"]),
                    gt_end=None if match is None else float(match["gt_end"]),
                    completed_step_ids=(step_id,),
                ),
                step_id,
                raw,
                elapsed,
            )
        )
    return exploded


def b_example_from_window(window: WindowSpec, instruction: str) -> TrainingExample:
    clip_duration = window.window_end - window.window_start
    return TrainingExample(
        module="B",
        source_dataset="egoper",
        video_path=window.video_path,
        video_id=window.video_id,
        task_id=window.task_id,
        step_index=window.step_index,
        window_start=window.window_start,
        window_end=window.window_end,
        gt_start=window.gt_start,
        gt_end=window.gt_end,
        prompt_text=build_module_b_prompt(instruction, clip_duration),
        target_text="not completed",
        label="LOCALIZE",
        metadata={"instruction": instruction, "clip_duration": clip_duration},
    )


def run_module_b(args: argparse.Namespace, triggers: list[tuple[WindowSpec, str, str, float]]) -> list[PipelineEvent]:
    if not triggers:
        return []
    label_mode = args.module_a_label_mode
    model, tokenizer = load_generation_eval_model(
        make_model_args(args, args.module_b_checkpoint, args.max_seq_length_b, args.vision_resize_b)
    )
    collator = make_collator(model, tokenizer, args.max_seq_length_b, args.vision_resize_b)
    generator = ModuleBTemporalMetricsCallback(
        eval_examples=[],
        data_collator=collator,
        tokenizer=tokenizer,
        max_samples=-1,
        max_new_tokens=args.module_b_max_new_tokens,
    )
    events: list[PipelineEvent] = []
    try:
        for window, module_a_prediction, module_a_raw, module_a_elapsed in triggers:
            for (
                exploded_window,
                exploded_prediction,
                exploded_raw,
                exploded_elapsed,
            ) in explode_module_a_trigger(
                window,
                module_a_prediction,
                module_a_raw,
                module_a_elapsed,
                label_mode=label_mode,
            ):
                events.append(
                    run_module_b_one(
                        args,
                        model=model,
                        generator=generator,
                        window=exploded_window,
                        module_a_prediction=exploded_prediction,
                        module_a_raw=exploded_raw,
                        module_a_elapsed=exploded_elapsed,
                        label_mode=label_mode,
                    )
                )
    finally:
        del model
        release_cuda_memory()
    return events


def run_module_b_one(
    args: argparse.Namespace,
    *,
    model: Any,
    generator: ModuleBTemporalMetricsCallback,
    window: WindowSpec,
    module_a_prediction: str,
    module_a_raw: str,
    module_a_elapsed: float,
    label_mode: str,
) -> PipelineEvent:
    instruction = resolved_instruction_for_prediction(
        window=window,
        prediction=module_a_prediction,
        label_mode=label_mode,
    )
    example = b_example_from_window(window, instruction)
    conversation = module_b_to_conversation(
        example,
        fps=args.fps,
        min_frames=args.min_frames,
        max_frames=args.max_frames_b,
    )
    started = time.perf_counter()
    raw = generator._generate_span(model, conversation)
    module_b_elapsed = time.perf_counter() - started
    parsed_result = parse_bounded_temporal_prediction(
        raw,
        clip_duration=window.window_end - window.window_start,
    )
    parsed = parsed_result.span
    raw_parsed = parsed_result.raw_span
    event = PipelineEvent(
        video_id=window.video_id,
        task_id=window.task_id,
        step_index=window.step_index,
        instruction=instruction,
        occurrence_id=window.occurrence_id,
        annotation_position=window.annotation_position,
        module_a_completed_steps=window.completed_step_ids or tuple(parse_completion_set(module_a_prediction)),
        module_a_completed_instructions=tuple(window.completed_steps),
        trigger_time=window.window_end,
        accumulation_start=window.window_start,
        accumulation_end=window.window_end,
        module_a_prediction=module_a_prediction,
        module_a_raw=module_a_raw,
        module_b_raw=raw,
        module_b_prompt_text=example.prompt_text,
        module_b_messages=json_safe(conversation.get("messages", [])),
        module_b_raw_prediction=raw_parsed,
        module_b_prediction=parsed,
        module_b_rejection_reason=parsed_result.rejection_reason,
        module_b_was_clamped=parsed_result.was_clamped,
        module_b_clamp_reason=parsed_result.clamp_reason,
        module_a_prediction_seconds=module_a_elapsed,
        module_b_prediction_seconds=module_b_elapsed,
    )
    if raw_parsed is not None:
        event.pred_raw_global_start = window.window_start + raw_parsed[0]
        event.pred_raw_global_end = window.window_start + raw_parsed[1]
    if parsed is not None:
        start = window.window_start + parsed[0]
        end = window.window_start + parsed[1]
        if end > start:
            event.pred_global_start = start
            event.pred_global_end = end
    if parsed_result.rejection_reason == "no_action" or is_not_completed_text(raw):
        event.stage_status = "B_NOT_COMPLETED"
    elif parsed is None:
        event.stage_status = "B_INVALID_PARSE"
    return event


def select_triggers_with_online_b_reset(
    args: argparse.Namespace,
    module_a_predictions: list[tuple[WindowSpec, str, str, float]],
) -> tuple[list[tuple[WindowSpec, str, str, float]], list[PipelineEvent]]:
    """Walk Module A predictions chronologically per video; after each trigger, run B and skip later windows whose start is before B's predicted end (fallback: accumulation end)."""
    by_video: dict[str, list[tuple[WindowSpec, str, str, float]]] = {}
    for item in module_a_predictions:
        by_video.setdefault(item[0].video_id, []).append(item)
    for items in by_video.values():
        items.sort(key=lambda row: (row[0].window_start, row[0].window_end, row[0].step_index))

    model, tokenizer = load_generation_eval_model(
        make_model_args(args, args.module_b_checkpoint, args.max_seq_length_b, args.vision_resize_b)
    )
    collator = make_collator(model, tokenizer, args.max_seq_length_b, args.vision_resize_b)
    generator = ModuleBTemporalMetricsCallback(
        eval_examples=[],
        data_collator=collator,
        tokenizer=tokenizer,
        max_samples=-1,
        max_new_tokens=args.module_b_max_new_tokens,
    )
    triggers: list[tuple[WindowSpec, str, str, float]] = []
    events: list[PipelineEvent] = []
    try:
        for video_id in sorted(by_video):
            video_triggers, video_events = _module_b_reset_one_video(
                args,
                model=model,
                generator=generator,
                items=by_video[video_id],
            )
            triggers.extend(video_triggers)
            events.extend(video_events)
    finally:
        del model
        release_cuda_memory()
    return triggers, events


def _module_b_reset_one_video(
    args: argparse.Namespace,
    *,
    model: Any,
    generator: ModuleBTemporalMetricsCallback,
    items: list[tuple[WindowSpec, str, str, float]],
) -> tuple[list[tuple[WindowSpec, str, str, float]], list[PipelineEvent]]:
    label_mode = args.module_a_label_mode
    ordered = sorted(items, key=lambda row: (row[0].window_start, row[0].window_end, row[0].step_index))
    triggers: list[tuple[WindowSpec, str, str, float]] = []
    events: list[PipelineEvent] = []
    stream_start = 0.0
    for window, prediction, raw, elapsed in ordered:
        if window.window_start + 1e-6 < stream_start:
            continue
        if not is_module_a_completion_label(prediction, label_mode=label_mode):
            continue
        exploded = explode_module_a_trigger(
            window, prediction, raw, elapsed, label_mode=label_mode
        )
        if not exploded:
            continue
        triggers.extend(exploded)
        ends: list[float] = []
        for exploded_window, exploded_prediction, exploded_raw, exploded_elapsed in exploded:
            event = run_module_b_one(
                args,
                model=model,
                generator=generator,
                window=exploded_window,
                module_a_prediction=exploded_prediction,
                module_a_raw=exploded_raw,
                module_a_elapsed=exploded_elapsed,
                label_mode=label_mode,
            )
            events.append(event)
            if event.pred_global_end is not None:
                ends.append(float(event.pred_global_end))
        if label_mode != "completion_set" and ends:
            stream_start = min(ends)
    return triggers, events


def run_bc_from_queue(
    args: argparse.Namespace,
    *,
    dataset: ModuleTrainingDataset,
    checkpoint_root: str | Path,
) -> list[PipelineEvent]:
    pending: dict[str, list[PipelineEvent]] = {}
    model, tokenizer = load_generation_eval_model(
        make_model_args(args, args.module_b_checkpoint, args.max_seq_length_b, args.vision_resize_b)
    )
    collator = make_collator(model, tokenizer, args.max_seq_length_b, args.vision_resize_b)
    generator = ModuleBTemporalMetricsCallback(
        eval_examples=[],
        data_collator=collator,
        tokenizer=tokenizer,
        max_samples=-1,
        max_new_tokens=args.module_b_max_new_tokens,
    )
    try:
        while True:
            video_id = claim_next_video(
                args.video_queue_dir,
                rank=args.video_shard_rank,
                remaining_name="bc_remaining.json",
                claims_name="bc_claims.jsonl",
            )
            if video_id is None:
                break
            if events_checkpoint_path(checkpoint_root, video_id).is_file():
                continue
            predictions = [
                module_a_prediction_from_row(row)
                for row in load_module_a_rows(checkpoint_root, video_id)
            ]
            print(f"Module B claimed {video_id} ({len(predictions)} A polls)", flush=True)
            _triggers, events = _module_b_reset_one_video(
                args,
                model=model,
                generator=generator,
                items=predictions,
            )
            pending[video_id] = events
            if RunControl.stop_requested:
                break
    finally:
        del model
        release_cuda_memory()
    finished: list[PipelineEvent] = []
    if not pending:
        return finished
    all_events = [event for events in pending.values() for event in events]
    run_module_c(args, all_events, attach_video_paths(all_events, {}, records=dataset.records))
    apply_graph_stage(all_events, dataset.records)
    for video_id, events in pending.items():
        write_events_checkpoint(checkpoint_root, video_id, [event_row(event) for event in events])
        finished.extend(events)
        print(f"Checkpointed Module B/C for {video_id}", flush=True)
    return finished


def record_duration(record: VideoRecord) -> float:
    duration = video_duration_seconds(record.video_path)
    if duration <= 0:
        duration = max((segment.end for segment in record.segments), default=0.0)
    return float(duration)


def run_causal_ab(
    args: argparse.Namespace,
    *,
    dataset: ModuleTrainingDataset,
    video_ids: list[str],
    claimed_video_ids: list[str],
) -> tuple[list[tuple[WindowSpec, str, str, float]], list[PipelineEvent], list[dict[str, Any]]]:
    """Stream each video through Module A and Module B with both models resident."""

    queue_dir = getattr(args, "video_queue_dir", None)
    checkpoint_root = getattr(args, "video_checkpoint_dir", None)
    records = {record.video_id: record for record in dataset.records}
    c_in_loop = bool(getattr(args, "causal_module_c_in_loop", False))
    bank: LoraAdapterBank | None = None
    if c_in_loop:
        model_a, tokenizer_a, bank = load_shared_abc_model(args)
        model_b, tokenizer_b = model_a, tokenizer_a
    else:
        model_a, tokenizer_a = load_generation_eval_model(
            make_model_args(args, args.module_a_checkpoint, args.max_seq_length_a, args.vision_resize_a)
        )
        model_b, tokenizer_b = load_generation_eval_model(
            make_model_args(args, args.module_b_checkpoint, args.max_seq_length_b, args.vision_resize_b)
        )
    generator_a = ModuleADetectionMetricsCallback(
        eval_examples=[],
        data_collator=make_collator(model_a, tokenizer_a, args.max_seq_length_a, args.vision_resize_a),
        tokenizer=tokenizer_a,
        max_samples=-1,
        max_new_tokens=args.module_a_max_new_tokens,
        beta=args.fbeta_beta,
        label_score_eval=args.label_score_eval,
    )
    generator_b = ModuleBTemporalMetricsCallback(
        eval_examples=[],
        data_collator=make_collator(model_b, tokenizer_b, args.max_seq_length_b, args.vision_resize_b),
        tokenizer=tokenizer_b,
        max_samples=-1,
        max_new_tokens=args.module_b_max_new_tokens,
    )
    run_c = None
    if bank is not None:
        generator_c = ModuleCMistakeMetricsCallback(
            eval_examples=[],
            data_collator=make_collator(model_a, tokenizer_a, args.max_seq_length_c, args.vision_resize_c),
            tokenizer=tokenizer_a,
            max_samples=-1,
            max_new_tokens=args.module_c_max_new_tokens,
        )

        def run_c(event: PipelineEvent, video_events: list[PipelineEvent]) -> None:
            if mark_module_c_extra(args, event):
                return
            bank.activate("C")
            classify_module_c_event(
                args,
                model=model_a,
                generator=generator_c,
                events=video_events,
                index=len(video_events) - 1,
                video_path=records[event.video_id].video_path,
            )

    def score_a(window: WindowSpec, example: TrainingExample) -> tuple[str, str, float]:
        if bank is not None:
            bank.activate("A")
        _window, prediction, raw, elapsed = _score_module_a_example(
            args=args,
            model=model_a,
            generator=generator_a,
            window=window,
            training_example=example,
        )
        return prediction, raw, elapsed

    def run_b(window: WindowSpec, prediction: str, raw: str, elapsed: float) -> PipelineEvent:
        if bank is not None:
            bank.activate("B")
        return run_module_b_one(
            args,
            model=model_b,
            generator=generator_b,
            window=window,
            module_a_prediction=prediction,
            module_a_raw=raw,
            module_a_elapsed=elapsed,
            label_mode="step_id",
        )

    pending = list(video_ids)
    predictions: list[tuple[WindowSpec, str, str, float]] = []
    events: list[PipelineEvent] = []
    poll_rows: list[dict[str, Any]] = []
    try:
        while not RunControl.stop_requested:
            if 0 <= args.causal_max_videos <= len(claimed_video_ids):
                break
            if queue_dir:
                video_id = claim_next_video(queue_dir, rank=args.video_shard_rank)
            else:
                video_id = pending.pop(0) if pending else None
            if video_id is None:
                break
            claimed_video_ids.append(video_id)
            record = records[video_id]
            print(f"Causal A+B claimed {video_id}", flush=True)
            result: CausalVideoResult = run_causal_video(
                record,
                duration=record_duration(record),
                score_a=score_a,
                run_b=run_b,
                run_c=run_c,
                stride_seconds=args.stride_seconds,
            )
            if c_in_loop:
                apply_graph_stage(result.events, dataset.records)
            predictions.extend(result.module_a_predictions)
            events.extend(result.events)
            video_poll_rows = json_safe(result.poll_rows)
            poll_rows.extend(video_poll_rows)
            print(
                f"Causal {'A+B+C' if c_in_loop else 'A+B'} {video_id}: {len(result.poll_rows)} polls, "
                f"{len(result.events)} Module B calls",
                flush=True,
            )
            if checkpoint_root:
                event_rows = [event_row(event) for event in result.events]
                if c_in_loop:
                    write_events_checkpoint(checkpoint_root, video_id, event_rows)
                else:
                    write_causal_ab_events(checkpoint_root, video_id, event_rows)
                write_module_a_checkpoint(
                    checkpoint_root,
                    video_id,
                    module_a_rows=video_poll_rows,
                    ground_truth_rows=[
                        json_safe(asdict(event))
                        for event in gt_events_for_records(
                            dataset.records, {video_id}, label_mode="step_id"
                        )
                    ],
                )
                if queue_dir and not c_in_loop:
                    enqueue_video(queue_dir, video_id)
    finally:
        del model_a, model_b, bank
        release_cuda_memory()
    return predictions, events, poll_rows


def load_shared_abc_model(args: argparse.Namespace) -> tuple[Any, Any, LoraAdapterBank]:
    """Load the base once with Module A's adapter and keep B's and C's weights for swapping."""

    paths = {
        "A": args.module_a_checkpoint,
        "B": args.module_b_checkpoint,
        "C": args.module_c_checkpoint,
    }
    missing = sorted(name for name, path in paths.items() if not path)
    if missing:
        raise ValueError(f"--causal-module-c-in-loop needs checkpoints for modules {missing}")
    check_adapter_configs(paths)
    max_seq_length = max(args.max_seq_length_a, args.max_seq_length_b, args.max_seq_length_c)
    model, tokenizer = load_generation_eval_model(
        make_model_args(args, args.module_a_checkpoint, max_seq_length, args.vision_resize_a)
    )
    bank = LoraAdapterBank(
        model,
        primary="A",
        adapters={name: load_adapter_weights(paths[name]) for name in ("B", "C")},
    )
    print(
        f"Shared base with adapters {sorted(bank.states)}: {len(bank.parameters)} LoRA tensors",
        flush=True,
    )
    return model, tokenizer, bank


def _causal_event_from_row(row: dict[str, Any]) -> PipelineEvent:
    event = event_from_row(row)
    event.module_a_completed_steps = tuple(row.get("module_a_completed_steps") or ())
    event.module_a_completed_instructions = tuple(row.get("module_a_completed_instructions") or ())
    return event


def run_causal_c_from_queue(
    args: argparse.Namespace,
    *,
    dataset: ModuleTrainingDataset,
    checkpoint_root: str | Path,
) -> list[PipelineEvent]:
    pending: dict[str, list[PipelineEvent]] = {}
    while True:
        video_id = claim_next_video(
            args.video_queue_dir,
            rank=args.video_shard_rank,
            remaining_name="bc_remaining.json",
            claims_name="bc_claims.jsonl",
        )
        if video_id is None:
            break
        if events_checkpoint_path(checkpoint_root, video_id).is_file():
            continue
        pending[video_id] = [
            _causal_event_from_row(row) for row in load_causal_ab_events(checkpoint_root, video_id)
        ]
    finished: list[PipelineEvent] = []
    if not pending:
        return finished
    all_events = [event for events in pending.values() for event in events]
    run_module_c(args, all_events, attach_video_paths(all_events, {}, records=dataset.records))
    apply_graph_stage(all_events, dataset.records)
    for video_id, events in pending.items():
        write_events_checkpoint(checkpoint_root, video_id, [event_row(event) for event in events])
        finished.extend(events)
        print(f"Checkpointed Module C for {video_id}", flush=True)
    return finished


def c_example_from_event(
    event: PipelineEvent,
    *,
    dataset_name: str,
    history: list[dict[str, str]] | None = None,
    history_max_events: int = 12,
    clip: str = "span",
) -> TrainingExample | None:
    """Module C input for a placed step."""
    if event.pred_global_start is None or event.pred_global_end is None:
        return None
    if clip not in MODULE_C_CLIPS:
        raise ValueError(f"clip must be one of {MODULE_C_CLIPS}, got {clip!r}")
    if clip == "window":
        clip_start, clip_end = event.accumulation_start, event.accumulation_end
    else:
        clip_start, clip_end = event.pred_global_start, event.pred_global_end
    prompt = build_module_c_prompt(
        event.instruction,
        EGOPER_MODULE_C_CLASSES,
        dataset_name="egoper",
    )
    history_rows = [] if history is None else history
    prompt = append_module_c_history(
        prompt,
        history_rows,
        max_events=history_max_events,
    )
    return TrainingExample(
        module="C",
        source_dataset=dataset_name,
        video_path=Path(""),
        video_id=event.video_id,
        task_id=event.task_id,
        step_index=event.step_index,
        window_start=clip_start,
        window_end=clip_end,
        prompt_text=prompt,
        target_text='{"classification":"CORRECT","reasoning":""}',
        label=None,
        metadata={
            "instruction": event.instruction,
            "module_c_history": (
                history_rows[-history_max_events:]
                if history_max_events > 0
                else []
            ),
            "module_c_history_mode": "predicted",
            "module_c_clip": clip,
        },
    )


def predicted_module_c_history(
    events: list[PipelineEvent],
    *,
    current_index: int,
    max_events: int,
) -> list[dict[str, str]]:
    """Build causal history from earlier pipeline predictions in one video."""
    if max_events <= 0:
        return []
    current = events[current_index]
    history: list[dict[str, str]] = []
    for event in events[:current_index]:
        if (
            event.video_id != current.video_id
            or not event.module_c_classification
            or event.stage_status == "C_SKIPPED_EXTRA"
        ):
            continue
        history.append(
            {
                "action": event.instruction,
                "classification": event.module_c_classification,
                "evidence": (
                    event.module_c_reasoning
                    or "No evidence was recorded."
                ),
            }
        )
    return history[-max_events:]


def attach_video_paths(
    events: list[PipelineEvent],
    windows_by_key: dict[tuple[str, int], WindowSpec],
    records: list[VideoRecord] | None = None,
) -> dict[int, Path]:
    paths: dict[int, Path] = {}
    record_paths = {record.video_id: record.video_path for record in (records or [])}
    for index, event in enumerate(events):
        window = windows_by_key.get((event.video_id, event.step_index))
        if window is not None:
            paths[index] = window.video_path
        elif event.video_id in record_paths:
            paths[index] = record_paths[event.video_id]
    return paths


def event_from_row(row: dict[str, Any]) -> PipelineEvent:
    module_b = row.get("module_b_prediction")
    if isinstance(module_b, list) and len(module_b) >= 2:
        module_b_prediction = (float(module_b[0]), float(module_b[1]))
    else:
        module_b_prediction = None
    module_b_raw = row.get("module_b_raw_prediction")
    if isinstance(module_b_raw, list) and len(module_b_raw) >= 2:
        module_b_raw_prediction = (float(module_b_raw[0]), float(module_b_raw[1]))
    else:
        module_b_raw_prediction = None
    return PipelineEvent(
        video_id=str(row["video_id"]),
        task_id=str(row["task_id"]),
        step_index=int(row["step_index"]),
        instruction=str(row["instruction"]),
        trigger_time=float(row["trigger_time"]),
        accumulation_start=float(row["accumulation_start"]),
        accumulation_end=float(row["accumulation_end"]),
        module_a_prediction=str(row.get("module_a_prediction") or "COMPLETE"),
        module_a_raw=row.get("module_a_raw"),
        module_b_raw=row.get("module_b_raw"),
        module_b_prompt_text=row.get("module_b_prompt_text"),
        module_b_messages=list(row.get("module_b_messages") or []),
        module_b_raw_prediction=module_b_raw_prediction,
        module_b_prediction=module_b_prediction,
        module_b_rejection_reason=row.get("module_b_rejection_reason"),
        module_b_was_clamped=bool(row.get("module_b_was_clamped", False)),
        module_b_clamp_reason=row.get("module_b_clamp_reason"),
        pred_raw_global_start=(
            None if row.get("pred_raw_global_start") is None else float(row["pred_raw_global_start"])
        ),
        pred_raw_global_end=(
            None if row.get("pred_raw_global_end") is None else float(row["pred_raw_global_end"])
        ),
        pred_global_start=None if row.get("pred_global_start") is None else float(row["pred_global_start"]),
        pred_global_end=None if row.get("pred_global_end") is None else float(row["pred_global_end"]),
        module_a_prediction_seconds=row.get("module_a_prediction_seconds"),
        module_b_prediction_seconds=row.get("module_b_prediction_seconds"),
        # Module C fields are cleared by the caller before re-inference.
        module_c_raw=row.get("module_c_raw"),
        module_c_prompt_text=row.get("module_c_prompt_text"),
        module_c_messages=list(row.get("module_c_messages") or []),
        module_c_history=list(row.get("module_c_history") or []),
        module_c_classification=row.get("module_c_classification"),
        module_c_aggregate_class=row.get("module_c_aggregate_class"),
        module_c_mistake=row.get("module_c_mistake"),
        module_c_error_score=(
            None
            if row.get("module_c_error_score") is None
            else float(row["module_c_error_score"])
        ),
        module_c_class_probabilities=(
            None
            if row.get("module_c_class_probabilities") is None
            else {
                str(key): float(value)
                for key, value in row["module_c_class_probabilities"].items()
            }
        ),
        module_c_candidate_nlls=(
            None
            if row.get("module_c_candidate_nlls") is None
            else {
                str(key): float(value)
                for key, value in row["module_c_candidate_nlls"].items()
            }
        ),
        module_c_reasoning=row.get("module_c_reasoning"),
        module_c_prediction_seconds=row.get("module_c_prediction_seconds"),
        module_c_score_seconds=row.get("module_c_score_seconds"),
        pipeline_prediction_seconds=row.get("pipeline_prediction_seconds"),
        matched_gt_index=row.get("matched_gt_index"),
        matched_iou=float(row.get("matched_iou") or 0.0),
        matched_gt_start=None if row.get("matched_gt_start") is None else float(row["matched_gt_start"]),
        matched_gt_end=None if row.get("matched_gt_end") is None else float(row["matched_gt_end"]),
        gt_mistake=row.get("gt_mistake"),
        gt_reasoning=row.get("gt_reasoning"),
        gt_fine_class=row.get("gt_fine_class"),
        gt_aggregate_class=row.get("gt_aggregate_class"),
        duplicate=bool(row.get("duplicate")),
        stage_status=row.get("stage_status"),
    )


def load_events_for_module_c_reuse(
    path: Path,
    *,
    clear_module_c: bool = True,
) -> list[PipelineEvent]:
    with open(path, encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict) or "events" not in payload:
        raise ValueError(f"Expected pipeline metrics JSON with an 'events' list: {path}")
    events = [event_from_row(row) for row in payload["events"]]
    if not clear_module_c:
        return events
    for event in events:
        event.module_c_raw = None
        event.module_c_prompt_text = None
        event.module_c_messages = []
        event.module_c_history = []
        event.module_c_classification = None
        event.module_c_aggregate_class = None
        event.module_c_mistake = None
        event.module_c_error_score = None
        event.module_c_class_probabilities = None
        event.module_c_candidate_nlls = None
        event.module_c_reasoning = None
        event.module_c_prediction_seconds = None
        event.module_c_score_seconds = None
        event.pipeline_prediction_seconds = None
        event.matched_gt_index = None
        event.matched_iou = 0.0
        event.matched_gt_start = None
        event.matched_gt_end = None
        event.gt_mistake = None
        event.gt_reasoning = None
        event.gt_fine_class = None
        event.gt_aggregate_class = None
        event.duplicate = False
    return events


def reconstruct_module_a_predictions(
    *,
    windows: list[WindowSpec],
    events: list[PipelineEvent],
) -> list[tuple[WindowSpec, str, str, float]]:
    trigger_keys = {
        (
            event.video_id,
            event.step_index,
            round(event.accumulation_start, 6),
            round(event.accumulation_end, 6),
        )
        for event in events
    }
    predictions: list[tuple[WindowSpec, str, str, float]] = []
    for window in windows:
        key = (
            window.video_id,
            window.step_index,
            round(window.window_start, 6),
            round(window.window_end, 6),
        )
        prediction = "COMPLETE" if key in trigger_keys else "WAIT"
        predictions.append((window, prediction, prediction, float(0.0)))
    return predictions


def module_a_extra_probability(raw_output: str | None) -> float | None:
    """Recover P(EXTRA) from Module A's saved A/B/C+ candidate NLLs."""

    if not raw_output:
        return None
    try:
        payload = json.loads(raw_output)
        raw_scores = payload["candidate_nlls"]
        if not isinstance(raw_scores, dict) or "B" not in raw_scores:
            return None
        probabilities = scores_to_probabilities({
            str(label): float(value)
            for label, value in raw_scores.items()
        })
        return float(probabilities["B"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def mark_module_c_extra(args: argparse.Namespace, event: PipelineEvent) -> bool:
    """Classify an extra step without Module C when the dataset allows it."""

    label_mode = getattr(args, "module_a_label_mode", "step_id")
    is_extra = (
        is_extra_module_a_prediction(event.module_a_prediction, label_mode=label_mode)
        or event.step_index < 0
    )
    if not is_extra:
        return False
    event.module_c_classification = "ADDITION"
    event.module_c_aggregate_class = "MISTAKE"
    event.module_c_mistake = True
    if getattr(args, "module_c_candidate_scoring", False):
        extra_probability = module_a_extra_probability(event.module_a_raw)
        if extra_probability is not None:
            event.module_c_error_score = extra_probability
            event.module_c_class_probabilities = {
                "CORRECT": 1.0 - extra_probability,
                "ADDITION": extra_probability,
            }
    event.stage_status = "C_SKIPPED_EXTRA"
    return True


def classify_module_c_event(
    args: argparse.Namespace,
    *,
    model: Any,
    generator: ModuleCMistakeMetricsCallback,
    events: list[PipelineEvent],
    index: int,
    video_path: Path,
) -> None:
    """Run Module C on ``events[index]`` with history from the earlier events."""

    dataset_name = getattr(args, "dataset", "egoper")
    score_candidates = bool(getattr(args, "module_c_candidate_scoring", False))
    score_only = bool(getattr(args, "module_c_score_only", False))
    event = events[index]
    module_c_history = predicted_module_c_history(
        events,
        current_index=index,
        max_events=args.module_c_history_max_events,
    )
    example = c_example_from_event(
        event,
        dataset_name=dataset_name,
        history=module_c_history,
        history_max_events=args.module_c_history_max_events,
        clip=getattr(args, "module_c_clip", "span"),
    )
    if example is None:
        return
    event.module_c_history = module_c_history
    event.module_c_prompt_text = example.prompt_text
    example = TrainingExample(**{**asdict(example), "video_path": video_path})
    conversation = module_c_to_conversation(
        example,
        fps=args.fps,
        min_frames=args.min_frames,
        max_frames=args.max_frames_c,
    )
    event.module_c_messages = json_safe(conversation.get("messages", []))
    if score_candidates:
        score_started = time.perf_counter()
        candidate_nlls = generator._score_class_nlls(
            model,
            conversation,
            list(EGOPER_MODULE_C_CLASSES),
        )
        class_probabilities = scores_to_probabilities(candidate_nlls)
        event.module_c_candidate_nlls = candidate_nlls
        event.module_c_class_probabilities = class_probabilities
        event.module_c_error_score = sum(
            probability
            for class_name, probability in class_probabilities.items()
            if class_name != "CORRECT"
        )
        event.module_c_score_seconds = time.perf_counter() - score_started
    if score_only:
        return
    started = time.perf_counter()
    raw = generator._generate_json(model, conversation)
    module_c_elapsed = time.perf_counter() - started
    parsed = parse_module_c_prediction(
        raw,
        allowed_classes=EGOPER_MODULE_C_CLASSES,
        aggregate_mapping=EGOPER_AGGREGATE_MAPPING,
    )
    event.module_c_raw = raw
    event.module_c_prediction_seconds = module_c_elapsed
    event.pipeline_prediction_seconds = sum(
        value
        for value in (
            event.module_a_prediction_seconds,
            event.module_b_prediction_seconds,
            event.module_c_prediction_seconds,
        )
        if value is not None
    )
    if is_not_completed_text(raw) or (
        isinstance(parsed, dict) and parsed.get("classification") == NOT_COMPLETED_STATUS
    ):
        event.stage_status = "C_NOT_COMPLETED"
        return
    if parsed is not None:
        event.module_c_classification = str(parsed["classification"])
        event.module_c_aggregate_class = (
            "MISTAKE"
            if parsed["mistake"]
            else "CORRECT"
            if parsed["classification"] == "CORRECT"
            else "CORRECTION"
        )
        event.module_c_mistake = bool(parsed["mistake"])
        event.module_c_reasoning = str(parsed["reasoning"])
        event.stage_status = "C_CLASSIFIED"
    else:
        event.stage_status = "C_INVALID_PARSE"


def run_module_c(args: argparse.Namespace, events: list[PipelineEvent], video_paths: dict[int, Path]) -> None:
    score_candidates = bool(getattr(args, "module_c_candidate_scoring", False))
    for event in events:
        if event.stage_status != "B_NOT_COMPLETED":
            mark_module_c_extra(args, event)
    valid_indices = [
        index for index, event in enumerate(events)
        if event.pred_global_start is not None
        and event.pred_global_end is not None
        and index in video_paths
        and event.stage_status not in {"B_NOT_COMPLETED", "C_SKIPPED_EXTRA"}
    ]
    if not valid_indices:
        return
    if score_candidates:
        os.environ["UNSLOTH_RETURN_LOGITS"] = "0"
        os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "1"
    model, tokenizer = load_generation_eval_model(
        make_model_args(args, args.module_c_checkpoint, args.max_seq_length_c, args.vision_resize_c)
    )
    collator = make_collator(model, tokenizer, args.max_seq_length_c, args.vision_resize_c)
    generator = ModuleCMistakeMetricsCallback(
        eval_examples=[],
        data_collator=collator,
        tokenizer=tokenizer,
        max_samples=-1,
        max_new_tokens=args.module_c_max_new_tokens,
    )
    try:
        for index in valid_indices:
            classify_module_c_event(
                args,
                model=model,
                generator=generator,
                events=events,
                index=index,
                video_path=video_paths[index],
            )
    finally:
        del model
        if score_candidates:
            os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "0"
        release_cuda_memory()


def match_events(events: list[PipelineEvent], gt_events: list[GroundTruthEvent]) -> None:
    candidates: list[tuple[float, int, int]] = []
    for event_index, event in enumerate(events):
        if event.pred_global_start is None or event.pred_global_end is None:
            continue
        pred = (event.pred_global_start, event.pred_global_end)
        for gt_index, gt in enumerate(gt_events):
            if not gt.eligible:
                continue
            if event.video_id != gt.video_id:
                continue
            if event.occurrence_id and gt.occurrence_id:
                if event.occurrence_id != gt.occurrence_id:
                    continue
            elif event.step_index != gt.step_index:
                continue
            iou = temporal_iou(pred, (gt.gt_start, gt.gt_end))
            candidates.append((iou, event_index, gt_index))
    used_events: set[int] = set()
    used_gt: set[int] = set()
    for iou, event_index, gt_index in sorted(candidates, reverse=True):
        if event_index in used_events:
            continue
        event = events[event_index]
        if gt_index in used_gt:
            event.duplicate = True
            continue
        used_events.add(event_index)
        used_gt.add(gt_index)
        gt = gt_events[gt_index]
        event.matched_gt_index = gt_index
        event.matched_iou = iou
        event.matched_gt_start = gt.gt_start
        event.matched_gt_end = gt.gt_end
        event.gt_mistake = gt.gt_mistake
        event.gt_reasoning = gt.gt_reasoning
        event.gt_fine_class = gt.gt_fine_class
        event.gt_aggregate_class = gt.gt_aggregate_class
    for event in events:
        if event.matched_gt_index is None and event.pred_global_start is not None:
            for gt_index, gt in enumerate(gt_events):
                if event.video_id == gt.video_id and event.step_index == gt.step_index:
                    event.duplicate = gt_index in used_gt
                    break


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * q
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[int(rank)]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def latency_summary(latencies: list[float], *, prefix: str) -> dict[str, float]:
    early_count = sum(1 for value in latencies if value < 0)
    on_time_count = sum(1 for value in latencies if value == 0)
    late_count = sum(1 for value in latencies if value > 0)
    absolute = [abs(value) for value in latencies]
    return {
        f"{prefix}/count": float(len(latencies)),
        f"{prefix}/mean_seconds": sum(latencies) / len(latencies) if latencies else 0.0,
        f"{prefix}/median_seconds": percentile(latencies, 0.5),
        f"{prefix}/p90_seconds": percentile(latencies, 0.9),
        f"{prefix}/p95_seconds": percentile(latencies, 0.95),
        f"{prefix}/min_seconds": min(latencies) if latencies else 0.0,
        f"{prefix}/max_seconds": max(latencies) if latencies else 0.0,
        f"{prefix}/mean_absolute_seconds": sum(absolute) / len(absolute) if absolute else 0.0,
        f"{prefix}/median_absolute_seconds": percentile(absolute, 0.5),
        f"{prefix}/early_count": float(early_count),
        f"{prefix}/on_time_count": float(on_time_count),
        f"{prefix}/late_count": float(late_count),
        f"{prefix}/early_rate": safe_div(early_count, len(latencies)),
        f"{prefix}/on_time_rate": safe_div(on_time_count, len(latencies)),
        f"{prefix}/late_rate": safe_div(late_count, len(latencies)),
    }


def duration_summary(durations: list[float], *, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}/count": float(len(durations)),
        f"{prefix}/mean_seconds": sum(durations) / len(durations) if durations else 0.0,
        f"{prefix}/median_seconds": percentile(durations, 0.5),
        f"{prefix}/p90_seconds": percentile(durations, 0.9),
        f"{prefix}/p95_seconds": percentile(durations, 0.95),
        f"{prefix}/min_seconds": min(durations) if durations else 0.0,
        f"{prefix}/max_seconds": max(durations) if durations else 0.0,
    }


def binary_stats(tp: int, fp: int, tn: int, fn: int) -> dict[str, float]:
    total = tp + fp + tn + fn
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    accuracy = safe_div(tp + tn, total)
    f1 = safe_div(2 * precision * recall, precision + recall)
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "balanced_accuracy": (recall + specificity) / 2.0,
        "mcc": ((tp * tn) - (fp * fn)) / denominator if denominator else 0.0,
        "false_positive_rate": safe_div(fp, fp + tn),
        "false_negative_rate": safe_div(fn, fn + tp),
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
        "num_samples": float(total),
    }


def confusion_matrix(tp: int, fp: int, tn: int, fn: int) -> dict[str, dict[str, int]]:
    return {
        "gt_mistake": {"pred_mistake": tp, "pred_correct_or_missed": fn},
        "gt_correct_or_unmatched": {"pred_mistake": fp, "pred_correct_or_none": tn},
    }


def end_to_end_confusion_matrix(
    *,
    events: list[PipelineEvent],
    gt_events: list[GroundTruthEvent],
    threshold: float,
) -> dict[str, dict[str, int]]:
    eligible_indices = [index for index, gt in enumerate(gt_events) if gt.eligible]
    detected_gt_indices = {
        event.matched_gt_index
        for event in events
        if event.matched_gt_index is not None and event.matched_iou >= threshold
    }
    return {
        "gt_mistake": {
            "pred_mistake": sum(
                1 for event in events
                if event.matched_gt_index is not None
                and event.matched_iou >= threshold
                and event.gt_mistake is True
                and event.module_c_mistake is True
            ),
            "pred_correct": sum(
                1 for event in events
                if event.matched_gt_index is not None
                and event.matched_iou >= threshold
                and event.gt_mistake is True
                and event.module_c_mistake is not True
            ),
            "missed": sum(
                index not in detected_gt_indices
                for index in eligible_indices
                if gt_events[index].gt_mistake
            ),
        },
        "gt_correct": {
            "pred_mistake": sum(
                1 for event in events
                if event.matched_gt_index is not None
                and event.matched_iou >= threshold
                and event.gt_mistake is False
                and event.module_c_mistake is True
            ),
            "pred_correct": sum(
                1 for event in events
                if event.matched_gt_index is not None
                and event.matched_iou >= threshold
                and event.gt_mistake is False
                and event.module_c_mistake is not True
            ),
            "missed": sum(
                index not in detected_gt_indices
                for index in eligible_indices
                if not gt_events[index].gt_mistake
            ),
        },
        "unmatched_prediction": {
            "pred_mistake": sum(
                1 for event in events
                if event.module_c_mistake is True
                and (event.matched_gt_index is None or event.matched_iou < threshold or event.duplicate)
            ),
            "pred_correct": sum(
                1 for event in events
                if event.module_c_mistake is not True
                and event.pred_global_start is not None
                and (event.matched_gt_index is None or event.matched_iou < threshold or event.duplicate)
            ),
            "missed": 0,
        },
    }


def eligible_gt(gt_events: list[GroundTruthEvent]) -> list[GroundTruthEvent]:
    return [gt for gt in gt_events if gt.eligible]


def prefixed_metrics(metrics: dict[str, float], old_prefix: str, new_prefix: str) -> dict[str, float]:
    return {
        key.replace(old_prefix, new_prefix, 1): value
        for key, value in metrics.items()
        if key.startswith(old_prefix)
    }


def module_a_stage_metrics(
    module_a_predictions: list[tuple[WindowSpec, str, str, float]] | None,
    *,
    label_mode: str = "step_id",
) -> dict[str, float]:
    if not module_a_predictions:
        return {}
    targets = [window.label for window, _prediction, _raw, _elapsed in module_a_predictions]
    predictions = [prediction for _window, prediction, _raw, _elapsed in module_a_predictions]
    if label_mode == "step_id":
        metrics = multiclass_letter_metrics(targets, predictions, prefix="module_a_step_id")
        metrics.update(
            binary_detection_metrics(
                targets,
                predictions,
                prefix="module_a",
                label_mode="step_id",
            )
        )
        metrics["module_a/num_complete_predictions"] = float(
            sum(is_module_a_completion_label(prediction, label_mode="step_id") for prediction in predictions)
        )
        probabilities: list[dict[str, float]] = []
        scored_targets: list[str] = []
        for target, (_window, _prediction, raw, _elapsed) in zip(
            targets,
            module_a_predictions,
        ):
            try:
                payload = json.loads(raw)
                candidate_nlls = payload["candidate_nlls"]
                if not isinstance(candidate_nlls, dict):
                    continue
                probabilities.append(
                    scores_to_probabilities(
                        {
                            str(label): float(value)
                            for label, value in candidate_nlls.items()
                        }
                    )
                )
                scored_targets.append(
                    (
                        "A"
                        if target.strip().upper() in {"NONE", "A", "NOT_COMPLETED"}
                        else extract_module_a_label(target, label_mode="step_id")
                    )
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
        metrics["module_a_step_id/score_coverage"] = (
            len(probabilities) / len(module_a_predictions)
            if module_a_predictions
            else 0.0
        )
        if probabilities and len(probabilities) == len(module_a_predictions):
            diagnostics = label_score_diagnostics(probabilities, scored_targets)
            metrics.update(
                {
                    key.replace("eval_detection/", "module_a_detection/")
                    .replace("eval_step_id/", "module_a_step_id/"): value
                    for key, value in diagnostics.items()
                }
            )
    else:
        metrics = binary_detection_metrics(targets, predictions, prefix="module_a", label_mode="legacy")
        metrics["module_a/num_complete_predictions"] = float(
            sum(prediction == "COMPLETE" for prediction in predictions)
        )
    metrics["module_a/num_windows"] = float(len(module_a_predictions))
    metrics["module_a/num_no_trigger"] = float(
        sum(
            not is_module_a_completion_label(prediction, label_mode=label_mode)
            for _window, prediction, _raw, _elapsed in module_a_predictions
        )
    )
    return metrics


def module_b_stage_metrics(events: list[PipelineEvent]) -> dict[str, float]:
    metrics: dict[str, float] = {
        "module_b_given_module_a_trigger/num_triggers": float(len(events)),
        "module_b_given_module_a_trigger/num_valid_windows": float(
            sum(1 for event in events if event.pred_global_start is not None and event.pred_global_end is not None)
        ),
        "module_b_given_module_a_trigger/invalid_rate": safe_div(
            sum(1 for event in events if event.pred_global_start is None or event.pred_global_end is None),
            len(events),
        ),
        "module_b_given_module_a_trigger/clamped_rate": safe_div(
            sum(event.module_b_was_clamped for event in events),
            len(events),
        ),
    }
    evaluable = [
        event for event in events
        if event.matched_gt_start is not None and event.matched_gt_end is not None
    ]
    if not evaluable:
        return metrics
    targets = [(float(event.matched_gt_start), float(event.matched_gt_end)) for event in evaluable]
    predictions = [
        TemporalPrediction(
            start=event.pred_global_start,
            end=event.pred_global_end,
            rejection_reason=event.module_b_rejection_reason,
            raw_start=event.pred_raw_global_start,
            raw_end=event.pred_raw_global_end,
            was_clamped=event.module_b_was_clamped,
            clamp_reason=event.module_b_clamp_reason,
        )
        for event in evaluable
    ]
    for view in ("raw", "clamped"):
        temporal = temporal_metrics(
            targets,
            predictions,
            prefix=f"eval_temporal/{view}",
            view=view,
        )
        metrics.update(prefixed_metrics(
            temporal,
            "eval_temporal",
            "module_b_given_module_a_trigger",
        ))
        if view == "clamped":
            metrics.update(prefixed_metrics(
                temporal,
                "eval_temporal/clamped",
                "module_b_given_module_a_trigger",
            ))
    metrics["module_b_given_module_a_trigger/out_of_bounds_rate"] = metrics[
        "module_b_given_module_a_trigger/clamped_rate"
    ]
    return metrics


def compute_metrics(
    events: list[PipelineEvent],
    gt_events: list[GroundTruthEvent],
    module_a_predictions: list[tuple[WindowSpec, str, str, float]] | None = None,
    *,
    label_mode: str = "step_id",
    dataset: str = "egoper",
) -> dict[str, Any]:
    eligible = eligible_gt(gt_events)
    matched = [event for event in events if event.matched_gt_index is not None]
    ious = [event.matched_iou for event in matched]
    metrics: dict[str, Any] = {
        "num_gt_events": len(eligible),
        "num_gt_mistakes": sum(1 for gt in eligible if gt.gt_mistake),
        "num_gt_correct": sum(1 for gt in eligible if not gt.gt_mistake),
        "num_ineligible_gt_events": sum(1 for gt in gt_events if not gt.eligible),
        "num_predictions": len(events),
        "num_predictions_with_window": sum(1 for event in events if event.pred_global_start is not None),
        "num_matched_predictions": len(matched),
        "num_duplicates": sum(1 for event in events if event.duplicate),
        "duplicate_detection_rate": safe_div(sum(1 for event in events if event.duplicate), len(events)),
        "module_c_invalid_json_rate": safe_div(
            sum(1 for event in events if event.pred_global_start is not None and event.module_c_mistake is None),
            sum(1 for event in events if event.pred_global_start is not None),
        ),
        "mean_iou": sum(ious) / len(ious) if ious else 0.0,
        "stage/A_NO_TRIGGER": float(
            sum(
                not is_module_a_completion_label(prediction, label_mode=label_mode)
                for _window, prediction, _raw, _elapsed in (module_a_predictions or [])
            )
        ),
        "stage/B_NOT_COMPLETED": float(sum(event.stage_status == "B_NOT_COMPLETED" for event in events)),
        "stage/C_NOT_COMPLETED": float(sum(event.stage_status == "C_NOT_COMPLETED" for event in events)),
        "stage/C_SKIPPED_EXTRA": float(sum(event.stage_status == "C_SKIPPED_EXTRA" for event in events)),
        "stage/C_CLASSIFIED": float(sum(event.stage_status == "C_CLASSIFIED" for event in events)),
        "stage/C_INVALID_PARSE": float(sum(event.stage_status == "C_INVALID_PARSE" for event in events)),
    }
    metrics.update(module_a_stage_metrics(module_a_predictions, label_mode=label_mode))
    metrics.update(module_b_stage_metrics(events))
    fine_classes = EGOPER_FINE_CLASSES
    aggregate_mapping = EGOPER_AGGREGATE_MAPPING
    standalone_classes = EGOPER_MODULE_C_CLASSES
    classified_events = [
        event
        for event in matched
        if event.gt_fine_class in standalone_classes
        and event.stage_status not in {"C_SKIPPED_EXTRA", "B_NOT_COMPLETED", "C_NOT_COMPLETED"}
    ]
    if classified_events:
        classification = classification_metrics(
            [str(event.gt_fine_class) for event in classified_events],
            [event.module_c_classification for event in classified_events],
            fine_classes=standalone_classes,
            aggregate_mapping=aggregate_mapping,
        )
        metrics["module_c_fine"] = classification["fine"]
        metrics["module_c_aggregate"] = classification["aggregate"]
        metrics["module_c_not_completed"] = classification["not_completed"]
        metrics["deprecated_module_c_binary"] = classification[
            "deprecated_binary"
        ]
    end_to_end_events = [
        event
        for event in matched
        if event.gt_fine_class in aggregate_mapping
    ]
    if end_to_end_events:
        end_to_end = classification_metrics(
            [str(event.gt_fine_class) for event in end_to_end_events],
            [event.module_c_classification for event in end_to_end_events],
            fine_classes=fine_classes,
            aggregate_mapping=aggregate_mapping,
        )
        metrics["end_to_end_fine"] = end_to_end["fine"]
        metrics["end_to_end_aggregate"] = end_to_end["aggregate"]
    boundary_events = [
        event for event in matched
        if event.pred_global_start is not None
        and event.pred_global_end is not None
        and event.matched_gt_start is not None
        and event.matched_gt_end is not None
    ]
    if boundary_events:
        metrics["boundary_mae/start_seconds"] = sum(
            abs(float(event.pred_global_start) - float(event.matched_gt_start)) for event in boundary_events
        ) / len(boundary_events)
        metrics["boundary_mae/end_seconds"] = sum(
            abs(float(event.pred_global_end) - float(event.matched_gt_end)) for event in boundary_events
        ) / len(boundary_events)
        metrics["boundary_mae/center_seconds"] = sum(
            abs(
                ((float(event.pred_global_start) + float(event.pred_global_end)) / 2.0)
                - ((float(event.matched_gt_start) + float(event.matched_gt_end)) / 2.0)
            )
            for event in boundary_events
        ) / len(boundary_events)
        metrics["boundary_mae/duration_seconds"] = sum(
            abs(
                (float(event.pred_global_end) - float(event.pred_global_start))
                - (float(event.matched_gt_end) - float(event.matched_gt_start))
            )
            for event in boundary_events
        ) / len(boundary_events)
    else:
        metrics["boundary_mae/start_seconds"] = 0.0
        metrics["boundary_mae/end_seconds"] = 0.0
        metrics["boundary_mae/center_seconds"] = 0.0
        metrics["boundary_mae/duration_seconds"] = 0.0
    reasoning_pairs = [
        (str(event.gt_reasoning or ""), str(event.module_c_reasoning or ""))
        for event in matched
        if event.module_c_reasoning
    ]
    metrics["reasoning/non_empty_rate"] = safe_div(
        sum(1 for event in matched if event.module_c_reasoning),
        len(matched),
    )
    metrics["reasoning/lexical_overlap"] = (
        sum(lexical_overlap(reference, prediction) for reference, prediction in reasoning_pairs) / len(reasoning_pairs)
        if reasoning_pairs
        else 0.0
    )
    latencies = [
        event.trigger_time - float(event.matched_gt_end)
        for event in matched
        if event.matched_gt_end is not None
    ]
    metrics.update(latency_summary(latencies, prefix="latency"))
    metrics["latency_mean_seconds"] = metrics["latency/mean_seconds"]
    metrics["early_trigger_rate"] = metrics["latency/early_rate"]
    metrics.update(duration_summary(
        [event.module_a_prediction_seconds for event in events if event.module_a_prediction_seconds is not None],
        prefix="prediction_runtime/module_a",
    ))
    metrics.update(duration_summary(
        [event.module_b_prediction_seconds for event in events if event.module_b_prediction_seconds is not None],
        prefix="prediction_runtime/module_b",
    ))
    metrics.update(duration_summary(
        [event.module_c_prediction_seconds for event in events if event.module_c_prediction_seconds is not None],
        prefix="prediction_runtime/module_c",
    ))
    metrics.update(duration_summary(
        [event.pipeline_prediction_seconds for event in events if event.pipeline_prediction_seconds is not None],
        prefix="prediction_runtime/pipeline_completed_action",
    ))
    metrics.update(duration_summary(
        [
            event.monolith_prediction_seconds
            for event in events
            if event.monolith_prediction_seconds is not None
        ],
        prefix="prediction_runtime/monolith",
    ))

    for threshold in IOU_THRESHOLDS:
        suffix = f"{threshold:.1f}"
        matched_gt_indices = {
            event.matched_gt_index
            for event in matched
            if event.matched_gt_index is not None and event.matched_iou >= threshold
        }
        temporal_tp = len(matched_gt_indices)
        temporal_fp = sum(
            1 for event in events
            if event.pred_global_start is not None
            and (event.matched_gt_index is None or event.matched_iou < threshold or event.duplicate)
        )
        temporal_fn = len(eligible) - temporal_tp
        temporal_precision = safe_div(temporal_tp, temporal_tp + temporal_fp)
        temporal_recall = safe_div(temporal_tp, temporal_tp + temporal_fn)
        metrics[f"temporal/precision_at_iou_{suffix}"] = temporal_precision
        metrics[f"temporal/recall_at_iou_{suffix}"] = temporal_recall
        metrics[f"temporal/f1_at_iou_{suffix}"] = safe_div(
            2 * temporal_precision * temporal_recall,
            temporal_precision + temporal_recall,
        )
        metrics[f"recall_at_iou_{suffix}"] = temporal_recall
        metrics[f"missed_step_rate_at_iou_{suffix}"] = safe_div(temporal_fn, len(eligible))
        threshold_latencies = [
            event.trigger_time - float(event.matched_gt_end)
            for event in matched
            if event.matched_iou >= threshold and event.matched_gt_end is not None
        ]
        metrics.update(latency_summary(threshold_latencies, prefix=f"latency_at_iou_{suffix}"))

        conditional_events = [
            event for event in matched
            if event.matched_iou >= threshold and event.gt_mistake is not None
        ]
        c_tp = sum(1 for event in conditional_events if event.gt_mistake and event.module_c_mistake is True)
        c_fp = sum(1 for event in conditional_events if not event.gt_mistake and event.module_c_mistake is True)
        c_tn = sum(1 for event in conditional_events if not event.gt_mistake and event.module_c_mistake is not True)
        c_fn = sum(1 for event in conditional_events if event.gt_mistake and event.module_c_mistake is not True)
        c_stats = binary_stats(c_tp, c_fp, c_tn, c_fn)
        for key, value in c_stats.items():
            metrics[f"module_c_given_match_iou_{suffix}/{key}"] = value
        metrics[f"mistake_precision_given_match_iou_{suffix}"] = c_stats["precision"]
        metrics[f"mistake_recall_given_match_iou_{suffix}"] = c_stats["recall"]
        metrics[f"mistake_f1_given_match_iou_{suffix}"] = c_stats["f1"]
        metrics[f"module_c_given_match_iou_{suffix}/confusion_matrix"] = confusion_matrix(c_tp, c_fp, c_tn, c_fn)

        e_tp = sum(
            1 for event in events
            if event.gt_mistake is True and event.matched_iou >= threshold and event.module_c_mistake is True
        )
        e_fp = sum(
            1 for event in events
            if event.module_c_mistake is True
            and (
                event.matched_gt_index is None
                or event.matched_iou < threshold
                or event.gt_mistake is not True
                or event.duplicate
            )
        )
        detected_mistake_gt = {
            event.matched_gt_index
            for event in events
            if event.matched_gt_index is not None
            and event.matched_iou >= threshold
            and event.gt_mistake is True
            and event.module_c_mistake is True
        }
        total_gt_mistakes = sum(1 for gt in eligible if gt.gt_mistake)
        e_fn = total_gt_mistakes - len(detected_mistake_gt)
        total_gt_correct = sum(1 for gt in eligible if not gt.gt_mistake)
        e_tn = max(0, total_gt_correct - e_fp)
        e_stats = binary_stats(e_tp, e_fp, e_tn, e_fn)
        for key, value in e_stats.items():
            metrics[f"end_to_end_mistake_at_iou_{suffix}/{key}"] = value
        metrics[f"end_to_end_mistake_precision_at_iou_{suffix}"] = e_stats["precision"]
        metrics[f"end_to_end_mistake_recall_at_iou_{suffix}"] = e_stats["recall"]
        metrics[f"end_to_end_mistake_f1_at_iou_{suffix}"] = e_stats["f1"]
        metrics[f"end_to_end_mistake_accuracy_at_iou_{suffix}"] = e_stats["accuracy"]
        metrics[f"end_to_end_mistake_specificity_at_iou_{suffix}"] = e_stats["specificity"]
        metrics[f"end_to_end_mistake_balanced_accuracy_at_iou_{suffix}"] = e_stats["balanced_accuracy"]
        metrics[f"end_to_end_mistake_mcc_at_iou_{suffix}"] = e_stats["mcc"]
        metrics[f"end_to_end_mistake_false_positive_rate_at_iou_{suffix}"] = e_stats["false_positive_rate"]
        metrics[f"end_to_end_mistake_false_negative_rate_at_iou_{suffix}"] = e_stats["false_negative_rate"]
        metrics[f"end_to_end_mistake_at_iou_{suffix}/confusion_matrix"] = end_to_end_confusion_matrix(
            events=events,
            gt_events=gt_events,
            threshold=threshold,
        )
    latest_alerts: dict[str, dict[str, StructuralAlert]] = {}
    for event_index, event in enumerate(events):
        for alert_index, row in enumerate(event.graph_alerts):
            alert = StructuralAlert(
                kind=str(row.get("kind") or ""),
                step_index=int(row.get("step_index", -1)),
                candidates=tuple(int(value) for value in row.get("candidates") or ()),
                status=str(row.get("status") or ""),
                reason=str(row.get("reason") or ""),
                trigger_time=float(row.get("trigger_time", event.trigger_time)),
                occurrence_id=str(row.get("occurrence_id") or ""),
                alert_id=str(row.get("alert_id") or ""),
            )
            key = alert.alert_id or (
                f"{event.video_id}:{event_index}:{alert_index}:"
                f"{alert.kind}:{alert.step_index}"
            )
            latest_alerts.setdefault(event.video_id, {})[key] = alert
    target_order_by_video: dict[str, set[int]] = {}
    target_missing_by_video: dict[str, set[int]] = {}
    for event in gt_events:
        if "ORDER_ERROR" in event.gt_structural_labels:
            target_order_by_video.setdefault(event.video_id, set()).add(event.step_index)
        if "MISSING_STEP" in event.gt_structural_labels:
            target_missing_by_video.setdefault(event.video_id, set()).add(event.step_index)
    structural_video_ids = sorted(
        set(latest_alerts) | set(target_order_by_video) | set(target_missing_by_video)
    )
    if structural_video_ids:
        per_video = [
            structural_metrics(
                predicted_alerts=list(latest_alerts.get(video_id, {}).values()),
                target_order_steps=target_order_by_video.get(video_id, set()),
                target_missing_steps=target_missing_by_video.get(video_id, set()),
                prefix="structural",
            )
            for video_id in structural_video_ids
        ]
        for metric_name in (
            "order_precision",
            "order_recall",
            "order_f1",
            "confirmed_omission_recall",
        ):
            metrics[f"structural/{metric_name}"] = sum(
                row[f"structural/{metric_name}"] for row in per_video
            ) / len(per_video)
        for metric_name in (
            "provisional_alert_count",
            "predicted_order_steps",
            "predicted_missing_steps",
        ):
            metrics[f"structural/{metric_name}"] = sum(
                row[f"structural/{metric_name}"] for row in per_video
            )
        metrics["structural/num_videos"] = float(len(structural_video_ids))
    else:
        metrics.update(
            {
                "structural/order_precision": 0.0,
                "structural/order_recall": 0.0,
                "structural/order_f1": 0.0,
                "structural/confirmed_omission_recall": 0.0,
                "structural/provisional_alert_count": 0.0,
                "structural/predicted_order_steps": 0.0,
                "structural/predicted_missing_steps": 0.0,
                "structural/num_videos": 0.0,
            }
        )
    emitted_alerts = [
        alert for event in events for alert in (event.graph_alerts or [])
    ]
    metrics.update(
        {
            "structural/revision_event_count": float(
                sum(event.structural_revision is not None for event in events)
            ),
            "structural/retracted_event_count": float(
                sum(event.structural_revision == "retracted" for event in events)
            ),
            "structural/alert_emission_count": float(len(emitted_alerts)),
            "structural/retracted_alert_count": float(
                sum(alert.get("status") == "retracted" for alert in emitted_alerts)
            ),
            "structural/confirmed_alert_count": float(
                sum(alert.get("status") == "confirmed" for alert in emitted_alerts)
            ),
        }
    )
    from ..metrics.egoper import egoper_standard_metrics

    standard = egoper_standard_metrics(events, gt_events)
    metrics["egoper_standard"] = standard
    if standard is not None:
        metrics["egoper/tas_f1_at_0.5"] = standard["tas"]["f1_at_0.5"]
        metrics["egoper/mistake_binary_f1_at_0.5"] = standard[
            "mistake_binary"
        ]["error_f1_at_0.5"]
        metrics["egoper/mistake_fine_f1_at_0.5"] = standard[
            "mistake_fine"
        ]["macro_f1_at_0.5"]
        metrics["egoper/eda"] = standard["eda"]["eda"]
        metrics["egoper/eda_majority"] = standard["eda"]["majority"]["eda"]
        metrics["egoper/framewise_auc"] = standard["framewise_auc"]["auc"]
        segment_ap = standard["segment_ap"]
        metrics["egoper/binary_ap_mean_0.1_0.3"] = segment_ap[
            "binary_ap_mean_0.1_0.3"
        ]
        metrics["egoper/fine_map_mean_0.1_0.3"] = segment_ap[
            "fine_map_mean_0.1_0.3"
        ]
        metrics["egoper/binary_ap_score_coverage"] = segment_ap[
            "binary_score_coverage"
        ]
        metrics["egoper/fine_map_score_coverage"] = segment_ap[
            "fine_score_coverage"
        ]
        for threshold, values in segment_ap["by_threshold"].items():
            metrics[f"egoper/binary_ap_at_{threshold}"] = values["binary_ap"]
            metrics[f"egoper/fine_map_at_{threshold}"] = values["fine_map"]
            for class_name, value in values["per_class_ap"].items():
                metrics[
                    f"egoper/fine_ap_{class_name.lower()}_at_{threshold}"
                ] = value
        metrics["egoper/o_acc"] = standard["omission"]["o_acc"]
        metrics["egoper/o_iou"] = standard["omission"]["o_iou"]
    return metrics


def event_row(event: PipelineEvent) -> dict[str, Any]:
    row = asdict(event)
    if event.module_b_prediction is not None:
        row["module_b_prediction"] = list(event.module_b_prediction)
    return json_safe(row)


def module_a_prediction_row(
    prediction: tuple[WindowSpec, str, str, float],
    *,
    entered_module_b_keys: set[tuple[str, float, float]],
    label_mode: str,
    online_reset_from_b: bool,
) -> dict[str, Any]:
    window, parsed_prediction, raw_output, prediction_seconds = prediction
    key = (
        window.video_id,
        round(window.window_start, 6),
        round(window.window_end, 6),
    )
    is_trigger = is_module_a_completion_label(
        parsed_prediction,
        label_mode=label_mode,
    )
    entered_module_b = key in entered_module_b_keys
    return {
        "window": json_safe(asdict(window)),
        "prediction": parsed_prediction,
        "raw_output": raw_output,
        "prediction_seconds": prediction_seconds,
        "is_trigger_prediction": is_trigger,
        "entered_module_b": entered_module_b,
        "skipped_by_online_b_reset": (
            online_reset_from_b and is_trigger and not entered_module_b
        ),
    }


def checkpoint_provenance(path_value: str | None) -> dict[str, Any] | None:
    if not path_value:
        return None
    path = Path(path_value).expanduser()
    provenance: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
    }
    if not path.is_dir():
        return provenance
    provenance["files"] = {
        child.name: child.stat().st_size
        for child in sorted(path.iterdir())
        if child.is_file()
    }
    for name in ("adapter_config.json", "trainer_state.json"):
        candidate = path / name
        if not candidate.is_file():
            continue
        try:
            provenance[name.removesuffix(".json")] = json.loads(
                candidate.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            provenance[name.removesuffix(".json")] = None
    return provenance


def write_json_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(rows, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def write_artifact_views(
    output_dir: Path,
    *,
    event_rows: list[dict[str, Any]],
    gt_rows: list[dict[str, Any]],
    module_a_rows: list[dict[str, Any]],
    output_json: Path,
    output_jsonl: Path,
    output_csv: Path,
    payload: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True, ensure_ascii=False)
        file.write("\n")
    write_json_rows(output_dir / "pipeline_events.json", event_rows)
    write_jsonl_rows(output_jsonl, event_rows)
    write_json_rows(output_dir / "pipeline_ground_truth.json", gt_rows)
    write_jsonl_rows(output_dir / "pipeline_ground_truth.jsonl", gt_rows)
    write_json_rows(output_dir / "module_a_predictions.json", module_a_rows)
    write_jsonl_rows(output_dir / "module_a_predictions.jsonl", module_a_rows)
    if event_rows:
        fieldnames = sorted({key for row in event_rows for key in row})
        with open(output_csv, "w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(event_rows)
    print(f"Saved self-contained pipeline artifact to: {output_json}")
    print(f"Saved event JSONL to: {output_jsonl}")
    print(f"Saved ground truth to: {output_dir / 'pipeline_ground_truth.json'}")
    print(f"Saved Module A predictions to: {output_dir / 'module_a_predictions.jsonl'}")
    if event_rows:
        print(f"Saved event CSV to: {output_csv}")


def write_outputs(
    args: argparse.Namespace,
    events: list[PipelineEvent],
    metrics: dict[str, Any],
    split: dict[str, Any],
    *,
    gt_events: list[GroundTruthEvent],
    module_a_predictions: list[tuple[WindowSpec, str, str, float]],
    module_a_rows: list[dict[str, Any]] | None = None,
    video_shard: dict[str, Any] | None = None,
) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    default_output_name = "pipeline_artifacts.json" if args.defer_metrics else "pipeline_metrics.json"
    output_json = Path(args.output_json) if args.output_json else output_dir / default_output_name
    output_jsonl = Path(args.output_jsonl) if args.output_jsonl else output_dir / "pipeline_events.jsonl"
    output_csv = Path(args.output_csv) if args.output_csv else output_dir / "pipeline_events.csv"
    event_rows = [event_row(event) for event in events]
    gt_rows = [json_safe(asdict(event)) for event in gt_events]
    entered_module_b_keys = {
        (
            event.video_id,
            round(event.accumulation_start, 6),
            round(event.accumulation_end, 6),
        )
        for event in events
    }
    if module_a_rows is None:
        module_a_rows = [
            module_a_prediction_row(
                prediction,
                entered_module_b_keys=entered_module_b_keys,
                label_mode=args.module_a_label_mode,
                online_reset_from_b=args.online_reset_from_b,
            )
            for prediction in module_a_predictions
        ]
    payload = {
        "schema_version": "online_pipeline_artifacts_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "args": json_safe(vars(args)),
        "split": json_safe(split),
        "checkpoint_provenance": {
            "module_a": checkpoint_provenance(args.module_a_checkpoint),
            "module_b": checkpoint_provenance(args.module_b_checkpoint),
            "module_c": checkpoint_provenance(args.module_c_checkpoint),
        },
        "metrics": json_safe(metrics),
        "events": event_rows,
        "ground_truth": gt_rows,
        "module_a_predictions": module_a_rows,
        "artifact_counts": {
            "pipeline_events": len(event_rows),
            "ground_truth_events": len(gt_rows),
            "eligible_ground_truth_events": sum(event.eligible for event in gt_events),
            "module_a_predictions": len(module_a_rows),
        },
        "video_shard": video_shard
        or {
            "merged": False,
            "rank": int(getattr(args, "video_shard_rank", 0)),
            "world_size": int(getattr(args, "video_shard_world_size", 1)),
            "video_ids": sorted({event.video_id for event in gt_events} | {event.video_id for event in events}),
        },
    }
    write_artifact_views(
        output_dir,
        event_rows=event_rows,
        gt_rows=gt_rows,
        module_a_rows=module_a_rows,
        output_json=output_json,
        output_jsonl=output_jsonl,
        output_csv=output_csv,
        payload=payload,
    )


def try_merge_if_all_done(output_dir: str | Path, world_size: int) -> Path | None:
    root = Path(output_dir)
    completed = completed_shard_dirs(root, world_size)
    if len(completed) < world_size:
        print(
            f"Waiting for remaining shards before merge: {len(completed)}/{world_size} done.",
            flush=True,
        )
        return None
    if not try_acquire_merge_lock(root):
        print("Another rank is merging shard artifacts.", flush=True)
        return None
    return write_merged_shard_directory(root)


def claim_remaining_videos(queue_dir: str | Path, *, rank: int) -> list[str]:
    claimed: list[str] = []
    while True:
        video_id = claim_next_video(queue_dir, rank=rank)
        if video_id is None:
            return claimed
        claimed.append(video_id)


def write_merged_if_complete(output_dir: str | Path) -> Path | None:
    root = Path(output_dir)
    queue_dir = root / "work_queue"
    checkpoint_root = root / "videos"
    manifest = load_run_manifest(queue_dir)
    eval_split = str(manifest.get("eval_split") or "val")
    eval_ids = [str(video_id) for video_id in manifest.get("eval_video_ids") or []]
    if not eval_ids:
        print(f"No run manifest under {queue_dir}; nothing to merge yet.", flush=True)
        return None
    done_a = list_module_a_done(checkpoint_root)
    done_pipeline = list_pipeline_done(checkpoint_root)
    progress = write_progress(
        queue_dir,
        eval_split=eval_split,
        val_video_ids=eval_ids,
        module_a_done=done_a,
        pipeline_done=done_pipeline,
    )
    print(json.dumps(progress, indent=2, sort_keys=True), flush=True)
    if not progress["complete"]:
        print(
            f"Pipeline incomplete: {progress['pipeline_done']}/{progress['eval_videos']} "
            f"{eval_split} videos done. Resubmit the same sbatch to continue.",
            flush=True,
        )
        return None
    if not try_acquire_merge_lock(root):
        print("Another rank is merging completed video checkpoints.", flush=True)
        return None
    event_rows = [row for video_id in eval_ids for row in load_event_rows(checkpoint_root, video_id)]
    gt_rows = [row for video_id in eval_ids for row in load_ground_truth_rows(checkpoint_root, video_id)]
    module_a_rows = [row for video_id in eval_ids for row in load_module_a_rows(checkpoint_root, video_id)]
    event_rows.sort(key=lambda row: (row.get("video_id"), row.get("trigger_time"), row.get("step_index")))
    payload = {
        "schema_version": "online_pipeline_artifacts_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "args": {"eval_split": eval_split, "output_dir": str(root)},
        "split": manifest.get("split") or {},
        "checkpoint_provenance": {},
        "metrics": {},
        "events": event_rows,
        "ground_truth": gt_rows,
        "module_a_predictions": module_a_rows,
        "artifact_counts": {
            "pipeline_events": len(event_rows),
            "ground_truth_events": len(gt_rows),
            "eligible_ground_truth_events": sum(1 for row in gt_rows if row.get("eligible")),
            "module_a_predictions": len(module_a_rows),
        },
        "video_shard": {
            "merged": True,
            "mode": "queue",
            "eval_split": eval_split,
            "video_ids": eval_ids,
        },
    }
    output_json = root / "pipeline_artifacts.json"
    write_artifact_views(
        root,
        event_rows=event_rows,
        gt_rows=gt_rows,
        module_a_rows=module_a_rows,
        output_json=output_json,
        output_jsonl=root / "pipeline_events.jsonl",
        output_csv=root / "pipeline_events.csv",
        payload=payload,
    )
    print(f"Merged completed {eval_split} checkpoints into {output_json}", flush=True)
    return output_json


def write_merged_shard_directory(output_dir: str | Path) -> Path:
    root = Path(output_dir)
    shard_paths = discover_shard_artifacts(root)
    payloads = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in shard_paths
    ]
    payload = merge_pipeline_artifacts(payloads, shard_paths=shard_paths)
    event_rows = list(payload["events"])
    gt_rows = list(payload["ground_truth"])
    module_a_rows = list(payload["module_a_predictions"])
    default_name = (
        "pipeline_artifacts.json"
        if not payload.get("metrics")
        else "pipeline_metrics.json"
    )
    output_json = root / default_name
    write_artifact_views(
        root,
        event_rows=event_rows,
        gt_rows=gt_rows,
        module_a_rows=module_a_rows,
        output_json=output_json,
        output_jsonl=root / "pipeline_events.jsonl",
        output_csv=root / "pipeline_events.csv",
        payload=payload,
    )
    print(
        f"Merged {len(shard_paths)} video shards into {output_json} "
        f"({payload['artifact_counts']['module_a_predictions']} Module A polls, "
        f"{payload['artifact_counts']['pipeline_events']} events).",
        flush=True,
    )
    return output_json


def apply_graph_stage(
    events: list[PipelineEvent],
    records: list[VideoRecord],
) -> None:
    graphs = {
        record.video_id: record.task_graph
        for record in records
        if record.task_graph is not None
    }
    if not graphs:
        return
    pipeline = DependencyGraphPipeline()
    processed_by_video: dict[str, list[PipelineEvent]] = {}
    for event in events:
        graph = graphs.get(event.video_id)
        if graph is None:
            continue
        if (
            event.pred_global_start is None
            or event.pred_global_end is None
            or event.stage_status in {
                "B_NOT_COMPLETED",
                "C_NOT_COMPLETED",
                "C_SKIPPED_EXTRA",
                "B_INVALID_PARSE",
            }
        ):
            continue
        detected_step = predicted_step_index(event)
        if detected_step is None or detected_step < 0:
            continue
        node = graph.node_for_instruction(detected_step)
        if node not in graph.recipe_nodes:
            continue
        tracker = pipeline.tracker(event.video_id, graph)
        updates = tracker.update(
            DetectedOccurrence(
                occurrence_id=event.occurrence_id
                or f"{event.video_id}:predicted:{detected_step}",
                instruction_index=detected_step,
                start=event.pred_global_start,
                end=event.pred_global_end,
                trigger_time=event.trigger_time,
                localized=event.pred_global_start is not None,
            )
        )
        event.graph_alerts = [
            {
                "kind": alert.kind,
                "step_index": alert.step_index,
                "candidates": list(alert.candidates),
                "status": alert.status,
                "reason": alert.reason,
                "trigger_time": alert.trigger_time,
                "occurrence_id": alert.occurrence_id,
                "alert_id": alert.alert_id,
            }
            for alert in updates
        ]
        event.fused_labels = tracker.fused_labels(event.module_c_classifications or (
            (event.module_c_classification,) if event.module_c_classification else ()
        ))
        if any(alert.status == "retracted" for alert in updates):
            event.structural_revision = "retracted"
        processed_by_video.setdefault(event.video_id, []).append(event)

    for video_id, video_events in processed_by_video.items():
        graph = graphs.get(video_id)
        if graph is None or not video_events:
            continue
        last_event = video_events[-1]
        updates = pipeline.tracker(video_id, graph).finalize(
            trigger_time=last_event.trigger_time
        )
        last_event.graph_alerts.extend(
            {
                "kind": alert.kind,
                "step_index": alert.step_index,
                "candidates": list(alert.candidates),
                "status": alert.status,
                "reason": alert.reason,
                "trigger_time": alert.trigger_time,
                "occurrence_id": alert.occurrence_id,
                "alert_id": alert.alert_id,
            }
            for alert in updates
        )
        last_event.fused_labels = pipeline.tracker(video_id, graph).fused_labels(
            last_event.module_c_classifications
            or (
                (last_event.module_c_classification,)
                if last_event.module_c_classification
                else ()
            )
        )
        if any(alert.status == "retracted" for alert in updates):
            last_event.structural_revision = "retracted"


def main() -> None:
    args = parse_args()
    if args.merge_if_complete:
        write_merged_if_complete(args.merge_if_complete)
        return
    if args.merge_shards_dir:
        write_merged_shard_directory(args.merge_shards_dir)
        return
    if args.zero_shot_eval:
        print(
            f"Zero-shot pipeline: loading base model {args.model_name} for "
            "Modules A, B, and C (no LoRA checkpoints).",
            flush=True,
        )
    provider = make_provider(args)
    config = module_a_config(args)
    dataset = build_module_a_training_dataset(
        provider,
        config=config,
        max_videos=args.max_videos,
    )
    split, split_path = load_or_create_video_split(dataset, args)
    eval_ids = eval_video_ids_from_split(split, args.eval_split)
    all_eval_windows = [window for window in dataset.windows if window.video_id in eval_ids]
    window_counts = window_counts_by_video(all_eval_windows)
    for video_id in eval_ids:
        window_counts.setdefault(video_id, 0)
    run_output_dir = Path(args.output_dir)
    if args.video_queue_dir:
        install_stop_handler()
        args.video_checkpoint_dir = args.video_checkpoint_dir or str(run_output_dir / "videos")
        args.queue_session_id = args.queue_session_id or os.environ.get("SLURM_JOB_ID") or "manual"
        write_run_manifest(
            args.video_queue_dir,
            split=split,
            eval_split=args.eval_split,
            eval_video_ids=eval_ids,
        )
        done_a = set(list_module_a_done(args.video_checkpoint_dir))
        done_pipeline = set(list_pipeline_done(args.video_checkpoint_dir))
        refresh_remaining_videos(
            args.video_queue_dir,
            eval_ids - done_a,
            window_counts=window_counts,
            session_id=f"{args.queue_session_id}:a",
            remaining_name="remaining.json",
        )
        refresh_remaining_videos(
            args.video_queue_dir,
            done_a - done_pipeline,
            window_counts=window_counts,
            session_id=f"{args.queue_session_id}:bc",
            remaining_name="bc_remaining.json",
        )
        progress = write_progress(
            args.video_queue_dir,
            eval_split=args.eval_split,
            val_video_ids=eval_ids,
            module_a_done=done_a,
            pipeline_done=done_pipeline,
        )
        print(
            f"Resume {args.eval_split}: {progress['pipeline_done']}/{progress['eval_videos']} "
            f"videos complete, {len(progress['remaining_a'])} need A, "
            f"{len(progress['remaining_bc'])} need B/C",
            flush=True,
        )
        shard_ids: set[str] = set()
        windows = val_windows(dataset, eval_ids, -1)
    else:
        shard_ids = set(
            shard_video_ids(
                eval_ids,
                rank=args.video_shard_rank,
                world_size=args.video_shard_world_size,
                window_counts=window_counts,
            )
        )
        windows = val_windows(dataset, shard_ids, args.max_samples)
    if args.video_shard_world_size > 1:
        args.output_dir = str(
            shard_output_dir(run_output_dir, args.video_shard_rank, args.video_shard_world_size)
        )
    print(f"Online pipeline split: {split_path}")
    print(f"Eval split: {args.eval_split} ({len(eval_ids)} videos)")
    print(f"Module A label mode: {args.module_a_label_mode}")
    if args.video_queue_dir:
        print(
            f"Video queue worker {args.video_shard_rank}/{args.video_shard_world_size} "
            f"pulls from {args.video_queue_dir}"
        )
    else:
        print(
            f"Video shard {args.video_shard_rank}/{args.video_shard_world_size}: "
            f"{len(shard_ids)} videos, {sum(window_counts[video_id] for video_id in shard_ids)} "
            f"windows before max_samples"
        )
        print(f"Validation windows: {len(windows)}")
    module_a_predictions: list[tuple[WindowSpec, str, str, float]] = []
    causal_poll_rows: list[dict[str, Any]] | None = None
    claimed_video_ids: list[str] = []
    if args.skip_inference:
        events: list[PipelineEvent] = []
        if args.video_queue_dir:
            claimed_video_ids = claim_remaining_videos(
                args.video_queue_dir, rank=args.video_shard_rank
            )
            shard_ids = set(claimed_video_ids)
    elif args.reuse_events_json:
        reuse_path = Path(args.reuse_events_json)
        print(
            f"Reusing events from {reuse_path}; "
            f"{'scoring saved Module C outputs' if args.module_c_score_only else 're-running Module C'} "
            f"(max_frames_c={args.max_frames_c}, vision_resize_c={args.vision_resize_c})."
        )
        if args.video_queue_dir:
            claimed_video_ids = claim_remaining_videos(
                args.video_queue_dir, rank=args.video_shard_rank
            )
            shard_ids = set(claimed_video_ids)
        events = [
            event
            for event in load_events_for_module_c_reuse(
                reuse_path,
                clear_module_c=not args.module_c_score_only,
            )
            if event.video_id in shard_ids
        ]
        if args.max_samples >= 0:
            events = events[: args.max_samples]
        windows = val_windows(dataset, shard_ids, args.max_samples)
        module_a_predictions = reconstruct_module_a_predictions(windows=windows, events=events)
        windows_by_key = {(window.video_id, window.step_index): window for window in windows}
        video_paths = attach_video_paths(events, windows_by_key, records=dataset.records)
        missing = sum(
            1
            for index, event in enumerate(events)
            if event.pred_global_start is not None and index not in video_paths
        )
        if missing:
            print(f"Warning: {missing} events missing video paths; Module C will skip them.")
        run_module_c(args, events, video_paths)
        apply_graph_stage(events, dataset.records)
    elif args.causal_stream:
        module_a_predictions, events, causal_poll_rows = run_causal_ab(
            args,
            dataset=dataset,
            video_ids=[] if args.video_queue_dir else sorted(shard_ids),
            claimed_video_ids=claimed_video_ids,
        )
        shard_ids = set(claimed_video_ids)
        windows = [window for window, _prediction, _raw, _elapsed in module_a_predictions]
        if args.causal_module_c_in_loop:
            print(f"Causal A+B+C events: {len(events)}")
        elif args.video_checkpoint_dir and args.video_queue_dir:
            events = run_causal_c_from_queue(
                args,
                dataset=dataset,
                checkpoint_root=args.video_checkpoint_dir,
            )
            shard_ids.update(event.video_id for event in events)
        else:
            run_module_c(args, events, attach_video_paths(events, {}, records=dataset.records))
            apply_graph_stage(events, dataset.records)
    else:
        module_a_predictions = run_module_a(
            args,
            windows,
            dataset,
            claimed_video_ids=claimed_video_ids if args.video_queue_dir else None,
        )
        if args.video_queue_dir:
            shard_ids = set(claimed_video_ids)
            windows = [window for window, _prediction, _raw, _elapsed in module_a_predictions]
        if args.video_checkpoint_dir and args.video_queue_dir:
            events = run_bc_from_queue(
                args,
                dataset=dataset,
                checkpoint_root=args.video_checkpoint_dir,
            )
            shard_ids.update(event.video_id for event in events)
        elif args.module_a_label_mode == "step_id" and args.online_reset_from_b:
            triggers, events = select_triggers_with_online_b_reset(args, module_a_predictions)
            print(f"Module A triggers (online B-reset): {len(triggers)}")
            windows_by_key = {
                (window.video_id, window.step_index): window
                for window, _pred, _raw, _elapsed in triggers
            }
            run_module_c(args, events, attach_video_paths(events, windows_by_key, records=dataset.records))
            apply_graph_stage(events, dataset.records)
        else:
            triggers = [
                item for item in module_a_predictions
                if is_module_a_completion_label(item[1], label_mode=args.module_a_label_mode)
            ]
            print(f"Module A triggers: {len(triggers)}")
            events = run_module_b(args, triggers)
            windows_by_key = {
                (window.video_id, window.step_index): window
                for window, _pred, _raw, _elapsed in triggers
            }
            run_module_c(args, events, attach_video_paths(events, windows_by_key, records=dataset.records))
            apply_graph_stage(events, dataset.records)
    if args.skip_inference and args.video_queue_dir:
        windows = val_windows(dataset, shard_ids, args.max_samples)
    gt_events = gt_events_for_records(dataset.records, shard_ids, label_mode=args.module_a_label_mode)
    print(f"Claimed videos: {len(shard_ids)}")
    print(f"Validation windows: {len(windows)}")
    print(f"Eligible GT events: {len(eligible_gt(gt_events))}")
    print(f"GT mistakes: {sum(1 for gt in eligible_gt(gt_events) if gt.gt_mistake)}")
    if args.defer_metrics:
        metrics: dict[str, Any] = {}
        print(
            "Metric calculation deferred; saving unmatched predictions and complete "
            "offline-evaluation inputs.",
            flush=True,
        )
    else:
        match_events(events, gt_events)
        metrics = compute_metrics(
            events,
            gt_events,
            module_a_predictions=module_a_predictions,
            label_mode=args.module_a_label_mode,
            dataset=args.dataset,
        )
        print(json.dumps(json_safe(metrics), indent=2, sort_keys=True))
    write_outputs(
        args,
        events,
        metrics,
        split,
        gt_events=gt_events,
        module_a_predictions=module_a_predictions,
        module_a_rows=causal_poll_rows,
        video_shard={
            "merged": False,
            "mode": "queue" if args.video_queue_dir else "static",
            "rank": args.video_shard_rank,
            "world_size": args.video_shard_world_size,
            "video_ids": sorted(shard_ids),
            "window_count": len(windows),
            "ground_truth_count": len(gt_events),
            "queue_dir": args.video_queue_dir,
            "eval_split": args.eval_split,
        },
    )
    if args.video_checkpoint_dir:
        mark_shard_done(args.output_dir)
        write_merged_if_complete(run_output_dir)
    elif args.video_shard_world_size > 1:
        mark_shard_done(args.output_dir)
        try_merge_if_all_done(run_output_dir, args.video_shard_world_size)


if __name__ == "__main__":
    main()
