#!/usr/bin/env python3
"""Run realistic ABC-monolith streaming evaluation."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Mapping

from online_mistake_detection.datasets import EgoPERProvider
from online_mistake_detection.metrics.abc_monolith import (
    SequentialMonolithState,
    build_monolith_artifact,
    monolith_runtime_metrics,
    parse_monolith_prediction,
    scores_to_probabilities,
    step_id_to_module_a_label,
)
from online_mistake_detection.metrics.pipeline import evaluate_pipeline_events
from online_mistake_detection.models.video import video_duration_seconds
from online_mistake_detection.pipeline.runner import (
    RunControl,
    apply_graph_stage,
    gt_events_for_records,
    install_stop_handler,
    json_safe,
)
from online_mistake_detection.pipeline.shards import claim_next_video
from online_mistake_detection.pipeline.video_checkpoints import (
    list_checkpoint_videos,
    refresh_remaining_videos,
    write_progress,
    write_run_manifest,
)
from online_mistake_detection.prompts.abc_monolith import (
    build_monolith_prompt,
    monolith_dataset_label,
    prompt_version_for_contract,
    resolve_monolith_contract,
    system_prompt_for_dataset,
)
from online_mistake_detection.prompts.contracts import NOT_COMPLETED_TARGET
from online_mistake_detection.schemas import Segment, VideoRecord, WindowSpec
from online_mistake_detection.text import normalize_instruction_text
from online_mistake_detection.training.abc_monolith import (
    ABCMonolithMetricsCallback,
    Qwen3VLMetadataVisionDataCollator,
)
from online_mistake_detection.training.module_a import (
    load_generation_eval_model,
    release_cuda_memory,
)
from online_mistake_detection.windowing.abc_monolith import gold_history_for_poll

HISTORY_MODES = ("predicted", "none", "gold")
HISTORY_EVALUATION_VIEW = {
    "predicted": "realistic_predicted_history_sequential",
    "none": "sequential_no_history",
    "gold": "sequential_gold_causal_history",
}
WINDOW_HISTORY_SOURCE = {
    "predicted": "predicted_prior_eligible_actions",
    "none": "no_prior_actions",
    "gold": "gold_prior_eligible_actions",
}

GenerateFn = Callable[[dict[str, Any]], str]
ScoreFn = Callable[
    [dict[str, Any], Any],
    tuple[Mapping[str, float], Mapping[str, float], float],
]


def _duration(record: VideoRecord) -> float:
    duration = video_duration_seconds(record.video_path)
    if duration <= 0:
        duration = max((segment.end for segment in record.segments), default=0.0)
    return float(duration)


def _eligible_segments(record: VideoRecord) -> list[Segment]:
    return sorted(
        (
            segment
            for segment in record.segments
            if segment.event_type != "BACKGROUND" and segment.duration >= 0.5
        ),
        key=lambda item: (item.end, item.start, item.annotation_position),
    )


def _target_for_window(
    record: VideoRecord,
    *,
    window_start: float,
    window_end: float,
) -> Segment | None:
    return next(
        (
            segment
            for segment in _eligible_segments(record)
            if segment.end > window_start + 1e-9
            and segment.end <= window_end + 1e-9
        ),
        None,
    )


def _window(
    record: VideoRecord,
    *,
    window_start: float,
    window_end: float,
    target: Segment | None,
    history_mode: str = "predicted",
) -> WindowSpec:
    label = (
        "NONE"
        if target is None
        else "B"
        if target.instruction_index < 0
        else chr(ord("C") + target.instruction_index)
    )
    return WindowSpec(
        video_path=record.video_path,
        video_id=record.video_id,
        task_id=record.task_id,
        step_index=-1 if target is None else target.instruction_index,
        current_step=(
            "" if target is None else normalize_instruction_text(target.instruction)
        ),
        window_start=window_start,
        window_end=window_end,
        gt_start=None if target is None else target.start,
        gt_end=None if target is None else target.end,
        label=label,
        all_instructions=tuple(
            normalize_instruction_text(item) for item in record.instructions
        ),
        label_mode="step_id",
        event_type="NORMAL" if target is None else target.event_type,
        fine_class=None if target is None else target.fine_class,
        aggregate_class=None if target is None else target.aggregate_class,
        observed_description=(
            "" if target is None else target.observed_description
        ),
        occurrence_id="" if target is None else target.occurrence_id,
        annotation_position=-1 if target is None else target.annotation_position,
        completed_step_ids=() if target is None else (label,),
        fine_classes=() if target is None else target.resolved_fine_classes,
        dataset_metadata={
            "dataset": record.dataset,
            "history_source": WINDOW_HISTORY_SOURCE[history_mode],
        },
    )


def _conversation(
    record: VideoRecord,
    *,
    window_start: float,
    window_end: float,
    history: list[dict[str, Any]],
    history_source: str,
    max_history: int,
    fps: float,
    min_frames: int,
    max_frames: int,
    prompt_contract: str | None = None,
) -> dict[str, Any]:
    contract = resolve_monolith_contract(record.dataset, prompt_contract)
    prompt = build_monolith_prompt(
        task_name=record.task_id,
        task_description=record.task_description,
        instructions=record.instructions,
        clip_duration=window_end - window_start,
        history=history,
        max_history=max_history,
        history_source=history_source,
        dataset=record.dataset,
        prompt_contract=prompt_contract,
        display_dataset=(
            None
            if contract == record.dataset
            else monolith_dataset_label(record.dataset)
        ),
    )
    return {
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": system_prompt_for_dataset(
                            record.dataset,
                            prompt_contract=prompt_contract,
                        ),
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": str(record.video_path),
                        "video_start": window_start,
                        "video_end": window_end,
                        "fps": fps,
                        "min_frames": min_frames,
                        "max_frames": max_frames,
                    },
                    {"type": "text", "text": prompt},
                ],
            },
            # Candidate scoring replaces this placeholder. Generation removes it.
            {
                "role": "assistant",
                "content": [{"type": "text", "text": NOT_COMPLETED_TARGET}],
            },
        ],
        "target": NOT_COMPLETED_TARGET,
        "video_id": record.video_id,
        "task_id": record.task_id,
        "window_start": window_start,
        "window_end": window_end,
        "metadata": {
            "all_instructions": list(record.instructions),
            "dataset": record.dataset,
            "data_dataset": record.dataset,
            "prompt_contract": contract,
        },
        "prompt_text": prompt,
    }


class EvaluationInterrupted(RuntimeError):
    """SIGTERM arrived between polls. The in-flight video is left uncheckpointed."""


def _prompt_history(
    record: VideoRecord,
    state: SequentialMonolithState,
    *,
    history_mode: str,
    window_start: float,
    window_end: float,
    target: Segment | None,
) -> tuple[list[dict[str, Any]], str, float | None, str | None]:
    """Return prompt history without changing the predicted cursor."""

    if history_mode == "predicted":
        return state.history, "predicted", None, None
    if history_mode == "none":
        return [], "none", None, None
    history, cutoff, kind = gold_history_for_poll(
        record,
        window_start=window_start,
        window_end=window_end,
        target=target,
        max_history=state.max_history,
    )
    return [dict(item) for item in history], "gold", cutoff, kind


def run_sequential_evaluation(
    records: list[VideoRecord],
    *,
    generate: GenerateFn,
    score: ScoreFn | None = None,
    stride_seconds: float = 5.0,
    fps: float = 1.0,
    min_frames: int = 2,
    max_frames: int = 54,
    checkpoint: str | None = None,
    split_name: str = "test",
    gpu_count: int = 1,
    video_shard: str = "single_process",
    compute_metrics: bool = True,
    history_mode: str = "predicted",
    history_max_events: int = 12,
    prompt_contract: str | None = None,
) -> dict[str, Any]:
    """Run one generation per chronological poll."""

    if history_mode not in HISTORY_MODES:
        raise ValueError(f"history_mode must be one of {HISTORY_MODES}")
    if stride_seconds != 5.0:
        raise ValueError("The monolith evaluation contract requires 5-second polls.")
    streaming_started = time.perf_counter()
    polls: list[dict[str, Any]] = []
    events = []
    module_a_rows: list[dict[str, Any]] = []
    streamed_video_seconds = 0.0
    score_seconds_total = 0.0

    for record in sorted(records, key=lambda item: item.video_id):
        duration = _duration(record)
        if duration <= 0:
            continue
        streamed_video_seconds += duration
        state = SequentialMonolithState(max_history=history_max_events)
        poll_end = min(duration, stride_seconds)
        while poll_end > 0:
            if RunControl.stop_requested:
                raise EvaluationInterrupted(record.video_id)
            window_start = state.accumulation_start
            target = _target_for_window(
                record, window_start=window_start, window_end=poll_end
            )
            history, prompt_history_source, cutoff, cutoff_kind = (
                _prompt_history(
                    record,
                    state,
                    history_mode=history_mode,
                    window_start=window_start,
                    window_end=poll_end,
                    target=target,
                )
            )
            example = _conversation(
                record,
                window_start=window_start,
                window_end=poll_end,
                history=history,
                history_source=prompt_history_source,
                max_history=history_max_events,
                fps=fps,
                min_frames=min_frames,
                max_frames=max_frames,
                prompt_contract=prompt_contract,
            )
            started = time.perf_counter()
            raw = generate(example)
            prediction_seconds = time.perf_counter() - started
            parsed = parse_monolith_prediction(
                raw,
                clip_duration=poll_end - window_start,
                dataset=record.dataset,
                prompt_contract=prompt_contract,
            )
            step_nlls: Mapping[str, float] = {}
            class_nlls: Mapping[str, float] = {}
            score_seconds = 0.0
            if score is not None:
                step_nlls, class_nlls, score_seconds = score(example, parsed)
                score_seconds_total += score_seconds
            step_probabilities = (
                scores_to_probabilities(step_nlls) if step_nlls else None
            )
            class_probabilities = (
                scores_to_probabilities(class_nlls) if class_nlls else None
            )
            instruction = "off-script action"
            if parsed.step_id and parsed.step_id.startswith("STEP_"):
                index = int(parsed.step_id.removeprefix("STEP_"))
                if 0 <= index < len(record.instructions):
                    instruction = normalize_instruction_text(record.instructions[index])
            event = state.consume(
                parsed,
                video_id=record.video_id,
                task_id=record.task_id,
                instruction=instruction,
                window_end=poll_end,
                prediction_seconds=prediction_seconds,
                raw_output=raw,
                class_probabilities=class_probabilities,
                candidate_nlls=class_nlls or None,
            )
            window = _window(
                record,
                window_start=window_start,
                window_end=poll_end,
                target=target,
                history_mode=history_mode,
            )
            module_a_rows.append(
                {
                    "window": json_safe(asdict(window)),
                    "prediction": (
                        step_id_to_module_a_label(parsed.step_id)
                        if parsed.valid
                        else "A"
                    ),
                    "raw_output": (
                        json.dumps(
                            {
                                "generated_text": raw,
                                "candidate_nlls": dict(step_nlls),
                            },
                            sort_keys=True,
                        )
                        if step_nlls
                        else raw
                    ),
                    "prediction_seconds": prediction_seconds,
                    "step_candidate_nlls": dict(step_nlls),
                    "step_probabilities": dict(step_probabilities or {}),
                    "extra_probability": (
                        None
                        if step_probabilities is None
                        else step_probabilities.get("B")
                    ),
                }
            )
            poll_row = state.polls[-1]
            poll_row.update(
                {
                    "video_id": record.video_id,
                    "task_id": record.task_id,
                    "trigger_time": poll_end,
                    "accumulation_start": window_start,
                    "accumulation_end": poll_end,
                    "ground_truth_occurrence_id": (
                        None if target is None else target.occurrence_id
                    ),
                    "raw_json": raw,
                    "parse_valid": parsed.valid,
                    "parse_error": parsed.error,
                    "completed": parsed.valid and parsed.completed,
                    "raw_local_start": (
                        (parsed.payload or {}).get("start")
                        if parsed.completed
                        else None
                    ),
                    "raw_local_end": (
                        (parsed.payload or {}).get("end")
                        if parsed.completed
                        else None
                    ),
                    "clamped_local_start": parsed.start,
                    "clamped_local_end": parsed.end,
                    "raw_global_start": (
                        None
                        if not (parsed.valid and parsed.completed)
                        else window_start
                        + float((parsed.payload or {})["start"])
                    ),
                    "raw_global_end": (
                        None
                        if not (parsed.valid and parsed.completed)
                        else window_start + float((parsed.payload or {})["end"])
                    ),
                    "global_start": (
                        None
                        if parsed.start is None
                        else window_start + parsed.start
                    ),
                    "global_end": (
                        None if parsed.end is None else window_start + parsed.end
                    ),
                    "prompt_text": example["prompt_text"],
                    "supplied_history": history,
                    "history_mode": history_mode,
                    "history_cutoff_seconds": cutoff,
                    "history_cutoff_kind": cutoff_kind,
                    "history_occurrence_ids": [
                        str(item["occurrence_id"])
                        for item in history
                        if item.get("occurrence_id")
                    ],
                    "monolith_prediction_seconds": prediction_seconds,
                    "score_pass_seconds": score_seconds,
                    "step_candidate_nlls": dict(step_nlls),
                    "step_probabilities": dict(step_probabilities or {}),
                    "extra_probability": (
                        None
                        if step_probabilities is None
                        else step_probabilities.get("B")
                    ),
                    "class_candidate_nlls": dict(class_nlls),
                    "class_probabilities": dict(class_probabilities or {}),
                    "non_correct_probability": (
                        None
                        if class_probabilities is None
                        else sum(
                            probability
                            for class_name, probability in class_probabilities.items()
                            if class_name != "CORRECT"
                        )
                    ),
                    "score_provenance": (
                        "candidate_nll_length_normalized" if score is not None else None
                    ),
                    "checkpoint": checkpoint,
                    "split": split_name,
                    "video_shard": video_shard,
                    "prompt_version": prompt_version_for_contract(
                        record.dataset,
                        prompt_contract,
                    ),
                    "parser_version": "abc_monolith_strict_v1",
                    "data_dataset": record.dataset,
                    "prompt_contract": resolve_monolith_contract(
                        record.dataset,
                        prompt_contract,
                    ),
                    "late_trigger": False,
                    "sampled_incomplete": False,
                    "reset_disposition": (
                        "predicted_global_end" if event is not None else "retained"
                    ),
                }
            )
            polls.append(dict(poll_row))
            if event is not None:
                events.append(event)
            if RunControl.stop_requested:
                raise EvaluationInterrupted(record.video_id)
            if poll_end >= duration - 1e-9:
                break
            poll_end = min(duration, poll_end + stride_seconds)

    streaming_wall_seconds = time.perf_counter() - streaming_started
    ground_truth = gt_events_for_records(
        records, {record.video_id for record in records}, label_mode="step_id"
    )
    apply_graph_stage(events, records)
    native_predictions = [
        (
            WindowSpec(
                **{
                    **row["window"],
                    "video_path": Path(row["window"]["video_path"]),
                    **{
                        key: tuple(row["window"].get(key) or ())
                        for key in (
                            "completed_steps",
                            "pending_steps",
                            "all_instructions",
                            "completed_step_ids",
                            "fine_classes",
                        )
                    },
                }
            ),
            row["prediction"],
            row["raw_output"],
            row["prediction_seconds"],
        )
        for row in module_a_rows
    ]
    data_dataset = records[0].dataset if records else "egoper"
    resolved_contract = resolve_monolith_contract(data_dataset, prompt_contract)
    native_contract = resolved_contract == data_dataset
    metrics: dict[str, Any] = {}
    if compute_metrics and native_contract:
        metrics = evaluate_pipeline_events(
            events,
            ground_truth,
            module_a_predictions=native_predictions,
            label_mode="step_id",
            dataset=data_dataset,
        )
    if compute_metrics:
        metrics.update(
            monolith_runtime_metrics(
                polls,
                streamed_video_seconds=streamed_video_seconds,
                gpu_count=gpu_count,
            )
        )
    metadata = {
        "dataset": data_dataset,
        "data_dataset": data_dataset,
        "prompt_contract": resolved_contract,
        "prompt_version": prompt_version_for_contract(data_dataset, prompt_contract),
        "native_taxonomy_metrics": native_contract,
        "evaluation_view": HISTORY_EVALUATION_VIEW[history_mode],
        "history_source": history_mode,
        "history_mode": history_mode,
        "streamed_video_seconds": streamed_video_seconds,
        "gpu_count": gpu_count,
        "checkpoint": checkpoint,
        "split": split_name,
        "candidate_scoring": score is not None,
        "metrics_deferred": not compute_metrics,
        "video_shard": video_shard,
        "score_pass_seconds": score_seconds_total,
        "one_pass_generation_seconds": sum(
            float(row["monolith_prediction_seconds"]) for row in polls
        ),
        "end_to_end_one_pass_seconds": max(
            0.0, streaming_wall_seconds - score_seconds_total
        ),
    }
    return build_monolith_artifact(
        polls=polls,
        events=events,
        ground_truth=ground_truth,
        metrics=metrics,
        metadata=metadata,
        module_a_predictions=module_a_rows,
        args={
            "stride_seconds": stride_seconds,
            "fps": fps,
            "min_frames": min_frames,
            "max_frames": max_frames,
            "gpu_count": gpu_count,
            "monolith_candidate_scoring": score is not None,
            "compute_metrics": compute_metrics,
            "history_mode": history_mode,
            "monolith_prompt_contract": prompt_contract,
        },
        split={"name": split_name, "video_ids": sorted(record.video_id for record in records)},
    )


def shard_output_path(path: Path, *, rank: int, world_size: int) -> Path:
    if world_size == 1:
        return path
    suffix = path.suffix or ".json"
    stem = path.name.removesuffix(suffix)
    return path.with_name(f"{stem}.rank-{rank:03d}-of-{world_size:03d}{suffix}")


def _matching_completed_shard(
    path: Path,
    *,
    checkpoint: str,
    split_name: str,
    video_ids: list[str],
    rank: int,
    world_size: int,
    metrics_deferred: bool,
) -> bool:
    if not path.exists():
        return False
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    metadata = artifact.get("metadata", {})
    return (
        artifact.get("schema_version") == "egoper_abc_monolith_v1"
        and metadata.get("checkpoint") == checkpoint
        and metadata.get("split") == split_name
        and int(metadata.get("video_shard_rank", -1)) == rank
        and int(metadata.get("video_shard_world_size", -1)) == world_size
        and bool(metadata.get("metrics_deferred", False)) == metrics_deferred
        and artifact.get("split", {}).get("video_ids") == video_ids
    )


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as writer:
            writer.write(
                json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n"
            )
        tmp_path.replace(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _load_split(path: Path, split_name: str) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    key = f"{split_name}_video_ids"
    values = payload.get(key)
    if not values:
        raise ValueError(f"Split file has no non-empty {key}.")
    return {str(value) for value in values}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("egoper",), default="egoper")
    parser.add_argument("--egoper-root", type=Path)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--split-file",
        type=Path,
        default=Path("configs/splits/egoper_video_split_seed3407.json"),
    )
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--model-name", default="unsloth/Qwen3.8-27B")
    parser.add_argument("--max-seq-length", type=int, default=20480)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--min-frames", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=54)
    parser.add_argument("--vision-resize", type=int, default=1024)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--load-in-8bit", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--load-in-16bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--monolith-candidate-scoring",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--gpu-count", type=int, default=1)
    parser.add_argument("--video-shard-rank", type=int)
    parser.add_argument("--video-shard-world-size", type=int)
    parser.add_argument(
        "--resume-completed-shards",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--defer-metrics",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--video-queue-dir", type=Path)
    parser.add_argument("--video-checkpoint-dir", type=Path)
    parser.add_argument("--queue-session-id")
    parser.add_argument(
        "--history-mode",
        choices=HISTORY_MODES,
        default="predicted",
    )
    parser.add_argument("--history-max-events", type=int, default=12)
    parser.add_argument(
        "--monolith-prompt-contract",
        choices=("egoper",),
        default=None,
    )
    args = parser.parse_args()
    if args.history_max_events < 0:
        parser.error("--history-max-events must be non-negative.")
    if args.monolith_prompt_contract is not None:
        try:
            resolve_monolith_contract(args.dataset, args.monolith_prompt_contract)
        except ValueError as exc:
            parser.error(str(exc))

    rank = (
        args.video_shard_rank
        if args.video_shard_rank is not None
        else int(os.environ.get("RANK", "0"))
    )
    world_size = (
        args.video_shard_world_size
        if args.video_shard_world_size is not None
        else int(os.environ.get("WORLD_SIZE", "1"))
    )
    if world_size < 1 or rank < 0 or rank >= world_size:
        raise ValueError(
            f"Invalid video shard rank/world size: rank={rank}, world_size={world_size}"
        )
    selected = _load_split(args.split_file, args.split)
    if args.egoper_root is None:
        parser.error("--egoper-root is required for EgoPER.")
    provider = EgoPERProvider(
        args.egoper_root, video_ids=selected, require_existing_videos=True
    )
    records = list(provider.iter_video_records())
    missing = selected - {record.video_id for record in records}
    if missing:
        raise ValueError(f"Split references unavailable videos: {sorted(missing)[:5]}")
    records_by_id = {record.video_id: record for record in records}
    eval_ids = sorted(records_by_id)
    queue_dir = args.video_queue_dir
    checkpoint_dir = args.video_checkpoint_dir
    if queue_dir is not None:
        checkpoint_dir = checkpoint_dir or queue_dir.parent / "videos"
        session_id = args.queue_session_id or os.environ.get("SLURM_JOB_ID") or "manual"
        write_run_manifest(
            queue_dir,
            split={"name": args.split, "video_ids": eval_ids},
            eval_split=args.split,
            eval_video_ids=eval_ids,
        )
        done = set(list_checkpoint_videos(checkpoint_dir, "monolith.json"))
        refresh_remaining_videos(
            queue_dir,
            [video_id for video_id in eval_ids if video_id not in done],
            session_id=f"{session_id}:monolith",
        )
        progress = write_progress(
            queue_dir,
            eval_split=args.split,
            val_video_ids=eval_ids,
            module_a_done=done,
            pipeline_done=done,
        )
        print(
            json.dumps(
                {
                    "status": "queue_resume",
                    "done": progress["pipeline_done"],
                    "expected": progress["eval_videos"],
                    "remaining": progress["remaining_a"],
                },
                sort_keys=True,
            )
        )
        if progress["complete"]:
            return
        install_stop_handler()
        claimed_ids: list[str] = []
    else:
        records = sorted(records, key=lambda item: item.video_id)[rank::world_size]
        claimed_ids = [record.video_id for record in records]
    video_shard = f"rank_{rank:03d}_of_{world_size:03d}"
    output_json = shard_output_path(
        args.output_json, rank=rank, world_size=world_size
    )
    if (
        queue_dir is None
        and args.resume_completed_shards
        and _matching_completed_shard(
            output_json,
            checkpoint=args.checkpoint,
            split_name=args.split,
            video_ids=claimed_ids,
            rank=rank,
            world_size=world_size,
            metrics_deferred=args.defer_metrics,
        )
    ):
        print(
            json.dumps(
                {
                    "output_json": str(output_json),
                    "video_shard": video_shard,
                    "status": "already_complete",
                },
                sort_keys=True,
            )
        )
        return

    model_args = argparse.Namespace(
        eval_checkpoint=args.checkpoint,
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        load_in_16bit=args.load_in_16bit,
    )
    model, tokenizer = load_generation_eval_model(model_args)
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
    generator = ABCMonolithMetricsCallback(
        eval_examples=[],
        data_collator=collator,
        tokenizer=tokenizer,
        max_samples=-1,
        max_new_tokens=args.max_new_tokens,
        batch_size=1,
        use_cache=args.use_cache,
        enable_thinking=False,
        candidate_scoring=args.monolith_candidate_scoring,
    )

    def generate(example: dict[str, Any]) -> str:
        return generator._generate(model, [example])[0]

    def score(
        example: dict[str, Any], prediction: Any
    ) -> tuple[Mapping[str, float], Mapping[str, float], float]:
        started = time.perf_counter()
        step_targets, class_targets = generator.scoring_targets(example, prediction)
        step_nlls = (
            generator._score_targets(model, example, step_targets)
            if step_targets
            else {}
        )
        class_nlls = (
            generator._score_targets(model, example, class_targets)
            if class_targets
            else {}
        )
        return step_nlls, class_nlls, time.perf_counter() - started

    def evaluate_records(selected_records: list[VideoRecord], *, shard_name: str) -> dict[str, Any]:
        artifact = run_sequential_evaluation(
            selected_records,
            generate=generate,
            score=score if args.monolith_candidate_scoring else None,
            fps=args.fps,
            min_frames=args.min_frames,
            max_frames=args.max_frames,
            checkpoint=args.checkpoint,
            split_name=args.split,
            gpu_count=args.gpu_count,
            video_shard=shard_name,
            compute_metrics=not args.defer_metrics,
            history_mode=args.history_mode,
            history_max_events=args.history_max_events,
            prompt_contract=args.monolith_prompt_contract,
        )
        artifact["metadata"].update(
            {
                "video_shard_rank": rank,
                "video_shard_world_size": world_size,
                "queue_mode": queue_dir is not None,
            }
        )
        return artifact

    try:
        if queue_dir is None:
            artifact = evaluate_records(records, shard_name=video_shard)
            _atomic_write_json(output_json, artifact)
            print(
                json.dumps(
                    {
                        "output_json": str(output_json),
                        "video_shard": video_shard,
                        "polls": len(artifact["polls"]),
                        "events": len(artifact["events"]),
                        "metrics": artifact["metrics"],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            while True:
                if RunControl.stop_requested:
                    print("Stop requested; not claiming another video.", flush=True)
                    break
                video_id = claim_next_video(queue_dir, rank=rank)
                if video_id is None:
                    break
                record = records_by_id[video_id]
                print(f"Claimed video {video_id}", flush=True)
                try:
                    artifact = evaluate_records(
                        [record],
                        shard_name=f"queue_rank_{rank:03d}",
                    )
                except EvaluationInterrupted:
                    print(
                        f"Interrupted during {video_id}; "
                        "its partial polls were not checkpointed.",
                        flush=True,
                    )
                    break
                _atomic_write_json(checkpoint_dir / video_id / "monolith.json", artifact)
                done = set(list_checkpoint_videos(checkpoint_dir, "monolith.json"))
                write_progress(
                    queue_dir,
                    eval_split=args.split,
                    val_video_ids=eval_ids,
                    module_a_done=done,
                    pipeline_done=done,
                )
                print(
                    json.dumps(
                        {
                            "output_json": str(checkpoint_dir / video_id / "monolith.json"),
                            "video_id": video_id,
                            "polls": len(artifact["polls"]),
                            "events": len(artifact["events"]),
                            "done": len(done),
                            "expected": len(eval_ids),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    finally:
        del model
        release_cuda_memory()


if __name__ == "__main__":
    main()
