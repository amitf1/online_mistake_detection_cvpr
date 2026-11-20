from __future__ import annotations

import random
from collections.abc import Iterable

from ..config import ModuleAConfig, ModuleBConfig
from ..models.video import video_duration_seconds
from ..prompts.contracts import NOT_COMPLETED_TARGET
from ..prompts.module_b import build_module_b_prompt, format_module_b_target
from ..schemas import TrainingExample, VideoRecord, WindowSpec
from .module_a import build_module_a_windows

def clamp_module_b_local_span(
    local_start: float,
    local_end: float,
    clip_duration: float,
) -> tuple[float, float]:
    """Clamp an annotated span to the visible Module A clip."""

    duration = float(clip_duration)
    start = max(0.0, float(local_start))
    end = min(duration, float(local_end))
    if start >= end:
        raise ValueError(
            f"Module A-anchored Module B target [{local_start}, {local_end}] "
            f"does not overlap [0, {duration}]"
        )
    return start, end


def build_module_b_examples(
    records: Iterable[VideoRecord],
    *,
    config: ModuleBConfig | None = None,
    module_a_config: ModuleAConfig | None = None,
) -> list[TrainingExample]:
    records = list(records)
    a_config = module_a_config or ModuleAConfig()
    all_windows = build_module_a_windows(records, config=a_config)
    completion_windows = [
        window
        for window in all_windows
        if (
            bool(window.completed_step_ids)
            if window.label_mode == "completion_set"
            else window.label != "NONE"
        )
    ]
    examples = build_module_b_examples_from_windows(
        records,
        completion_windows,
        config=config,
        stride_seconds=a_config.stride_seconds,
    )
    cfg = config or ModuleBConfig()
    if cfg.include_incomplete_negatives:
        examples.extend(
            build_module_b_incomplete_examples(
                records,
                all_windows,
                completion_windows,
                config=cfg,
            )
        )
    return examples


def build_module_b_examples_from_windows(
    records: Iterable[VideoRecord],
    completion_windows: Iterable[WindowSpec],
    *,
    config: ModuleBConfig | None = None,
    stride_seconds: float,
) -> list[TrainingExample]:
    """Build localization examples from clips Module A could actually emit."""
    cfg = config or ModuleBConfig()
    rng = random.Random(cfg.seed)
    records = list(records)
    records_by_video = {record.video_id: record for record in records}
    examples: list[TrainingExample] = []
    for window in completion_windows:
        record = records_by_video[window.video_id]
        video_end = video_duration_seconds(record.video_path)
        if video_end <= 0:
            video_end = max(
                (segment.end for segment in record.segments),
                default=window.window_end,
            )
        late_trigger_augmented = rng.random() < cfg.late_trigger_probability
        base_window_end = min(video_end, window.window_end)
        window_end = min(
            video_end,
            base_window_end + (stride_seconds if late_trigger_augmented else 0.0),
        )
        window_start = window.window_start
        clip_duration = window_end - window_start
        occurrences = list(window.dataset_metadata.get("completed_occurrences") or [])
        if not occurrences:
            if window.gt_start is None or window.gt_end is None:
                continue
            occurrences = [
                {
                    "occurrence_id": window.occurrence_id,
                    "annotation_position": window.annotation_position,
                    "instruction_index": window.step_index,
                    "instruction": window.current_step,
                    "gt_start": window.gt_start,
                    "gt_end": window.gt_end,
                    "fine_class": window.fine_class,
                    "fine_classes": list(window.fine_classes),
                    "event_type": window.event_type,
                    "aggregate_class": window.aggregate_class,
                    "observed_description": window.observed_description,
                }
            ]
        for occurrence in occurrences:
            gt_start = float(occurrence["gt_start"])
            gt_end = float(occurrence["gt_end"])
            try:
                local_start, local_end = clamp_module_b_local_span(
                    gt_start - window_start,
                    gt_end - window_start,
                    clip_duration,
                )
            except ValueError as exc:
                raise ValueError(f"{record.video_id}: {exc}") from exc
            instruction = str(occurrence.get("instruction") or window.current_step)
            examples.append(
                TrainingExample(
                    module="B",
                    source_dataset=record.dataset,
                    video_path=record.video_path,
                    video_id=record.video_id,
                    task_id=record.task_id,
                    step_index=int(occurrence.get("instruction_index", window.step_index)),
                    window_start=window_start,
                    window_end=window_end,
                    prompt_text=build_module_b_prompt(instruction, clip_duration),
                    target_text=format_module_b_target(local_start, local_end),
                    gt_start=gt_start,
                    gt_end=gt_end,
                    label="LOCALIZE",
                    event_type=occurrence.get("event_type", window.event_type),
                    fine_class=occurrence.get("fine_class", window.fine_class),
                    aggregate_class=occurrence.get("aggregate_class", window.aggregate_class),
                    observed_description=occurrence.get(
                        "observed_description", window.observed_description
                    ),
                    occurrence_id=str(occurrence.get("occurrence_id") or ""),
                    fine_classes=tuple(occurrence.get("fine_classes") or window.fine_classes),
                    metadata={
                        "instruction": instruction,
                        "clip_duration": clip_duration,
                        "module_a_window_start": window.window_start,
                        "module_a_window_end": window.window_end,
                        "late_trigger_augmented": late_trigger_augmented,
                        "late_trigger_stride_seconds": (
                            stride_seconds if late_trigger_augmented else 0.0
                        ),
                        "annotation_position": occurrence.get("annotation_position"),
                        "occurrence_id": occurrence.get("occurrence_id"),
                    },
                )
            )
    return examples


def build_module_b_incomplete_examples(
    records: Iterable[VideoRecord],
    module_a_windows: Iterable[WindowSpec],
    completion_windows: Iterable[WindowSpec],
    *,
    config: ModuleBConfig,
) -> list[TrainingExample]:
    records_by_video = {record.video_id: record for record in records}
    wait_windows = [
        window
        for window in module_a_windows
        if (
            window.label in {"NONE", "A", "WAIT"}
            if window.label_mode != "completion_set"
            else not window.completed_step_ids
        )
    ]
    complete = list(completion_windows)
    rng = random.Random(config.seed)
    k = min(len(wait_windows), round(len(complete) * max(0.0, config.negative_ratio)))
    selected = rng.sample(wait_windows, k) if k > 0 else []
    examples: list[TrainingExample] = []
    for window in selected:
        record = records_by_video[window.video_id]
        instruction = window.current_step or (
            record.instructions[window.step_index]
            if 0 <= window.step_index < len(record.instructions)
            else "unspecified step"
        )
        clip_duration = window.window_end - window.window_start
        if clip_duration <= 0:
            continue
        examples.append(
            TrainingExample(
                module="B",
                source_dataset=record.dataset,
                video_path=window.video_path,
                video_id=window.video_id,
                task_id=window.task_id,
                step_index=window.step_index,
                window_start=window.window_start,
                window_end=window.window_end,
                gt_start=window.gt_start,
                gt_end=window.gt_end,
                prompt_text=build_module_b_prompt(instruction, clip_duration),
                target_text=NOT_COMPLETED_TARGET,
                label="NO_ACTION",
                event_type=window.event_type,
                fine_class=window.fine_class,
                aggregate_class=window.aggregate_class,
                observed_description=window.observed_description,
                occurrence_id=window.occurrence_id,
                metadata={
                    "instruction": instruction,
                    "clip_duration": clip_duration,
                    "not_completed": True,
                },
            )
        )
    return examples
