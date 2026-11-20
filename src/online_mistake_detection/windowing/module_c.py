from __future__ import annotations

import json
import random
from collections.abc import Iterable, Mapping, Sequence

from ..config import ModuleAConfig, ModuleCConfig
from ..prompts.contracts import NOT_COMPLETED_STATUS, NOT_COMPLETED_TARGET
from ..prompts.module_c import (
    append_module_c_history,
    build_module_c_prompt,
    build_reasoning,
    module_c_classes_for_dataset,
)
from ..schemas import AggregateClass, TrainingExample, VideoRecord
from .module_a import build_module_a_windows


def _action_history_lookup(
    record: VideoRecord,
    *,
    max_events: int,
    min_duration_seconds: float,
):
    """Return completed annotated actions strictly preceding a cutoff."""
    ordered = sorted(
        (
            segment
            for segment in record.segments
            if segment.event_type != "BACKGROUND"
            and segment.fine_class != "ADDITION"
            and segment.duration >= min_duration_seconds
        ),
        key=lambda segment: (
            segment.start,
            segment.end,
            segment.annotation_position,
            segment.occurrence_id,
        ),
    )

    def preceding(cutoff: float) -> list[dict[str, str]]:
        history: list[dict[str, str]] = []
        for segment in ordered:
            if segment.end > cutoff:
                continue
            classifications = segment.resolved_fine_classes
            classification = (
                ", ".join(classifications)
                if classifications
                else segment.fine_class
            )
            history.append(
                {
                    "action": segment.instruction,
                    "classification": classification,
                    "evidence": build_reasoning(
                        observed_description=(
                            segment.observed_description or segment.caption
                        ),
                        instruction=segment.instruction,
                        fine_class=segment.fine_class,
                    ),
                }
            )
        return history[-max_events:] if max_events > 0 else []

    return preceding


def build_module_c_examples(
    records: Iterable[VideoRecord],
    *,
    fine_taxonomy: Sequence[str],
    aggregate_mapping: Mapping[str, AggregateClass],
    config: ModuleCConfig | None = None,
    module_a_config: ModuleAConfig | None = None,
) -> list[TrainingExample]:
    cfg = config or ModuleCConfig()
    rng = random.Random(cfg.seed)
    records = list(records)
    examples: list[TrainingExample] = []
    for record in records:
        video_end = max((segment.end for segment in record.segments), default=0.0)
        allowed = module_c_classes_for_dataset(record.dataset, fine_taxonomy)
        history_max_events = cfg.history_max_events
        action_history = _action_history_lookup(
            record,
            max_events=history_max_events,
            min_duration_seconds=cfg.min_duration_seconds,
        )
        for segment in record.segments:
            if (
                segment.event_type == "BACKGROUND"
                or segment.duration < cfg.min_duration_seconds
            ):
                continue
            if segment.fine_class == "ADDITION":
                continue
            if segment.instruction_index < 0:
                continue
            if segment.fine_class not in aggregate_mapping:
                raise ValueError(
                    f"{record.video_id}: unmapped class {segment.fine_class!r}"
                )
            if allowed and segment.fine_class not in allowed:
                continue
            jitter_before = segment.duration * rng.uniform(
                cfg.jitter_ratio_min, cfg.jitter_ratio_max
            )
            jitter_after = segment.duration * rng.uniform(
                cfg.jitter_ratio_min, cfg.jitter_ratio_max
            )
            window_start = max(0.0, segment.start - jitter_before)
            window_end = min(video_end, segment.end + jitter_after)
            reasoning = build_reasoning(
                observed_description=segment.observed_description or segment.caption,
                instruction=segment.instruction,
                fine_class=segment.fine_class,
            )
            target = json.dumps(
                {
                    "classification": segment.fine_class,
                    "reasoning": reasoning,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            prompt = build_module_c_prompt(
                segment.instruction,
                allowed or fine_taxonomy,
                dataset_name=record.dataset,
            )
            history = action_history(segment.start)
            base_prompt = prompt
            prompt = append_module_c_history(
                base_prompt,
                history,
                max_events=history_max_events,
            )
            examples.append(
                TrainingExample(
                    module="C",
                    source_dataset=record.dataset,
                    video_path=record.video_path,
                    video_id=record.video_id,
                    task_id=record.task_id,
                    step_index=segment.instruction_index,
                    window_start=window_start,
                    window_end=window_end,
                    prompt_text=prompt,
                    target_text=target,
                    gt_start=segment.start,
                    gt_end=segment.end,
                    label=segment.fine_class,
                    mistake=segment.aggregate_class == "MISTAKE",
                    reasoning=reasoning,
                    event_type=segment.event_type,
                    fine_class=segment.fine_class,
                    aggregate_class=segment.aggregate_class,
                    observed_description=segment.observed_description,
                    occurrence_id=segment.occurrence_id,
                    fine_classes=segment.resolved_fine_classes,
                    metadata={
                        **segment.dataset_metadata,
                        "module_c_history": history,
                        "module_c_history_mode": (
                            "oracle" if history_max_events > 0 else "disabled"
                        ),
                        "module_c_prompt_without_history": base_prompt,
                        "multi_label": False,
                        "classifications": list(segment.resolved_fine_classes),
                        "descriptions_by_class": {
                            key: list(values)
                            for key, values in segment.descriptions_by_class.items()
                        },
                        "modified_description": segment.modified_description,
                        "structural_labels": list(segment.structural_labels),
                    },
                )
            )
    if cfg.include_not_completed_examples:
        examples.extend(
            build_module_c_not_completed_examples(
                records,
                positive_count=len(examples),
                fine_taxonomy=fine_taxonomy,
                config=cfg,
                module_a_config=module_a_config,
            )
        )
    return examples


def build_module_c_not_completed_examples(
    records: Sequence[VideoRecord],
    *,
    positive_count: int,
    fine_taxonomy: Sequence[str],
    config: ModuleCConfig,
    module_a_config: ModuleAConfig | None = None,
) -> list[TrainingExample]:
    windows = build_module_a_windows(records, config=module_a_config or ModuleAConfig())
    wait_windows = [
        window
        for window in windows
        if window.label in {"NONE", "A", "WAIT"}
        and window.step_index >= 0
    ]
    rng = random.Random(config.seed + 1)
    k = min(len(wait_windows), round(positive_count * max(0.0, config.not_completed_ratio)))
    selected = rng.sample(wait_windows, k) if k > 0 else []
    records_by_video = {record.video_id: record for record in records}
    history_by_video = {
        record.video_id: _action_history_lookup(
            record,
            max_events=config.history_max_events,
            min_duration_seconds=config.min_duration_seconds,
        )
        for record in records
    }
    examples: list[TrainingExample] = []
    for window in selected:
        record = records_by_video[window.video_id]
        history_max_events = config.history_max_events
        instruction = window.current_step or (
            record.instructions[window.step_index]
            if 0 <= window.step_index < len(record.instructions)
            else "unspecified step"
        )
        reasoning = build_reasoning(
            observed_description="",
            instruction=instruction,
            fine_class=NOT_COMPLETED_STATUS,
        )
        history = history_by_video[record.video_id](window.window_end)
        base_prompt = build_module_c_prompt(
            instruction,
            fine_taxonomy,
            dataset_name=record.dataset,
        )
        examples.append(
            TrainingExample(
                module="C",
                source_dataset=record.dataset,
                video_path=window.video_path,
                video_id=window.video_id,
                task_id=window.task_id,
                step_index=window.step_index,
                window_start=window.window_start,
                window_end=window.window_end,
                prompt_text=append_module_c_history(
                    base_prompt,
                    history,
                    max_events=history_max_events,
                ),
                target_text=NOT_COMPLETED_TARGET,
                gt_start=window.gt_start,
                gt_end=window.gt_end,
                label=NOT_COMPLETED_STATUS,
                mistake=False,
                reasoning=reasoning,
                event_type=window.event_type,
                fine_class=NOT_COMPLETED_STATUS,
                aggregate_class=None,
                observed_description=window.observed_description,
                occurrence_id=window.occurrence_id,
                metadata={
                    "not_completed": True,
                    "instruction": instruction,
                    "module_c_history": history,
                    "module_c_history_mode": (
                        "oracle" if history_max_events > 0 else "disabled"
                    ),
                    "module_c_prompt_without_history": base_prompt,
                },
            )
        )
    return examples
