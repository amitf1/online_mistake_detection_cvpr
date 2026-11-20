from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterable, Sequence

from ..config import ModuleAConfig
from ..models.video import video_duration_seconds
from ..prompts.module_a import format_completion_set_target, step_id_for_instruction_index
from ..schemas import Segment, TrainingExample, VideoRecord, WindowSpec
from ..text import normalize_instruction_text


NEGATIVE_LABELS = {"NONE", "A", "WAIT"}


def build_module_a_windows(
    records: Iterable[VideoRecord],
    *,
    config: ModuleAConfig | None = None,
) -> list[WindowSpec]:
    """Build accumulated windows in observed event order."""

    cfg = config or ModuleAConfig()
    if cfg.label_mode == "completion_set":
        return build_module_a_completion_set_windows(records, config=cfg)
    windows: list[WindowSpec] = []
    rng = random.Random(cfg.seed)
    for record in records:
        video_duration = video_duration_seconds(record.video_path)
        if video_duration <= 0:
            video_duration = max((segment.end for segment in record.segments), default=0.0)
        completed_steps: list[str] = []
        accumulation_start = 0.0
        events = sorted(
            (
                segment
                for segment in record.segments
                if segment.event_type != "BACKGROUND" and segment.duration >= 0.5
            ),
            key=lambda segment: (
                segment.end,
                segment.start,
                segment.annotation_position,
            ),
        )
        open_window_start = accumulation_start
        for segment in events:
            label = step_id_for_instruction_index(segment.instruction_index)
            # Two actions can end together. The second one belongs to the clip
            # that just reached that instant, not to a new clip that starts there.
            if segment.end <= accumulation_start + 1e-6:
                windows.append(
                    _window(
                        record,
                        segment,
                        open_window_start,
                        min(video_duration, max(accumulation_start, segment.end)),
                        label,
                        completed_steps,
                    )
                )
                if segment.instruction_index >= 0:
                    completed_steps.append(
                        normalize_instruction_text(segment.instruction)
                    )
                continue
            open_window_start = accumulation_start
            poll = min(video_duration, accumulation_start + cfg.stride_seconds)
            while poll < segment.end:
                windows.append(
                    _window(
                        record,
                        segment,
                        accumulation_start,
                        poll,
                        "NONE",
                        completed_steps,
                    )
                )
                poll = min(video_duration, poll + cfg.stride_seconds)
                if poll <= windows[-1].window_end:
                    break
            windows.append(
                _window(
                    record,
                    segment,
                    accumulation_start,
                    min(video_duration, max(segment.end, poll)),
                    label,
                    completed_steps,
                )
            )
            if segment.instruction_index >= 0:
                completed_steps.append(normalize_instruction_text(segment.instruction))
            accumulation_start = _next_reset(segment.end, cfg, rng)
    return windows


def _next_reset(boundary: float, cfg: ModuleAConfig, rng: random.Random) -> float:
    jitter = max(0.0, float(cfg.reset_jitter_seconds))
    if jitter <= 0:
        return boundary
    return max(0.0, boundary - rng.uniform(0.0, jitter))


def _window(
    record: VideoRecord,
    segment: Segment,
    start: float,
    end: float,
    label: str,
    completed_steps: list[str],
) -> WindowSpec:
    occurrence_id = segment.occurrence_id or (
        f"{record.video_id}:{segment.annotation_position}:{segment.start:.3f}:{segment.end:.3f}"
    )
    return WindowSpec(
        video_path=record.video_path,
        video_id=record.video_id,
        task_id=record.task_id,
        step_index=segment.instruction_index,
        current_step=normalize_instruction_text(segment.instruction),
        window_start=start,
        window_end=end,
        gt_start=segment.start,
        gt_end=segment.end,
        label=label,
        completed_steps=tuple(completed_steps),
        all_instructions=tuple(normalize_instruction_text(item) for item in record.instructions),
        label_mode="step_id",
        event_type=segment.event_type,
        fine_class=segment.fine_class,
        aggregate_class=segment.aggregate_class,
        observed_description=segment.observed_description,
        occurrence_id=occurrence_id,
        annotation_position=segment.annotation_position,
        completed_step_ids=() if label == "NONE" else (label,),
        fine_classes=segment.resolved_fine_classes,
        dataset_metadata={
            "dataset": record.dataset,
            "episode_id": f"{record.video_id}:{start:.6f}",
            **segment.dataset_metadata,
        },
    )


def build_module_a_completion_set_windows(
    records: Iterable[VideoRecord],
    *,
    config: ModuleAConfig | None = None,
) -> list[WindowSpec]:
    """Emit one causal poll window with every occurrence that ended since the last poll."""

    cfg = config or ModuleAConfig()
    windows: list[WindowSpec] = []
    for record in records:
        video_duration = video_duration_seconds(record.video_path)
        if video_duration <= 0:
            video_duration = max((segment.end for segment in record.segments), default=0.0)
        events = [
            segment
            for segment in record.segments
            if not segment.is_missing and segment.duration >= 0.5
        ]
        previous_poll = 0.0
        poll = min(video_duration, cfg.stride_seconds)
        completed_before: list[str] = []
        while True:
            completions = [
                segment
                for segment in events
                if previous_poll < segment.end <= poll
            ]
            step_ids = []
            occurrences = []
            for segment in completions:
                step_id = step_id_for_instruction_index(segment.instruction_index)
                if step_id not in step_ids:
                    step_ids.append(step_id)
                occurrences.append(
                    {
                        "occurrence_id": segment.occurrence_id or f"{record.video_id}:{segment.annotation_position}",
                        "annotation_position": segment.annotation_position,
                        "instruction_index": segment.instruction_index,
                        "instruction": normalize_instruction_text(segment.instruction),
                        "step_id": step_id,
                        "gt_start": segment.start,
                        "gt_end": segment.end,
                        "fine_class": segment.fine_class,
                        "fine_classes": list(segment.resolved_fine_classes),
                        "structural_labels": list(segment.structural_labels),
                        "event_type": segment.event_type,
                        "aggregate_class": segment.aggregate_class,
                        "observed_description": segment.observed_description,
                    }
                )
            window_start = max(0.0, poll - cfg.context_seconds)
            representative = completions[0] if completions else None
            windows.append(
                WindowSpec(
                    video_path=record.video_path,
                    video_id=record.video_id,
                    task_id=record.task_id,
                    step_index=(
                        representative.instruction_index if representative is not None else -1
                    ),
                    current_step=(
                        normalize_instruction_text(representative.instruction)
                        if representative is not None
                        else ""
                    ),
                    window_start=window_start,
                    window_end=poll,
                    gt_start=representative.start if representative is not None else None,
                    gt_end=representative.end if representative is not None else None,
                    label=format_completion_set_target(step_ids),
                    completed_steps=tuple(completed_before),
                    all_instructions=tuple(
                        normalize_instruction_text(item) for item in record.instructions
                    ),
                    label_mode="completion_set",
                    event_type=(
                        representative.event_type if representative is not None else "NORMAL"
                    ),
                    fine_class=representative.fine_class if representative is not None else None,
                    aggregate_class=(
                        representative.aggregate_class if representative is not None else None
                    ),
                    observed_description=(
                        representative.observed_description if representative is not None else ""
                    ),
                    occurrence_id=(
                        representative.occurrence_id if representative is not None else ""
                    ),
                    annotation_position=(
                        representative.annotation_position if representative is not None else -1
                    ),
                    completed_step_ids=tuple(step_ids),
                    fine_classes=(
                        representative.resolved_fine_classes if representative is not None else ()
                    ),
                    dataset_metadata={
                        "dataset": record.dataset,
                        "completed_occurrences": occurrences,
                        "collision_bucket": len(completions) > 1,
                        "episode_id": f"{record.video_id}:{window_start:.6f}",
                    },
                )
            )
            for segment in completions:
                if segment.instruction_index >= 0:
                    completed_before.append(normalize_instruction_text(segment.instruction))
            if poll >= video_duration:
                break
            previous_poll = poll
            next_poll = min(video_duration, poll + cfg.stride_seconds)
            if next_poll <= poll:
                break
            poll = next_poll
    return windows


def is_negative_label(label: str) -> bool:
    return str(label).strip().upper()[:8] in NEGATIVE_LABELS or str(label).strip().upper() == "NONE"


def select_spread_items(items: Sequence[object], limit: int) -> list[object]:
    if limit <= 0 or not items:
        return []
    if len(items) <= limit:
        return list(items)
    if limit == 1:
        return [items[-1]]
    chosen = {0, len(items) - 1}
    if limit >= 3:
        chosen.add(len(items) // 2)
    if len(chosen) < limit:
        for index in range(1, len(items) - 1):
            chosen.add(index)
            if len(chosen) >= limit:
                break
    return [items[index] for index in sorted(chosen)[:limit]]


def cap_negative_windows(
    windows: Sequence[WindowSpec],
    *,
    max_negatives_per_episode: int,
) -> tuple[list[WindowSpec], dict[str, float]]:
    groups: dict[tuple[str, float], list[WindowSpec]] = defaultdict(list)
    for window in windows:
        groups[(window.video_id, round(window.window_start, 6))].append(window)
    kept: list[WindowSpec] = []
    natural_negatives = 0
    sampled_negatives = 0
    affected = 0
    for group in groups.values():
        ordered = sorted(group, key=lambda item: item.window_end)
        negatives = [item for item in ordered if is_negative_label(item.label)]
        positives = [item for item in ordered if item not in negatives]
        natural_negatives += len(negatives)
        if len(negatives) > max_negatives_per_episode:
            affected += 1
        selected = select_spread_items(negatives, max_negatives_per_episode)
        sampled_negatives += len(selected)
        kept.extend(positives)
        kept.extend(selected)
    kept.sort(key=lambda item: (item.video_id, item.window_start, item.window_end, item.step_index))
    return kept, {
        "natural_negative_windows": float(natural_negatives),
        "sampled_negative_windows": float(sampled_negatives),
        "affected_episodes": float(affected),
        "max_negative_windows_per_episode": float(max_negatives_per_episode),
    }


def cap_negative_examples(
    examples: Sequence[TrainingExample],
    *,
    max_negatives_per_episode: int,
) -> tuple[list[TrainingExample], dict[str, float]]:
    groups: dict[tuple[str, float], list[TrainingExample]] = defaultdict(list)
    for example in examples:
        groups[(example.video_id, round(example.window_start, 6))].append(example)
    kept: list[TrainingExample] = []
    natural_negatives = 0
    sampled_negatives = 0
    affected = 0
    for group in groups.values():
        ordered = sorted(group, key=lambda item: item.window_end)
        negatives = [
            item
            for item in ordered
            if is_negative_label(str(item.label or item.target_text))
            or str((item.metadata or {}).get("semantic_label") or "") == "NONE"
        ]
        positives = [item for item in ordered if item not in negatives]
        natural_negatives += len(negatives)
        if len(negatives) > max_negatives_per_episode:
            affected += 1
        selected = select_spread_items(negatives, max_negatives_per_episode)
        sampled_negatives += len(selected)
        kept.extend(positives)
        kept.extend(selected)
    kept.sort(key=lambda item: (item.video_id, item.window_start, item.window_end, item.step_index))
    return kept, {
        "natural_negative_windows": float(natural_negatives),
        "sampled_negative_windows": float(sampled_negatives),
        "affected_episodes": float(affected),
        "max_negative_windows_per_episode": float(max_negatives_per_episode),
    }


def cap_negative_conversations(
    conversations: Sequence[dict],
    *,
    max_negatives_per_episode: int,
) -> tuple[list[dict], dict[str, float]]:
    groups: dict[tuple[str, float], list[int]] = defaultdict(list)
    for index, item in enumerate(conversations):
        groups[(str(item["video_id"]), round(float(item["window_start"]), 6))].append(index)
    keep: set[int] = set()
    natural_negatives = 0
    sampled_negatives = 0
    affected = 0
    for indices in groups.values():
        ordered = sorted(indices, key=lambda index: float(conversations[index]["window_end"]))
        negatives = [
            index
            for index in ordered
            if is_negative_label(str(conversations[index].get("target") or ""))
        ]
        positives = [index for index in ordered if index not in negatives]
        natural_negatives += len(negatives)
        if len(negatives) > max_negatives_per_episode:
            affected += 1
        selected = select_spread_items(negatives, max_negatives_per_episode)
        sampled_negatives += len(selected)
        keep.update(positives)
        keep.update(selected)
    kept = [conversations[index] for index in sorted(keep)]
    return kept, {
        "natural_negative_windows": float(natural_negatives),
        "sampled_negative_windows": float(sampled_negatives),
        "affected_episodes": float(affected),
        "max_negative_windows_per_episode": float(max_negatives_per_episode),
    }
