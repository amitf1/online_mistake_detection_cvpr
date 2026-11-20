from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Iterable, Sequence
from dataclasses import replace

from ..config import ModuleAConfig, ModuleBConfig, ModuleCConfig
from ..datasets.training import build_module_b_training_dataset, build_module_c_training_dataset
from ..models.video import video_duration_seconds
from ..prompts.abc_monolith import build_monolith_prompt, format_monolith_target
from ..prompts.contracts import NOT_COMPLETED_TARGET
from ..prompts.module_a import step_id_for_instruction_index
from ..prompts.module_c import build_reasoning
from ..schemas import Segment, TrainingExample, VideoRecord, WindowSpec
from ..text import normalize_instruction_text
from .module_a import build_module_a_windows, cap_negative_examples
from .module_b import clamp_module_b_local_span


PIPELINE_MARKER = "abc_monolith"
NATURAL_NOT_COMPLETED = "natural_not_completed"
LATE_TRIGGER = "late_trigger_step_id"
SAMPLED_INCOMPLETE = "sampled_incomplete"
STEP_COMPLETION = "step_id_completion"
MISTAKE_CLASSES = {"SLIP", "MODIFICATION", "CORRECTION"}


def mistake_repeat_classes(
    dataset: str,
    *,
    prompt_contract: str | None = None,
) -> frozenset[str]:
    """Classes copied by the training repeat."""

    from ..prompts.abc_monolith import resolve_monolith_contract

    resolve_monolith_contract(dataset, prompt_contract)
    return frozenset(MISTAKE_CLASSES)


def _prompt_classification(
    fine_class: str,
    *,
    dataset: str = "egoper",
    prompt_contract: str | None = None,
    instruction_index: int = 0,
) -> str:
    return str(fine_class).upper()


def stable_monolith_row_id(
    *,
    family: str,
    video_id: str,
    window_start: float,
    window_end: float,
    occurrence_id: str = "",
    source_module: str = "A",
) -> str:
    identity = json.dumps(
        [
            PIPELINE_MARKER,
            family,
            source_module,
            video_id,
            round(float(window_start), 6),
            round(float(window_end), 6),
            occurrence_id,
        ],
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]


def build_abc_monolith_examples(
    records: Iterable[VideoRecord],
    *,
    training: bool = False,
    seed: int = 13,
    stride_seconds: float = 5.0,
    max_history: int = 12,
    history_dropout_probability: float = 0.25,
    max_natural_not_completed_per_episode: int = 2,
    late_trigger_probability: float = 0.10,
    mistake_repeat_factor: int = 4,
    sampled_incomplete_examples: Sequence[TrainingExample] = (),
    prompt_contract: str | None = None,
    not_completed_ratio: float | None = None,
) -> list[TrainingExample]:
    """Build the one-pass examples from exact-reset Module A polls."""
    records = list(records)
    if float(stride_seconds) != 5.0:
        raise ValueError("abc_monolith requires the approved 5-second stride")
    records_by_video = {record.video_id: record for record in records}
    windows = build_module_a_windows(
        records,
        config=ModuleAConfig(
            stride_seconds=stride_seconds,
            reset_jitter_seconds=0.0,
            seed=seed,
            label_mode="step_id",
        ),
    )
    rng = random.Random(seed)
    examples = [
        _example_from_window(
            records_by_video[window.video_id],
            window,
            family=(
                NATURAL_NOT_COMPLETED if window.label == "NONE" else STEP_COMPLETION
            ),
            history_dropped=(
                training and rng.random() < max(0.0, history_dropout_probability)
            ),
            max_history=max_history,
            prompt_contract=prompt_contract,
        )
        for window in windows
    ]

    if training:
        examples, _ = cap_negative_examples(
            examples,
            max_negatives_per_episode=max_natural_not_completed_per_episode,
        )
        positives = [item for item in examples if item.metadata["target_kind"] == "step_id"]
        late_rng = random.Random(seed + 1)
        late_examples = [
            _late_trigger_example(
                item,
                records_by_video[item.video_id],
                stride_seconds=stride_seconds,
            )
            for item in positives
            if late_rng.random() < max(0.0, late_trigger_probability)
        ]
        examples.extend(item for item in late_examples if item is not None)

    examples.extend(
        project_sampled_incomplete_examples(
            sampled_incomplete_examples,
            records_by_video=records_by_video,
            max_history=max_history,
            training=training,
            history_dropout_probability=history_dropout_probability,
            seed=seed + 2,
            prompt_contract=prompt_contract,
        )
    )
    if training and mistake_repeat_factor > 1:
        repeat_classes = mistake_repeat_classes(
            records[0].dataset if records else "egoper",
            prompt_contract=prompt_contract,
        )
        originals = [
            item for item in examples if item.fine_class in repeat_classes
        ]
        for copy_index in range(1, mistake_repeat_factor):
            examples.extend(
                replace(
                    item,
                    metadata={
                        **item.metadata,
                        "balance_repeat_index": copy_index,
                        "balance_repeat_factor": mistake_repeat_factor,
                    },
                )
                for item in originals
            )
    if training and not_completed_ratio is not None:
        examples = _limit_not_completed_examples(
            examples,
            ratio=float(not_completed_ratio),
        )
    return sorted(
        examples,
        key=lambda item: (
            item.video_id,
            item.window_start,
            item.window_end,
            str(item.metadata.get("cohort_family")),
            int(item.metadata.get("balance_repeat_index", 0)),
        ),
    )


def _limit_not_completed_examples(
    examples: list[TrainingExample],
    *,
    ratio: float,
) -> list[TrainingExample]:
    """Keep NOT_COMPLETED rows at ``ratio`` times the completed-row count."""

    if not 0.0 <= ratio <= 1.0:
        raise ValueError("not_completed_ratio must be in [0, 1]")
    negatives = [
        item
        for item in examples
        if item.fine_class == "NOT_COMPLETED" or item.label == "NOT_COMPLETED"
    ]
    negative_ids = {id(item) for item in negatives}
    others = [item for item in examples if id(item) not in negative_ids]
    limit = round(len(others) * ratio)
    if limit >= len(negatives):
        return examples
    ordered = sorted(
        negatives,
        key=lambda item: (item.video_id, item.window_start, item.window_end),
    )
    if limit <= 0:
        kept_ids: set[int] = set()
    elif limit == 1:
        kept_ids = {id(ordered[len(ordered) // 2])}
    else:
        indexes = {
            round(index * (len(ordered) - 1) / (limit - 1))
            for index in range(limit)
        }
        kept_ids = {id(ordered[index]) for index in indexes}
    return [item for item in examples if id(item) not in negative_ids or id(item) in kept_ids]


def _example_from_window(
    record: VideoRecord,
    window: WindowSpec,
    *,
    family: str,
    history_dropped: bool,
    max_history: int,
    prompt_contract: str | None = None,
) -> TrainingExample:
    completed = window.label != "NONE"
    target_segment = _target_segment(record, window) if completed else None
    cutoff = target_segment.start if target_segment is not None else window.window_end
    history = _gold_history(
        record,
        cutoff=cutoff,
        exclude_occurrence_id=(
            _occurrence_id(record, target_segment) if target_segment is not None else ""
        ),
        max_history=max_history,
        prompt_contract=prompt_contract,
    )
    prompt_history = None if history_dropped else history
    clip_duration = window.window_end - window.window_start
    prompt = build_monolith_prompt(
        task_name=record.task_id,
        task_description=record.task_description,
        instructions=record.instructions,
        clip_duration=clip_duration,
        history=prompt_history,
        max_history=max_history,
        dataset=record.dataset,
        prompt_contract=prompt_contract,
    )
    if target_segment is None:
        target = NOT_COMPLETED_TARGET
        label = "NOT_COMPLETED"
        reasoning = None
        fine_class = "NOT_COMPLETED"
        aggregate_class = None
        occurrence_id = window.occurrence_id
    else:
        step_id, classification = _step_and_classification(
            target_segment,
            dataset=record.dataset,
            prompt_contract=prompt_contract,
        )
        local_start, local_end = clamp_module_b_local_span(
            target_segment.start - window.window_start,
            target_segment.end - window.window_start,
            clip_duration,
        )
        reasoning = build_reasoning(
            observed_description=(
                target_segment.observed_description or target_segment.caption
            ),
            instruction=target_segment.instruction,
            fine_class=classification,
        )
        target = format_monolith_target(
            step_id=step_id,
            start=local_start,
            end=local_end,
            classification=classification,
            reasoning=reasoning,
            dataset=record.dataset,
            prompt_contract=prompt_contract,
        )
        label = step_id
        fine_class = classification
        aggregate_class = target_segment.aggregate_class
        occurrence_id = _occurrence_id(record, target_segment)
    row_id = stable_monolith_row_id(
        family=family,
        video_id=record.video_id,
        window_start=window.window_start,
        window_end=window.window_end,
        occurrence_id=occurrence_id,
    )
    return TrainingExample(
        module="A",  # Shared schema is intentionally closed to A/B/C.
        source_dataset=record.dataset,
        video_path=record.video_path,
        video_id=record.video_id,
        task_id=record.task_id,
        step_index=target_segment.instruction_index if target_segment else window.step_index,
        window_start=window.window_start,
        window_end=window.window_end,
        prompt_text=prompt,
        target_text=target,
        gt_start=target_segment.start if target_segment else window.gt_start,
        gt_end=target_segment.end if target_segment else window.gt_end,
        label=label,
        mistake=(
            target_segment.aggregate_class == "MISTAKE" if target_segment else False
        ),
        reasoning=reasoning,
        event_type=target_segment.event_type if target_segment else window.event_type,
        fine_class=fine_class,
        aggregate_class=aggregate_class,
        observed_description=(
            target_segment.observed_description if target_segment else window.observed_description
        ),
        occurrence_id=occurrence_id,
        fine_classes=(
            target_segment.resolved_fine_classes if target_segment else ()
        ),
        metadata={
            "pipeline": PIPELINE_MARKER,
            "dataset": record.dataset,
            **(
                {"prompt_contract": prompt_contract}
                if prompt_contract
                else {}
            ),
            "module_schema_fallback": "A",
            "cohort_family": family,
            "target_kind": "step_id" if completed else "not_completed",
            "semantic_label": "LOCALIZE" if completed else "NONE",
            "row_id": row_id,
            "reset_jitter_seconds": 0.0,
            "stride_seconds": 5.0,
            "clip_duration": clip_duration,
            "all_instructions": list(record.instructions),
            "history_source": "gold_prior_eligible_actions",
            "history_cutoff_seconds": cutoff,
            "history_cutoff_kind": (
                "target_segment_start" if completed else "poll_time"
            ),
            "history_occurrence_ids": [
                str(item["occurrence_id"]) for item in history
            ],
            "history_dropped": history_dropped,
            "history": [] if history_dropped else history,
            "history_before_dropout": history,
            "late_trigger_augmented": False,
            "sampled_incomplete": False,
            "positive_token_weight": 1.0,
        },
    )


def _target_segment(record: VideoRecord, window: WindowSpec) -> Segment:
    for segment in record.segments:
        if window.occurrence_id and _occurrence_id(record, segment) == window.occurrence_id:
            return segment
    candidates = [
        segment
        for segment in record.segments
        if segment.instruction_index == window.step_index
        and abs(segment.start - float(window.gt_start or 0.0)) < 1e-6
        and abs(segment.end - float(window.gt_end or 0.0)) < 1e-6
    ]
    if len(candidates) != 1:
        raise ValueError(f"{record.video_id}: could not resolve target occurrence")
    return candidates[0]


def _occurrence_id(record: VideoRecord, segment: Segment) -> str:
    return segment.occurrence_id or (
        f"{record.video_id}:{segment.annotation_position}:{segment.start:.3f}:{segment.end:.3f}"
    )


def _step_and_classification(
    segment: Segment,
    *,
    dataset: str,
    prompt_contract: str | None = None,
) -> tuple[str, str]:
    from ..prompts.abc_monolith import classification_matches_step

    step_id = step_id_for_instruction_index(segment.instruction_index)
    classification = _prompt_classification(
        str(segment.fine_class),
        dataset=dataset,
        prompt_contract=prompt_contract,
        instruction_index=segment.instruction_index,
    )
    if not classification_matches_step(
        step_id,
        classification,
        dataset,
        prompt_contract=prompt_contract,
    ):
        raise ValueError(
            f"{dataset} monolith step_id {step_id} does not match {classification}"
        )
    return step_id, classification


def _gold_history(
    record: VideoRecord,
    *,
    cutoff: float,
    exclude_occurrence_id: str,
    max_history: int,
    prompt_contract: str | None = None,
) -> list[dict[str, object]]:
    def _history_addition(segment: Segment) -> bool:
        return segment.fine_class == "ADDITION" or segment.event_type == "ADDITION"

    eligible = sorted(
        (
            segment
            for segment in record.segments
            if segment.event_type != "BACKGROUND"
            and not _history_addition(segment)
            and segment.duration >= 0.5
            and segment.end <= cutoff
            and _occurrence_id(record, segment) != exclude_occurrence_id
        ),
        key=lambda segment: (
            segment.end,
            segment.start,
            segment.annotation_position,
            _occurrence_id(record, segment),
        ),
    )
    result = []
    for segment in eligible[-max(0, max_history) :]:
        classification = _prompt_classification(
            str(segment.fine_class),
            dataset=record.dataset,
            prompt_contract=prompt_contract,
            instruction_index=segment.instruction_index,
        )
        result.append(
            {
                "occurrence_id": _occurrence_id(record, segment),
                "step_id": step_id_for_instruction_index(segment.instruction_index),
                "action": normalize_instruction_text(segment.instruction),
                "classification": classification,
                "reasoning": build_reasoning(
                    observed_description=(
                        segment.observed_description or segment.caption
                    ),
                    instruction=segment.instruction,
                    fine_class=classification,
                ),
            }
        )
    return result


def gold_history_for_poll(
    record: VideoRecord,
    *,
    window_start: float,
    window_end: float,
    target: Segment | None,
    max_history: int = 12,
) -> tuple[list[dict[str, object]], float, str]:
    """Causal gold history for one poll, using the training cutoff rules."""

    if not math.isfinite(window_start) or not math.isfinite(window_end):
        raise ValueError("window bounds must be finite")
    if window_end < window_start:
        raise ValueError("window_end must be at least window_start")
    if target is not None:
        cutoff = float(target.start)
        exclude = _occurrence_id(record, target)
        kind = "target_segment_start"
    else:
        cutoff = float(window_end)
        exclude = ""
        kind = "poll_time"
    history = _gold_history(
        record,
        cutoff=cutoff,
        exclude_occurrence_id=exclude,
        max_history=max_history,
    )
    return history, cutoff, kind


def _late_trigger_example(
    example: TrainingExample,
    record: VideoRecord,
    *,
    stride_seconds: float,
) -> TrainingExample | None:
    video_end = video_duration_seconds(record.video_path)
    if video_end <= 0:
        video_end = max((segment.end for segment in record.segments), default=example.window_end)
    delayed_end = min(video_end, example.window_end + stride_seconds)
    if delayed_end <= example.window_end:
        return None
    prompt_contract = example.metadata.get("prompt_contract")
    payload = json.loads(example.target_text)
    assert example.gt_start is not None and example.gt_end is not None
    local_start, local_end = clamp_module_b_local_span(
        example.gt_start - example.window_start,
        example.gt_end - example.window_start,
        delayed_end - example.window_start,
    )
    target = format_monolith_target(
        step_id=payload["step_id"],
        start=local_start,
        end=local_end,
        classification=payload["classification"],
        reasoning=payload["reasoning"],
        dataset=record.dataset,
        prompt_contract=prompt_contract,
    )
    row_id = stable_monolith_row_id(
        family=LATE_TRIGGER,
        video_id=example.video_id,
        window_start=example.window_start,
        window_end=delayed_end,
        occurrence_id=example.occurrence_id,
    )
    return replace(
        example,
        window_end=delayed_end,
        prompt_text=build_monolith_prompt(
            task_name=record.task_id,
            task_description=record.task_description,
            instructions=record.instructions,
            clip_duration=delayed_end - example.window_start,
            history=(
                None
                if example.metadata["history_dropped"]
                else example.metadata["history_before_dropout"]
            ),
            dataset=record.dataset,
            prompt_contract=prompt_contract,
        ),
        target_text=target,
        metadata={
            **example.metadata,
            "cohort_family": LATE_TRIGGER,
            "row_id": row_id,
            "clip_duration": delayed_end - example.window_start,
            "late_trigger_augmented": True,
            "late_trigger_stride_seconds": stride_seconds,
            "target_kind": "step_id",
            "semantic_label": "LOCALIZE",
        },
    )


def project_sampled_incomplete_examples(
    examples: Sequence[TrainingExample],
    *,
    records_by_video: dict[str, VideoRecord],
    max_history: int = 12,
    training: bool = False,
    history_dropout_probability: float = 0.25,
    seed: int = 15,
    prompt_contract: str | None = None,
) -> list[TrainingExample]:
    """Project native B/C sampled incomplete rows without mixing them with natural polls."""
    rng = random.Random(seed)
    projected: list[TrainingExample] = []
    for source in examples:
        if source.video_id not in records_by_video:
            continue
        if str(source.label) not in {"NO_ACTION", "NOT_COMPLETED"} and not source.metadata.get(
            "not_completed"
        ):
            continue
        record = records_by_video[source.video_id]
        history = _gold_history(
            record,
            cutoff=source.window_end,
            exclude_occurrence_id="",
            max_history=max_history,
            prompt_contract=prompt_contract,
        )
        dropped = training and rng.random() < max(0.0, history_dropout_probability)
        source_module = source.module
        row_id = stable_monolith_row_id(
            family=SAMPLED_INCOMPLETE,
            source_module=source_module,
            video_id=source.video_id,
            window_start=source.window_start,
            window_end=source.window_end,
            occurrence_id=source.occurrence_id,
        )
        projected.append(
            replace(
                source,
                module="A",
                prompt_text=build_monolith_prompt(
                    task_name=record.task_id,
                    task_description=record.task_description,
                    instructions=record.instructions,
                    clip_duration=source.window_end - source.window_start,
                    history=None if dropped else history,
                    max_history=max_history,
                    dataset=record.dataset,
                    prompt_contract=prompt_contract,
                ),
                target_text=NOT_COMPLETED_TARGET,
                label="NOT_COMPLETED",
                fine_class="NOT_COMPLETED",
                reasoning=None,
                metadata={
                    **source.metadata,
                    "pipeline": PIPELINE_MARKER,
                    "dataset": record.dataset,
                    **(
                        {"prompt_contract": prompt_contract}
                        if prompt_contract
                        else {}
                    ),
                    "module_schema_fallback": "A",
                    "cohort_family": SAMPLED_INCOMPLETE,
                    "source_module": source_module,
                    "row_id": row_id,
                    "target_kind": "not_completed",
                    "semantic_label": "NONE",
                    "sampled_incomplete": True,
                    "late_trigger_augmented": False,
                    "history_source": "gold_prior_eligible_actions",
                    "history_cutoff_seconds": source.window_end,
                    "history_cutoff_kind": "poll_time",
                    "history_occurrence_ids": [
                        str(item["occurrence_id"]) for item in history
                    ],
                    "history_dropped": dropped,
                    "history": [] if dropped else history,
                    "history_before_dropout": history,
                    "positive_token_weight": 1.0,
                },
            )
        )
    return projected


def materialize_validation_manifest(
    examples: Sequence[TrainingExample],
) -> dict[str, list[str]]:
    """Return deterministic fixed row lists suitable for JSON persistence."""
    cohorts: dict[str, list[str]] = {}
    for example in examples:
        family = str(example.metadata.get("cohort_family") or "unknown")
        row_id = str(example.metadata.get("row_id") or "")
        if row_id:
            cohorts.setdefault(family, []).append(row_id)
    return {family: sorted(set(row_ids)) for family, row_ids in sorted(cohorts.items())}


def materialize_native_validation_manifest(
    *,
    natural_a: Sequence[TrainingExample],
    native_b: Sequence[TrainingExample] = (),
    native_c: Sequence[TrainingExample] = (),
) -> dict[str, list[str]]:
    """Freeze native oracle-projection cohorts without depending on list order."""

    cohorts: dict[str, list[str]] = {
        "a_natural": [],
        "b_on_time": [],
        "b_late": [],
        "b_sampled_incomplete": [],
        "c_completed_non_addition": [],
        "c_sampled_incomplete": [],
    }

    def add(family: str, example: TrainingExample, source_module: str) -> None:
        cohorts[family].append(
            stable_monolith_row_id(
                family=family,
                source_module=source_module,
                video_id=example.video_id,
                window_start=example.window_start,
                window_end=example.window_end,
                occurrence_id=example.occurrence_id,
            )
        )

    for example in natural_a:
        add("a_natural", example, "A")
    for example in native_b:
        if (
            example.label == "NO_ACTION"
            or example.metadata.get("not_completed")
            or example.target_text == NOT_COMPLETED_TARGET
        ):
            family = "b_sampled_incomplete"
        elif example.metadata.get("late_trigger_augmented"):
            family = "b_late"
        else:
            family = "b_on_time"
        add(family, example, "B")
    for example in native_c:
        if (
            example.label == "NOT_COMPLETED"
            or example.metadata.get("not_completed")
            or example.target_text == NOT_COMPLETED_TARGET
        ):
            add("c_sampled_incomplete", example, "C")
        elif example.fine_class != "ADDITION":
            add("c_completed_non_addition", example, "C")
    return {family: sorted(set(row_ids)) for family, row_ids in cohorts.items()}


def monolith_sampling_counts(
    examples: Sequence[TrainingExample],
) -> dict[str, dict[str, int]]:
    """Count effective rows, including repeated mistake examples."""
    by_family: dict[str, int] = {}
    by_class: dict[str, int] = {}
    for example in examples:
        family = str(example.metadata.get("cohort_family") or "unknown")
        classification = str(example.fine_class or example.label or "unknown")
        by_family[family] = by_family.get(family, 0) + 1
        by_class[classification] = by_class.get(classification, 0) + 1
    return {
        "by_family": dict(sorted(by_family.items())),
        "by_class": dict(sorted(by_class.items())),
    }


def build_abc_monolith_training_examples(
    records: Iterable[VideoRecord],
    **kwargs: object,
) -> tuple[list[TrainingExample], dict[str, dict[str, dict[str, int]]]]:
    """Build training rows and explicitly report the pre/post sampling distribution."""
    records = list(records)
    options = dict(kwargs)
    options.pop("training", None)
    pre_sampling = build_abc_monolith_examples(records, training=False, **options)
    post_sampling = build_abc_monolith_examples(records, training=True, **options)
    return post_sampling, {
        "pre_sampling": monolith_sampling_counts(pre_sampling),
        "post_sampling": monolith_sampling_counts(post_sampling),
    }


def _project_native_example(
    source: TrainingExample,
    *,
    record: VideoRecord,
    projection_stage: str,
    cohort_family: str,
    max_history: int,
    prompt_contract: str | None = None,
) -> TrainingExample | None:
    incomplete = (
        str(source.label) in {"NO_ACTION", "NOT_COMPLETED"}
        or source.target_text == NOT_COMPLETED_TARGET
        or bool(source.metadata.get("not_completed"))
    )
    if incomplete:
        segment = None
    else:
        segment = _target_segment(
            record,
            WindowSpec(
                video_path=source.video_path,
                video_id=source.video_id,
                task_id=source.task_id,
                step_index=source.step_index,
                current_step=str(source.metadata.get("instruction") or ""),
                window_start=source.window_start,
                window_end=source.window_end,
                gt_start=source.gt_start,
                gt_end=source.gt_end,
                label=source.label or "",
                occurrence_id=source.occurrence_id,
            ),
        )
    if projection_stage == "C" and segment is not None and segment.fine_class == "ADDITION":
        return None
    cutoff = segment.start if segment is not None else source.window_end
    history = _gold_history(
        record,
        cutoff=cutoff,
        exclude_occurrence_id=(
            _occurrence_id(record, segment) if segment is not None else ""
        ),
        max_history=max_history,
        prompt_contract=prompt_contract,
    )
    clip_duration = source.window_end - source.window_start
    prompt = build_monolith_prompt(
        task_name=record.task_id,
        task_description=record.task_description,
        instructions=record.instructions,
        clip_duration=clip_duration,
        history=history,
        max_history=max_history,
        dataset=record.dataset,
        prompt_contract=prompt_contract,
    )
    if segment is None:
        target = NOT_COMPLETED_TARGET
        label = "NOT_COMPLETED"
        fine_class = "NOT_COMPLETED"
        reasoning = None
        occurrence_id = source.occurrence_id
    else:
        step_id, fine_class = _step_and_classification(
            segment,
            dataset=record.dataset,
            prompt_contract=prompt_contract,
        )
        local_start, local_end = clamp_module_b_local_span(
            segment.start - source.window_start,
            segment.end - source.window_start,
            clip_duration,
        )
        reasoning = build_reasoning(
            observed_description=segment.observed_description or segment.caption,
            instruction=segment.instruction,
            fine_class=fine_class,
        )
        target = format_monolith_target(
            step_id=step_id,
            start=local_start,
            end=local_end,
            classification=fine_class,
            reasoning=reasoning,
            dataset=record.dataset,
            prompt_contract=prompt_contract,
        )
        label = step_id
        occurrence_id = _occurrence_id(record, segment)
    row_id = stable_monolith_row_id(
        family=cohort_family,
        source_module=projection_stage,
        video_id=source.video_id,
        window_start=source.window_start,
        window_end=source.window_end,
        occurrence_id=occurrence_id,
    )
    return replace(
        source,
        module="A",
        prompt_text=prompt,
        target_text=target,
        label=label,
        fine_class=fine_class,
        reasoning=reasoning,
        occurrence_id=occurrence_id,
        metadata={
            **source.metadata,
            "pipeline": PIPELINE_MARKER,
            "dataset": record.dataset,
            **(
                {"prompt_contract": prompt_contract}
                if prompt_contract
                else {}
            ),
            "module_schema_fallback": "A",
            "projection_stage": projection_stage,
            "cohort_family": cohort_family,
            "row_id": row_id,
            "target_kind": "not_completed" if incomplete else "step_id",
            "semantic_label": "NONE" if incomplete else "LOCALIZE",
            "history": history,
            "history_before_dropout": history,
            "history_dropped": False,
            "all_instructions": list(record.instructions),
            "history_cutoff_seconds": cutoff,
            "history_cutoff_kind": (
                "poll_time" if incomplete else "target_segment_start"
            ),
            "history_occurrence_ids": [
                str(item["occurrence_id"]) for item in history
            ],
            "sampled_incomplete": incomplete,
            "late_trigger_augmented": bool(
                source.metadata.get("late_trigger_augmented")
            ),
            "positive_token_weight": 1.0,
        },
    )


def build_oracle_projection_examples(
    records: Iterable[VideoRecord],
    *,
    provider: object,
    seed: int = 3407,
    max_history: int = 12,
    native_b: Sequence[TrainingExample] | None = None,
    native_c: Sequence[TrainingExample] | None = None,
    prompt_contract: str | None = None,
) -> list[TrainingExample]:
    """Build fixed A/B/C validation cohorts rendered through the joint prompt."""

    records = list(records)
    records_by_video = {record.video_id: record for record in records}
    natural_a = build_abc_monolith_examples(
        records,
        training=False,
        seed=seed,
        max_history=max_history,
        late_trigger_probability=0.0,
        mistake_repeat_factor=1,
        prompt_contract=prompt_contract,
    )
    projected: list[TrainingExample] = [
        replace(
            example,
            metadata={
                **example.metadata,
                "projection_stage": "A",
                "cohort_family": "a_natural",
                "row_id": stable_monolith_row_id(
                    family="a_natural",
                    source_module="A",
                    video_id=example.video_id,
                    window_start=example.window_start,
                    window_end=example.window_end,
                    occurrence_id=example.occurrence_id,
                ),
            },
        )
        for example in natural_a
    ]
    module_a_config = ModuleAConfig(
        stride_seconds=5.0,
        reset_jitter_seconds=0.0,
        seed=seed,
        label_mode="step_id",
    )
    if native_b is None:
        native_b = build_module_b_training_dataset(
            provider,
            config=ModuleBConfig(
                late_trigger_probability=0.10,
                include_incomplete_negatives=True,
                negative_ratio=0.10,
                seed=seed,
            ),
            module_a_config=module_a_config,
        ).examples
    if native_c is None:
        native_c = build_module_c_training_dataset(
            provider,
            config=ModuleCConfig(
                min_duration_seconds=0.5,
                jitter_ratio_min=0.05,
                jitter_ratio_max=0.10,
                include_not_completed_examples=True,
                not_completed_ratio=0.10,
                history_max_events=max_history,
                seed=seed,
            ),
        ).examples
    selected_videos = set(records_by_video)
    for source in native_b:
        if source.video_id not in selected_videos:
            continue
        if (
            source.label == "NO_ACTION"
            or source.metadata.get("not_completed")
            or source.target_text == NOT_COMPLETED_TARGET
        ):
            family = "b_sampled_incomplete"
        elif source.metadata.get("late_trigger_augmented"):
            family = "b_late"
        else:
            family = "b_on_time"
        row = _project_native_example(
            source,
            record=records_by_video[source.video_id],
            projection_stage="B",
            cohort_family=family,
            max_history=max_history,
            prompt_contract=prompt_contract,
        )
        if row is not None:
            projected.append(row)
    for source in native_c:
        if source.video_id not in selected_videos:
            continue
        family = (
            "c_sampled_incomplete"
            if (
                source.label == "NOT_COMPLETED"
                or source.metadata.get("not_completed")
                or source.target_text == NOT_COMPLETED_TARGET
            )
            else "c_completed_non_addition"
        )
        row = _project_native_example(
            source,
            record=records_by_video[source.video_id],
            projection_stage="C",
            cohort_family=family,
            max_history=max_history,
            prompt_contract=prompt_contract,
        )
        if row is not None:
            projected.append(row)
    return sorted(
        projected,
        key=lambda item: (
            str(item.metadata.get("projection_stage")),
            str(item.metadata.get("cohort_family")),
            item.video_id,
            item.window_start,
            item.window_end,
            item.occurrence_id,
        ),
    )
