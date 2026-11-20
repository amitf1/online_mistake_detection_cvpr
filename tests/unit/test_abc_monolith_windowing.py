from __future__ import annotations

import json
from pathlib import Path

from online_mistake_detection.schemas import Segment, TrainingExample, VideoRecord
from online_mistake_detection.windowing.abc_monolith import (
    LATE_TRIGGER,
    NATURAL_NOT_COMPLETED,
    SAMPLED_INCOMPLETE,
    build_abc_monolith_examples,
    materialize_validation_manifest,
)


def _record() -> VideoRecord:
    return VideoRecord(
        dataset="egoper",
        task_id="oats",
        video_id="video",
        video_path=Path("/missing.mp4"),
        instructions=["pour water", "stir oats"],
        task_description="prepare oats",
        segments=[
            Segment(
                start=1.0,
                end=6.0,
                instruction_index=0,
                instruction="pour water",
                is_mistake=True,
                event_type="MISTAKE",
                fine_class="SLIP",
                aggregate_class="MISTAKE",
                observed_description="Water enters the empty bowl.",
                occurrence_id="first",
            ),
            Segment(
                start=6.0,
                end=26.0,
                instruction_index=1,
                instruction="stir oats",
                is_mistake=False,
                fine_class="CORRECT",
                aggregate_class="CORRECT",
                occurrence_id="second",
            ),
        ],
    )


def test_windows_use_zero_jitter_local_spans_and_leak_free_history() -> None:
    examples = build_abc_monolith_examples([_record()])
    first = next(item for item in examples if item.occurrence_id == "first" and item.label == "STEP_0")
    payload = json.loads(first.target_text)
    assert payload["start"] == 1.0
    assert payload["end"] == 6.0
    assert payload["classification"] == "SLIP"
    assert first.metadata["reset_jitter_seconds"] == 0.0
    assert first.metadata["history_cutoff_kind"] == "target_segment_start"
    assert first.metadata["history_occurrence_ids"] == []

    second = next(item for item in examples if item.occurrence_id == "second" and item.label == "STEP_1")
    assert second.window_start == 6.0
    assert json.loads(second.target_text)["start"] == 0.0
    assert second.metadata["history_occurrence_ids"] == ["first"]
    assert "second" not in second.metadata["history_occurrence_ids"]

    wait = next(
        item
        for item in examples
        if item.metadata["cohort_family"] == NATURAL_NOT_COMPLETED
    )
    assert wait.target_text == '{"status":"NOT_COMPLETED"}'
    assert wait.metadata["history_cutoff_kind"] == "poll_time"


def test_training_caps_natural_waits_adds_late_steps_and_repeats_mistakes() -> None:
    examples = build_abc_monolith_examples(
        [_record()],
        training=True,
        history_dropout_probability=1.0,
        late_trigger_probability=1.0,
        mistake_repeat_factor=4,
    )
    natural = [
        item for item in examples if item.metadata["cohort_family"] == NATURAL_NOT_COMPLETED
    ]
    assert len(natural) == 3  # Two episodes, each capped independently at two.
    late = [item for item in examples if item.metadata["cohort_family"] == LATE_TRIGGER]
    assert late
    assert all(item.metadata["target_kind"] == "step_id" for item in late)
    assert all("step_id" in json.loads(item.target_text) for item in late)
    assert all(item.metadata["history_dropped"] for item in examples)
    assert all("unavailable due to training history dropout" in item.prompt_text for item in examples)

    first_rows = [
        item
        for item in examples
        if item.occurrence_id == "first"
        and item.metadata["cohort_family"] != LATE_TRIGGER
        and item.metadata["target_kind"] == "step_id"
    ]
    assert len(first_rows) == 4
    assert all(item.metadata["positive_token_weight"] == 1.0 for item in examples)


def test_extra_addition_and_sampled_incomplete_cohorts_remain_distinct() -> None:
    extra_record = VideoRecord(
        dataset="egoper",
        task_id="oats",
        video_id="extra",
        video_path=Path("/missing.mp4"),
        instructions=["pour water"],
        segments=[
            Segment(
                start=0.0,
                end=2.0,
                instruction_index=-1,
                instruction="add cinnamon",
                is_mistake=True,
                event_type="ADDITION",
                fine_class="ADDITION",
                aggregate_class="MISTAKE",
                occurrence_id="extra-action",
            )
        ],
    )
    sampled = TrainingExample(
        module="B",
        source_dataset="egoper",
        video_path=extra_record.video_path,
        video_id="extra",
        task_id="oats",
        step_index=0,
        window_start=0.0,
        window_end=1.0,
        prompt_text="native B",
        target_text='{"status":"NOT_COMPLETED"}',
        label="NO_ACTION",
        metadata={"not_completed": True},
    )
    examples = build_abc_monolith_examples(
        [extra_record],
        sampled_incomplete_examples=[sampled],
    )
    completed = next(item for item in examples if item.occurrence_id == "extra-action")
    payload = json.loads(completed.target_text)
    assert (payload["step_id"], payload["classification"]) == ("EXTRA", "ADDITION")
    projected = next(
        item for item in examples if item.metadata["cohort_family"] == SAMPLED_INCOMPLETE
    )
    assert projected.target_text == '{"status":"NOT_COMPLETED"}'
    assert projected.metadata["source_module"] == "B"
    assert projected.metadata["cohort_family"] != NATURAL_NOT_COMPLETED

    manifest = materialize_validation_manifest(examples)
    assert manifest == materialize_validation_manifest(list(reversed(examples)))


