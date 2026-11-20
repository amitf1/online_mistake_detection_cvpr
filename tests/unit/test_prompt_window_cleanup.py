from __future__ import annotations

from pathlib import Path

from online_mistake_detection.config import ModuleAConfig
from online_mistake_detection.metrics.classification import parse_module_c_prediction
from online_mistake_detection.metrics.module_a import negative_sampling_metrics
from online_mistake_detection.metrics.temporal import parse_temporal_prediction
from online_mistake_detection.models.vision import (
    apply_resize_to_video_part,
    estimate_vision_tokens,
    pad_spatial_array,
    resolve_spatial_size,
)
from online_mistake_detection.windowing.module_b import clamp_module_b_local_span
from online_mistake_detection.pipeline.events import PipelineEvent
from online_mistake_detection.pipeline.runner import (
    apply_graph_stage,
    c_example_from_event,
    is_extra_module_a_prediction,
    predicted_module_c_history,
)
from online_mistake_detection.prompts.contracts import NOT_COMPLETED_STATUS, NOT_COMPLETED_TARGET
from online_mistake_detection.prompts.module_a import (
    MODULE_A_STEP_ID_SYSTEM_PROMPT,
    build_module_a_step_id_user_prompt,
)
from online_mistake_detection.prompts.module_b import MODULE_B_SYSTEM_PROMPT, format_module_b_target
from online_mistake_detection.prompts.module_c import (
    MODULE_C_SYSTEM_PROMPT,
    build_module_c_prompt,
)
from online_mistake_detection.schemas import Segment, VideoRecord
from online_mistake_detection.text import normalize_instruction_text
from online_mistake_detection.training.module_a import training_example_to_conversation as a_conversation
from online_mistake_detection.training.module_b import training_example_to_conversation as b_conversation
from online_mistake_detection.training.module_c import training_example_to_conversation as c_conversation
from online_mistake_detection.windowing.module_a import (
    build_module_a_windows as build_egoper_windows,
    cap_negative_windows,
)


def _record(*, dataset: str = "egoper", instructions=None, segments=None) -> VideoRecord:
    return VideoRecord(
        dataset=dataset,
        task_id="tea",
        video_id="vid",
        video_path=Path("/missing.mp4"),
        instructions=instructions or ["Clean-Clean the knife", "quarter-circle fold"],
        segments=segments
        or [
            Segment(
                start=0.0,
                end=6.0,
                instruction_index=0,
                instruction="Clean-Clean the knife",
                is_mistake=False,
                fine_class="CORRECT",
                aggregate_class="CORRECT",
                occurrence_id="vid:0",
            ),
            Segment(
                start=6.0,
                end=12.0,
                instruction_index=0,
                instruction="Clean-Clean the knife",
                is_mistake=True,
                fine_class="SLIP",
                aggregate_class="MISTAKE",
                occurrence_id="vid:1",
            ),
        ],
        task_description="tea",
    )


def test_normalize_duplicated_verb_prefix_only() -> None:
    assert normalize_instruction_text("Clean-Clean the knife") == "Clean the knife"
    assert normalize_instruction_text("quarter-circle fold") == "quarter-circle fold"
    assert normalize_instruction_text("Add-1/4 teaspoon") == "Add-1/4 teaspoon"


def test_prompt_ownership_is_not_duplicated() -> None:
    user = build_module_a_step_id_user_prompt(
        task_name="tea",
        task_description="brew tea",
        instructions=["Clean the knife"],
        completed_steps=[],
        dataset_name="egoper",
    )
    assert "Detect which procedural step" in MODULE_A_STEP_ID_SYSTEM_PROMPT
    assert "closest match" in MODULE_A_STEP_ID_SYSTEM_PROMPT
    assert "Measure 12 ounces of cold water using measuring cup" in MODULE_A_STEP_ID_SYSTEM_PROMPT
    assert "Reply with exactly one choice letter" in MODULE_A_STEP_ID_SYSTEM_PROMPT
    assert "Detect which procedural step" not in user
    assert "closed set" not in user
    assert '{"start"' in MODULE_B_SYSTEM_PROMPT
    assert "relevant_windows" not in MODULE_B_SYSTEM_PROMPT
    c_user = build_module_c_prompt("Clean the knife", ("CORRECT", "SLIP"), dataset_name="egoper")
    assert "ADDITION" not in c_user
    assert "SLIP means executing the requested step" in c_user
    assert "different bowl from the one containing oats" in c_user
    assert "dropping a tortilla on the floor" in c_user
    assert "MODIFICATION means" not in c_user
    assert "CORRECTION means performing an action that mitigates" in c_user
    assert "NOT_COMPLETED" in c_user
    full_c_user = build_module_c_prompt(
        "Pour the water",
        ("CORRECT", "CORRECTION", "MODIFICATION", "SLIP"),
        dataset_name="egoper",
    )
    assert "scooping nut butter with a spoon" in full_c_user
    assert "without the specified circular motion" in full_c_user
    assert "discarding a tortilla dropped on the floor" in full_c_user
    assert "Return JSON only" in MODULE_C_SYSTEM_PROMPT
    assert "Return JSON only" not in c_user


def test_egoper_reset_jitter_and_repeated_occurrences() -> None:
    record = _record()
    exact = build_egoper_windows([record], config=ModuleAConfig(stride_seconds=5.0, reset_jitter_seconds=0.0))
    jittered = build_egoper_windows(
        [record],
        config=ModuleAConfig(stride_seconds=5.0, reset_jitter_seconds=0.5, seed=0),
    )
    first_complete = next(window for window in exact if window.label == "STEP_0")
    second_complete = [window for window in exact if window.label == "STEP_0"][1]
    assert first_complete.window_start == 0.0
    assert second_complete.window_start == 6.0
    assert first_complete.occurrence_id != second_complete.occurrence_id
    assert "Clean the knife" in second_complete.completed_steps
    assert second_complete.completed_steps.count("Clean the knife") == 1
    third_record = _record(
        segments=list(_record().segments)
        + [
            Segment(
                start=12.0,
                end=18.0,
                instruction_index=0,
                instruction="Clean-Clean the knife",
                is_mistake=False,
                fine_class="CORRECT",
                aggregate_class="CORRECT",
                occurrence_id="vid:2",
            )
        ]
    )
    third_windows = build_egoper_windows(
        [third_record],
        config=ModuleAConfig(stride_seconds=5.0, reset_jitter_seconds=0.0),
    )
    third_complete = [window for window in third_windows if window.label == "STEP_0"][2]
    assert third_complete.completed_steps.count("Clean the knife") == 2
    jittered_second = [window for window in jittered if window.label == "STEP_0"][1]
    assert jittered_second.window_start <= 6.0


def test_train_only_negative_cap_keeps_first_mid_last() -> None:
    record = VideoRecord(
        dataset="egoper",
        task_id="tea",
        video_id="long",
        video_path=Path("/missing.mp4"),
        instructions=["step"],
        segments=[
            Segment(
                start=0.0,
                end=30.0,
                instruction_index=0,
                instruction="step",
                is_mistake=False,
                fine_class="CORRECT",
                aggregate_class="CORRECT",
            )
        ],
    )
    windows = build_egoper_windows([record], config=ModuleAConfig(stride_seconds=5.0))
    negatives = [window for window in windows if window.label == "NONE"]
    assert len(negatives) > 3
    capped, stats = cap_negative_windows(windows, max_negatives_per_episode=3)
    kept = [window for window in capped if window.label == "NONE"]
    assert len(kept) == 3
    assert kept[0].window_end == negatives[0].window_end
    assert kept[1].window_end == negatives[len(negatives) // 2].window_end
    assert kept[-1].window_end == negatives[-1].window_end
    assert stats["affected_episodes"] == 1
    assert stats["sampled_negative_windows"] == 3


def test_module_b_clamps_annotation_video_overshoot() -> None:
    start, end = clamp_module_b_local_span(1.87, 25.39, 25.25)
    assert start == 1.87
    assert end == 25.25
    start, end = clamp_module_b_local_span(0.73, 11.16, 10.02)
    assert (start, end) == (0.73, 10.02)
    try:
        clamp_module_b_local_span(30.0, 40.0, 25.25)
    except ValueError as exc:
        assert "does not overlap" in str(exc)
    else:
        raise AssertionError("non-overlapping span must remain an error")


def test_legacy_and_canonical_module_b_parse() -> None:
    assert parse_temporal_prediction(
        '{"relevant_windows":[[1.0, 2.0]]}',
        clip_duration=5,
    ).span == (1.0, 2.0)
    assert format_module_b_target(1.0, 2.0) == '{"start":1.0,"end":2.0}'
    assert parse_temporal_prediction('{"start":1.0,"end":2.0}', clip_duration=5).span == (1.0, 2.0)


def test_not_completed_propagation_and_extra_skips_c() -> None:
    parsed = parse_module_c_prediction(
        NOT_COMPLETED_TARGET,
        allowed_classes=("CORRECT", "SLIP"),
    )
    assert parsed.valid
    assert parsed.classification == NOT_COMPLETED_STATUS
    event = PipelineEvent(
        video_id="vid",
        task_id="tea",
        step_index=-1,
        instruction="undefined / extra step",
        trigger_time=10.0,
        accumulation_start=0.0,
        accumulation_end=10.0,
        module_a_prediction="B",
        pred_global_start=1.0,
        pred_global_end=4.0,
        stage_status="C_SKIPPED_EXTRA",
    )
    assert is_extra_module_a_prediction("B", label_mode="step_id")
    assert c_example_from_event(event, dataset_name="egoper") is not None
    apply_graph_stage([event], [])


def test_pipeline_module_c_history_uses_prior_predictions_only() -> None:
    first = PipelineEvent(
        video_id="vid",
        task_id="task",
        step_index=0,
        instruction="pour water",
        trigger_time=5.0,
        accumulation_start=0.0,
        accumulation_end=5.0,
        module_a_prediction="A",
        module_c_classification="SLIP",
        module_c_reasoning="water went into the wrong bowl",
    )
    current = PipelineEvent(
        video_id="vid",
        task_id="task",
        step_index=1,
        instruction="move water",
        trigger_time=8.0,
        accumulation_start=5.0,
        accumulation_end=8.0,
        module_a_prediction="B",
        pred_global_start=5.0,
        pred_global_end=8.0,
    )
    future = PipelineEvent(
        video_id="vid",
        task_id="task",
        step_index=2,
        instruction="stir",
        trigger_time=12.0,
        accumulation_start=8.0,
        accumulation_end=12.0,
        module_a_prediction="C",
        module_c_classification="CORRECT",
    )
    events = [first, current, future]

    history = predicted_module_c_history(events, current_index=1, max_events=12)
    example = c_example_from_event(
        current,
        dataset_name="egoper",
        history=history,
        history_max_events=12,
    )

    assert example is not None
    assert "Classification: SLIP" in example.prompt_text
    assert "water went into the wrong bowl" in example.prompt_text
    assert "Action: stir" not in example.prompt_text


def test_aspect_preserving_and_native_geometry() -> None:
    native = resolve_spatial_size(1280, 720, 0)
    assert native.mode == "native"
    assert (native.width, native.height) == (1280, 736)
    assert native.interpolated is False
    resized = resolve_spatial_size(1280, 720, 720)
    assert resized.mode == "resize"
    assert resized.interpolated is True
    assert resized.width <= 720
    assert resized.width % 32 == 0
    assert resized.height % 32 == 0
    no_upscale = resolve_spatial_size(640, 360, 1280)
    assert no_upscale.mode == "native"
    import numpy as np

    padded = pad_spatial_array(np.zeros((2, 720, 1280, 3), dtype=np.uint8), 1280, 736)
    assert padded.shape == (2, 736, 1280, 3)
    assert estimate_vision_tokens(1280, 736, 32) == 14720
    part = {"type": "video", "video": np.zeros((2, 720, 1280, 3), dtype=np.uint8)}
    apply_resize_to_video_part(part, 0)
    assert (part["resized_width"], part["resized_height"]) == (1280, 736)
    assert (part["vision_effective_width"], part["vision_effective_height"]) == (1280, 736)
    assert part["vision_interpolated"] is False
    assert "max_pixels" not in part


def test_conversations_keep_static_system_and_dynamic_user() -> None:
    from online_mistake_detection.schemas import TrainingExample

    example = TrainingExample(
        module="A",
        source_dataset="egoper",
        video_path=Path("/missing.mp4"),
        video_id="vid",
        task_id="tea",
        step_index=0,
        window_start=0.0,
        window_end=5.0,
        prompt_text=build_module_a_step_id_user_prompt(
            task_name="tea",
            task_description="brew",
            instructions=["step"],
            completed_steps=[],
            dataset_name="egoper",
        ),
        target_text="C",
        metadata={"label_mode": "step_id"},
    )
    conversation = a_conversation(example, fps=1.0, min_frames=2, max_frames=4)
    system = conversation["messages"][0]["content"][0]["text"]
    user = conversation["messages"][1]["content"][1]["text"]
    assert system == MODULE_A_STEP_ID_SYSTEM_PROMPT
    assert "Task: tea" in user
    b_example = TrainingExample(
        module="B",
        source_dataset="egoper",
        video_path=Path("/missing.mp4"),
        video_id="vid",
        task_id="tea",
        step_index=0,
        window_start=0.0,
        window_end=5.0,
        prompt_text="Instruction: step",
        target_text=NOT_COMPLETED_TARGET,
    )
    b_conv = b_conversation(b_example, fps=1.0, min_frames=2, max_frames=4)
    assert "relevant_windows" not in b_conv["messages"][0]["content"][0]["text"]
    c_example = TrainingExample(
        module="C",
        source_dataset="egoper",
        video_path=Path("/missing.mp4"),
        video_id="vid",
        task_id="tea",
        step_index=0,
        window_start=0.0,
        window_end=5.0,
        prompt_text=build_module_c_prompt("step", ("CORRECT", "SLIP"), dataset_name="egoper"),
        target_text=NOT_COMPLETED_TARGET,
    )
    c_conv = c_conversation(c_example, fps=1.0, min_frames=2, max_frames=4)
    assert "NOT_COMPLETED" in c_conv["messages"][0]["content"][0]["text"]


def test_negative_sampling_metrics_mark_full_validation() -> None:
    metrics = negative_sampling_metrics(
        {
            "natural_negative_windows": 10,
            "sampled_negative_windows": 3,
            "affected_episodes": 1,
            "validation_windows_sampled": False,
        }
    )
    assert metrics["eval/natural_negative_windows"] == 10
    assert metrics["eval/sampled_negative_windows"] == 3
    assert metrics["eval/negative_drop_rate"] == 0.7
    assert metrics["eval/full_window_validation"] == 1.0
