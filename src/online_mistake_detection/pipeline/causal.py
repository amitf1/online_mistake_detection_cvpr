"""Causal A -> B (-> C) streaming for one video."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from ..datasets.labels import EXTRA_STEP_INSTRUCTION, is_module_a_completion_label
from ..prompts.module_a import build_module_a_step_id_user_prompt
from ..schemas import Segment, TrainingExample, VideoRecord, WindowSpec
from ..text import normalize_instruction_text
from .events import PipelineEvent

ModuleAPrediction = tuple[WindowSpec, str, str, float]
ScoreModuleA = Callable[[WindowSpec, TrainingExample], tuple[str, str, float]]
RunModuleB = Callable[[WindowSpec, str, str, float], PipelineEvent]
RunModuleC = Callable[[PipelineEvent, list[PipelineEvent]], None]

MODULE_C_NOT_COMPLETED = "C_NOT_COMPLETED"

CAUSAL_STREAM_VIEW = "causal_predicted_cursor_predicted_history"


def eligible_segments(record: VideoRecord) -> list[Segment]:
    return sorted(
        (
            segment
            for segment in record.segments
            if segment.event_type != "BACKGROUND" and segment.duration >= 0.5
        ),
        key=lambda item: (item.end, item.start, item.annotation_position),
    )


def target_for_window(
    record: VideoRecord,
    *,
    window_start: float,
    window_end: float,
) -> Segment | None:
    """First eligible gold action that ends inside the clip. Scoring only."""

    return next(
        (
            segment
            for segment in eligible_segments(record)
            if segment.end > window_start + 1e-9 and segment.end <= window_end + 1e-9
        ),
        None,
    )


def gold_letter(target: Segment | None) -> str:
    if target is None:
        return "A"
    if target.instruction_index < 0:
        return "B"
    return chr(ord("C") + target.instruction_index)


def causal_window(
    record: VideoRecord,
    *,
    window_start: float,
    window_end: float,
    completed_steps: tuple[str, ...],
) -> WindowSpec:
    target = target_for_window(record, window_start=window_start, window_end=window_end)
    label = gold_letter(target)
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
        completed_steps=completed_steps,
        all_instructions=tuple(normalize_instruction_text(item) for item in record.instructions),
        label_mode="step_id",
        event_type="NORMAL" if target is None else target.event_type,
        fine_class=None if target is None else target.fine_class,
        aggregate_class=None if target is None else target.aggregate_class,
        observed_description="" if target is None else target.observed_description,
        occurrence_id="" if target is None else target.occurrence_id,
        annotation_position=-1 if target is None else target.annotation_position,
        completed_step_ids=() if label == "A" else (label,),
        fine_classes=() if target is None else target.resolved_fine_classes,
        dataset_metadata={
            "dataset": record.dataset,
            "evaluation_view": CAUSAL_STREAM_VIEW,
        },
    )


def causal_module_a_example(record: VideoRecord, window: WindowSpec) -> TrainingExample:
    prompt = build_module_a_step_id_user_prompt(
        task_name=record.task_id,
        task_description=record.task_description or record.task_id,
        instructions=record.instructions,
        completed_steps=window.completed_steps,
        dataset_name=record.dataset,
    )
    return TrainingExample(
        module="A",
        source_dataset=record.dataset,
        video_path=window.video_path,
        video_id=window.video_id,
        task_id=window.task_id,
        step_index=window.step_index,
        window_start=window.window_start,
        window_end=window.window_end,
        gt_start=window.gt_start,
        gt_end=window.gt_end,
        prompt_text=prompt,
        target_text=window.label,
        label=window.label,
        event_type=window.event_type,
        fine_class=window.fine_class,
        aggregate_class=window.aggregate_class,
        observed_description=window.observed_description,
        occurrence_id=window.occurrence_id,
        fine_classes=window.fine_classes,
        metadata={
            "completed_steps": list(window.completed_steps),
            "completed_step_ids": list(window.completed_step_ids),
            "all_instructions": list(window.all_instructions),
            "label_mode": "step_id",
        },
    )


@dataclass
class CausalPipelineState:
    cursor: float = 0.0
    completed_steps: list[str] = field(default_factory=list)
    poll_rows: list[dict[str, Any]] = field(default_factory=list)

    def should_poll(self, poll_end: float) -> bool:
        return float(poll_end) > self.cursor + 1e-6

    def accept(self, event: PipelineEvent, prediction: str) -> bool:
        """Advance on a placed step. Return whether the cursor moved."""

        if event.pred_global_end is None:
            return False
        self.cursor = float(event.pred_global_end)
        if prediction.strip().upper().startswith("B"):
            return True
        text = normalize_instruction_text(event.instruction or "")
        if text and text != normalize_instruction_text(EXTRA_STEP_INSTRUCTION):
            self.completed_steps.append(text)
        return True


def predicted_letter_step_index(prediction: str) -> int:
    letter = prediction.strip().upper()[:1]
    if not letter or letter < "C" or letter > "Z":
        return -1
    return ord(letter) - ord("C")


def strip_gold_from_event(event: PipelineEvent, prediction: str) -> PipelineEvent:
    """Module B copies the clip's gold target onto the event; replace it with Module A's answer."""

    event.step_index = predicted_letter_step_index(prediction)
    event.occurrence_id = ""
    event.annotation_position = -1
    event.module_a_completed_steps = (prediction.strip().upper()[:1],)
    return event


@dataclass
class CausalVideoResult:
    module_a_predictions: list[ModuleAPrediction]
    events: list[PipelineEvent]
    poll_rows: list[dict[str, Any]]


def _poll_row(
    *,
    window: WindowSpec,
    prediction: str,
    raw: str,
    seconds: float,
    event: PipelineEvent | None,
    disposition: str,
) -> dict[str, Any]:
    window_payload = asdict(window)
    window_payload["video_path"] = str(window.video_path)
    return {
        "window": window_payload,
        "prediction": prediction,
        "raw_output": raw,
        "prediction_seconds": seconds,
        "is_trigger_prediction": is_module_a_completion_label(prediction, label_mode="step_id"),
        "entered_module_b": event is not None,
        "skipped_by_online_b_reset": False,
        "cursor": window.window_start,
        "completed_steps_in_prompt": list(window.completed_steps),
        "gold_label": window.label,
        "disposition": disposition,
        "module_b_end": None if event is None else event.pred_global_end,
    }


def run_causal_video(
    record: VideoRecord,
    *,
    duration: float,
    score_a: ScoreModuleA,
    run_b: RunModuleB,
    run_c: RunModuleC | None = None,
    stride_seconds: float = 5.0,
) -> CausalVideoResult:
    """Stream one video through Module A, Module B and optionally Module C."""

    if stride_seconds <= 0:
        raise ValueError("stride_seconds must be positive")
    state = CausalPipelineState()
    predictions: list[ModuleAPrediction] = []
    events: list[PipelineEvent] = []
    poll_end = min(duration, stride_seconds)
    while poll_end > 0:
        if state.should_poll(poll_end):
            window = causal_window(
                record,
                window_start=state.cursor,
                window_end=poll_end,
                completed_steps=tuple(state.completed_steps),
            )
            example = causal_module_a_example(record, window)
            prediction, raw, seconds = score_a(window, example)
            predictions.append((window, prediction, raw, seconds))
            event: PipelineEvent | None = None
            disposition = "retained"
            if is_module_a_completion_label(prediction, label_mode="step_id"):
                event = strip_gold_from_event(run_b(window, prediction, raw, seconds), prediction)
                events.append(event)
                if event.pred_global_end is None:
                    disposition = "module_b_not_completed"
                else:
                    if run_c is not None:
                        run_c(event, events)
                    if event.stage_status == MODULE_C_NOT_COMPLETED:
                        disposition = "module_c_not_completed"
                    else:
                        state.accept(event, prediction)
                        disposition = "reset_to_module_b_end"
            state.poll_rows.append(
                _poll_row(
                    window=window,
                    prediction=prediction,
                    raw=raw,
                    seconds=seconds,
                    event=event,
                    disposition=disposition,
                )
            )
        if poll_end >= duration - 1e-9:
            break
        poll_end = min(duration, poll_end + stride_seconds)
    return CausalVideoResult(
        module_a_predictions=predictions,
        events=events,
        poll_rows=state.poll_rows,
    )
