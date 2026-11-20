from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar


@dataclass(frozen=True)
class GroundTruthEvent:
    video_id: str
    task_id: str
    step_index: int
    instruction: str
    gt_start: float
    gt_end: float
    gt_mistake: bool
    gt_reasoning: str
    eligible: bool
    gt_fine_class: str | None = None
    gt_aggregate_class: str | None = None
    occurrence_id: str = ""
    annotation_position: int = -1
    gt_fine_classes: tuple[str, ...] = ()
    gt_structural_labels: tuple[str, ...] = ()


@dataclass
class PipelineEvent:
    video_id: str
    task_id: str
    step_index: int
    instruction: str
    trigger_time: float
    accumulation_start: float
    accumulation_end: float
    module_a_prediction: str
    module_a_raw: str | None = None
    module_b_raw: str | None = None
    module_b_prompt_text: str | None = None
    module_b_messages: list[dict[str, Any]] = field(default_factory=list)
    module_b_raw_prediction: tuple[float, float] | None = None
    module_b_prediction: tuple[float, float] | None = None
    module_b_rejection_reason: str | None = None
    module_b_was_clamped: bool = False
    module_b_clamp_reason: str | None = None
    pred_raw_global_start: float | None = None
    pred_raw_global_end: float | None = None
    pred_global_start: float | None = None
    pred_global_end: float | None = None
    module_c_raw: str | None = None
    module_c_prompt_text: str | None = None
    module_c_messages: list[dict[str, Any]] = field(default_factory=list)
    module_c_history: list[dict[str, str]] = field(default_factory=list)
    module_c_classification: str | None = None
    module_c_aggregate_class: str | None = None
    module_c_mistake: bool | None = None
    module_c_error_score: float | None = None
    module_c_class_probabilities: dict[str, float] | None = None
    module_c_candidate_nlls: dict[str, float] | None = None
    module_c_reasoning: str | None = None
    module_a_prediction_seconds: float | None = None
    module_b_prediction_seconds: float | None = None
    module_c_prediction_seconds: float | None = None
    module_c_score_seconds: float | None = None
    pipeline_prediction_seconds: float | None = None
    monolith_prediction_seconds: float | None = None
    matched_gt_index: int | None = None
    matched_iou: float = 0.0
    matched_gt_start: float | None = None
    matched_gt_end: float | None = None
    gt_mistake: bool | None = None
    gt_reasoning: str | None = None
    gt_fine_class: str | None = None
    gt_aggregate_class: str | None = None
    duplicate: bool = False
    occurrence_id: str = ""
    annotation_position: int = -1
    module_a_completed_steps: tuple[str, ...] = ()
    # Instruction text of the steps completed before this window, as opposed to
    # the step ids above. Module C reads these when classifying an off-script clip.
    module_a_completed_instructions: tuple[str, ...] = ()
    module_c_classifications: tuple[str, ...] = ()
    module_c_reasoning_by_class: dict[str, list[str]] | None = None
    graph_alerts: list[dict[str, Any]] = field(default_factory=list)
    fused_labels: tuple[str, ...] = ()
    structural_revision: str | None = None
    gt_fine_classes: tuple[str, ...] = ()
    gt_structural_labels: tuple[str, ...] = ()
    stage_status: str = ""


def predicted_step_index(event: PipelineEvent) -> int | None:
    """Decode the step identity emitted by Module A."""

    prediction = str(event.module_a_prediction or "").strip().upper()
    if prediction in {"A", "NONE", "NOT_COMPLETED"}:
        return None
    if prediction in {"B", "EXTRA"}:
        return -1
    if len(prediction) == 1 and "C" <= prediction <= "Z":
        return ord(prediction) - ord("C")
    if prediction.startswith("STEP_"):
        try:
            return int(prediction.removeprefix("STEP_"))
        except ValueError:
            return None
    return None


T = TypeVar("T", GroundTruthEvent, PipelineEvent)


def write_events(path: str | Path, events: list[PipelineEvent]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps([asdict(event) for event in events], indent=2) + "\n",
        encoding="utf-8",
    )


def events_from_rows(event_type: type[T], rows: list[dict[str, Any]]) -> list[T]:
    allowed = {field.name for field in fields(event_type)}
    return [
        event_type(**{key: value for key, value in row.items() if key in allowed})
        for row in rows
    ]
