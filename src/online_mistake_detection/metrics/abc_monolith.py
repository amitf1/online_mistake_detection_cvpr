"""Evaluation helpers for the EgoPER one-pass A/B/C monolith."""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field, is_dataclass
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

from ..pipeline.events import PipelineEvent


NOT_COMPLETED = "NOT_COMPLETED"
EXECUTION_CLASSES = (
    "CORRECT",
    "SLIP",
    "MODIFICATION",
    "CORRECTION",
    "ADDITION",
)
MODULE_C_CLASSES = ("CORRECT", "SLIP", "MODIFICATION", "CORRECTION")
_STEP_PATTERN = re.compile(r"STEP_(0|[1-9]\d*)\Z")


@dataclass(frozen=True)
class MonolithPrediction:
    valid: bool
    completed: bool
    step_id: str | None = None
    start: float | None = None
    end: float | None = None
    classification: str | None = None
    reasoning: str | None = None
    error: str | None = None
    payload: dict[str, Any] | None = None
    field_validity: dict[str, bool] = field(default_factory=dict)

    @property
    def status(self) -> str | None:
        return NOT_COMPLETED if self.valid and not self.completed else None


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def parse_monolith_prediction(
    text: str,
    *,
    clip_duration: float | None = None,
    dataset: str = "egoper",
    prompt_contract: str | None = None,
) -> MonolithPrediction:
    """Parse only the exact documented JSON object; never salvage prose/JSON."""

    try:
        payload = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return MonolithPrediction(False, False, error="invalid_json")
    if not isinstance(payload, dict):
        return MonolithPrediction(False, False, error="not_an_object")
    if payload == {"status": NOT_COMPLETED}:
        return MonolithPrediction(
            True,
            False,
            payload=payload,
            field_validity={"status": True},
        )
    if "status" in payload:
        return MonolithPrediction(
            False,
            False,
            error="status_is_not_allowed_for_completion",
            payload=payload,
        )

    required = {"step_id", "start", "end", "classification", "reasoning"}
    if set(payload) != required:
        return MonolithPrediction(
            False,
            False,
            error="unexpected_or_missing_fields",
            payload=payload,
        )
    step_id = payload["step_id"]
    classification = payload["classification"]
    reasoning = payload["reasoning"]
    start = payload["start"]
    end = payload["end"]
    step_valid = isinstance(step_id, str) and (
        step_id == "EXTRA" or _STEP_PATTERN.fullmatch(step_id) is not None
    )
    from ..prompts.abc_monolith import (
        classification_matches_step,
        execution_classes_for_dataset,
    )

    allowed_classes = execution_classes_for_dataset(
        dataset,
        prompt_contract=prompt_contract,
    )
    class_valid = (
        isinstance(classification, str) and classification in allowed_classes
    )
    reasoning_valid = isinstance(reasoning, str) and bool(reasoning.strip())
    start_valid = _finite_number(start)
    end_valid = _finite_number(end)
    bounds_valid = start_valid and end_valid and 0.0 <= float(start) < float(end)
    normalized_end = float(end) if end_valid else None
    if bounds_valid and clip_duration is not None:
        if not _finite_number(clip_duration) or float(clip_duration) <= 0.0:
            raise ValueError("clip_duration must be a finite positive number")
        bounds_valid = (
            float(end) <= round(float(clip_duration), 2) + 1e-9
            and float(start) < float(clip_duration)
        )
        normalized_end = min(float(end), float(clip_duration))
    extra_consistent = (
        step_valid
        and class_valid
        and classification_matches_step(
            str(step_id),
            str(classification),
            dataset,
            prompt_contract=prompt_contract,
        )
    )
    validity = {
        "step_id": step_valid,
        "start": start_valid,
        "end": end_valid,
        "bounds": bounds_valid,
        "classification": class_valid,
        "reasoning": reasoning_valid,
        "extra_addition_consistency": extra_consistent,
    }
    if not all(validity.values()):
        return MonolithPrediction(
            False,
            True,
            error=next(key for key, valid in validity.items() if not valid),
            payload=payload,
            field_validity=validity,
        )
    return MonolithPrediction(
        True,
        True,
        step_id=str(step_id),
        start=float(start),
        end=float(normalized_end),
        classification=str(classification),
        reasoning=str(reasoning).strip(),
        payload=payload,
        field_validity=validity,
    )


# Stable short aliases used by evaluation entry points.
parse_prediction = parse_monolith_prediction
parse_abc_monolith_prediction = parse_monolith_prediction


def monolith_global_span(
    prediction: MonolithPrediction,
    *,
    window_start: float,
) -> tuple[float, float] | None:
    if not prediction.valid or not prediction.completed:
        return None
    assert prediction.start is not None and prediction.end is not None
    return (
        float(window_start) + prediction.start,
        float(window_start) + prediction.end,
    )


def step_id_to_module_a_label(step_id: str | None) -> str:
    if step_id is None:
        return "A"
    if step_id == "EXTRA":
        return "B"
    match = _STEP_PATTERN.fullmatch(step_id)
    if match is None:
        raise ValueError(f"Invalid monolith step id: {step_id!r}")
    index = int(match.group(1))
    codepoint = ord("C") + index
    if codepoint > ord("Z"):
        raise ValueError("Module A native projection supports STEP_0 through STEP_23")
    return chr(codepoint)


@dataclass(frozen=True)
class MonolithProjection:
    module_a: dict[str, Any]
    module_b: dict[str, Any] | None
    module_c: dict[str, Any] | None
    event: PipelineEvent | None


def project_monolith_prediction(
    prediction: MonolithPrediction | str,
    *,
    video_id: str,
    task_id: str,
    instruction: str,
    window_start: float,
    window_end: float,
    trigger_time: float | None = None,
    prediction_seconds: float | None = None,
    raw_output: str | None = None,
    class_probabilities: Mapping[str, float] | None = None,
    candidate_nlls: Mapping[str, float] | None = None,
    occurrence_id: str = "",
    annotation_position: int = -1,
    predicted_history: Sequence[Mapping[str, Any]] = (),
) -> MonolithProjection:
    """Project one joint response to native stage rows and ``PipelineEvent``."""

    parsed = (
        parse_monolith_prediction(
            prediction,
            clip_duration=float(window_end) - float(window_start),
        )
        if isinstance(prediction, str)
        else prediction
    )
    raw = raw_output if raw_output is not None else (
        prediction if isinstance(prediction, str) else json.dumps(parsed.payload)
    )
    module_a = {
        "prediction": step_id_to_module_a_label(parsed.step_id)
        if parsed.valid
        else "A",
        "semantic_prediction": parsed.step_id if parsed.completed else NOT_COMPLETED,
        "raw_output": raw,
        "prediction_seconds": prediction_seconds,
        "valid": parsed.valid,
    }
    if not parsed.valid or not parsed.completed:
        return MonolithProjection(module_a, None, None, None)

    assert parsed.start is not None and parsed.end is not None
    assert parsed.step_id is not None and parsed.classification is not None
    raw_start = float((parsed.payload or {}).get("start", parsed.start))
    raw_end = float((parsed.payload or {}).get("end", parsed.end))
    global_start = float(window_start) + parsed.start
    global_end = float(window_start) + parsed.end
    raw_global_start = float(window_start) + raw_start
    raw_global_end = float(window_start) + raw_end
    step_index = (
        -1 if parsed.step_id == "EXTRA" else int(parsed.step_id.removeprefix("STEP_"))
    )
    module_b = {
        "start": parsed.start,
        "end": parsed.end,
        "raw_start": raw_start,
        "raw_end": raw_end,
        "global_start": global_start,
        "global_end": global_end,
        "raw_global_start": raw_global_start,
        "raw_global_end": raw_global_end,
        "was_clamped": raw_start != parsed.start or raw_end != parsed.end,
        "valid": True,
    }
    # ADDITION is a pipeline class, but is deliberately outside native Module C.
    module_c = None if parsed.classification == "ADDITION" else {
        "classification": parsed.classification,
        "reasoning": parsed.reasoning,
        "valid": True,
    }
    probabilities = (
        {str(key): float(value) for key, value in class_probabilities.items()}
        if class_probabilities is not None
        else None
    )
    error_score = (
        sum(
            probability
            for class_name, probability in probabilities.items()
            if class_name != "CORRECT"
        )
        if probabilities is not None
        else None
    )
    event = PipelineEvent(
        video_id=video_id,
        task_id=task_id,
        step_index=step_index,
        instruction=instruction,
        trigger_time=float(window_end if trigger_time is None else trigger_time),
        accumulation_start=float(window_start),
        accumulation_end=float(window_end),
        module_a_prediction=step_id_to_module_a_label(parsed.step_id),
        module_a_raw=raw,
        module_b_raw=raw,
        module_b_raw_prediction=(raw_start, raw_end),
        module_b_prediction=(parsed.start, parsed.end),
        module_b_was_clamped=raw_start != parsed.start or raw_end != parsed.end,
        module_b_clamp_reason=(
            "displayed_clip_boundary"
            if raw_start != parsed.start or raw_end != parsed.end
            else None
        ),
        pred_raw_global_start=raw_global_start,
        pred_raw_global_end=raw_global_end,
        pred_global_start=global_start,
        pred_global_end=global_end,
        module_c_raw=raw,
        module_c_history=[
            {str(key): str(value) for key, value in row.items()}
            for row in predicted_history
        ],
        module_c_classification=parsed.classification,
        module_c_aggregate_class=(
            "CORRECT"
            if parsed.classification == "CORRECT"
            else "CORRECTION"
            if parsed.classification == "CORRECTION"
            else "MISTAKE"
        ),
        module_c_mistake=parsed.classification != "CORRECT",
        module_c_error_score=error_score,
        module_c_class_probabilities=probabilities,
        module_c_candidate_nlls=(
            {str(key): float(value) for key, value in candidate_nlls.items()}
            if candidate_nlls is not None
            else None
        ),
        module_c_reasoning=parsed.reasoning,
        monolith_prediction_seconds=prediction_seconds,
        stage_status=(
            "C_SKIPPED_EXTRA"
            if parsed.classification == "ADDITION"
            else "C_CLASSIFIED"
        ),
        occurrence_id=occurrence_id,
        annotation_position=annotation_position,
    )
    return MonolithProjection(module_a, module_b, module_c, event)


project_prediction = project_monolith_prediction


@dataclass
class SequentialMonolithState:
    """State machine for realistic predicted-history streaming evaluation."""

    cursor: float = 0.0
    accumulation_start: float = 0.0
    max_history: int = 12
    polls: list[dict[str, Any]] = field(default_factory=list)
    events: list[PipelineEvent] = field(default_factory=list)
    _history: list[dict[str, Any]] = field(default_factory=list)

    @property
    def history(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self._history[-self.max_history :]]

    def should_poll(self, poll_time: float) -> bool:
        return float(poll_time) + 1e-9 >= self.cursor

    def consume(
        self,
        prediction: MonolithPrediction | str,
        *,
        video_id: str,
        task_id: str,
        instruction: str,
        window_end: float,
        prediction_seconds: float | None = None,
        raw_output: str | None = None,
        **projection_kwargs: Any,
    ) -> PipelineEvent | None:
        if not self.should_poll(window_end):
            self.polls.append(
                {"window_end": float(window_end), "disposition": "skipped_before_cursor"}
            )
            return None
        parsed = (
            parse_monolith_prediction(
                prediction,
                clip_duration=float(window_end) - self.accumulation_start,
            )
            if isinstance(prediction, str)
            else prediction
        )
        projection = project_monolith_prediction(
            parsed,
            video_id=video_id,
            task_id=task_id,
            instruction=instruction,
            window_start=self.accumulation_start,
            window_end=window_end,
            prediction_seconds=prediction_seconds,
            raw_output=(
                raw_output
                if raw_output is not None
                else prediction if isinstance(prediction, str) else None
            ),
            predicted_history=self.history,
            **projection_kwargs,
        )
        self.polls.append(
            {
                "window_start": self.accumulation_start,
                "window_end": float(window_end),
                "prediction": asdict(parsed),
                "history": self.history,
                "disposition": (
                    "completed" if parsed.valid and parsed.completed
                    else "retained" if parsed.valid
                    else "invalid"
                ),
            }
        )
        if projection.event is None:
            return None
        event = projection.event
        self.events.append(event)
        assert event.pred_global_end is not None
        self.cursor = float(event.pred_global_end)
        self.accumulation_start = self.cursor
        if parsed.classification != "ADDITION":
            self._history.append(
                {
                    "step_id": parsed.step_id,
                    "action": instruction,
                    "start": event.pred_global_start,
                    "end": event.pred_global_end,
                    "classification": parsed.classification,
                    "reasoning": parsed.reasoning,
                }
            )
            self._history = self._history[-self.max_history :]
        return event

    process_poll = consume


def scores_to_probabilities(candidate_nlls: Mapping[str, float]) -> dict[str, float]:
    if not candidate_nlls:
        return {}
    values = {str(key): float(value) for key, value in candidate_nlls.items()}
    if any(not math.isfinite(value) for value in values.values()):
        raise ValueError("candidate NLLs must be finite")
    best = min(values.values())
    weights = {key: math.exp(-(value - best)) for key, value in values.items()}
    total = sum(weights.values())
    return {key: value / total for key, value in weights.items()}


normalize_candidate_nlls = scores_to_probabilities


def score_coverage(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    covered = sum(
        isinstance(row.get("candidate_nlls"), Mapping)
        and bool(row.get("candidate_nlls"))
        for row in rows
    )
    total = len(rows)
    return {
        "count": float(total),
        "scored_count": float(covered),
        "coverage": covered / total if total else 1.0,
        "complete": covered == total,
    }


def monolith_runtime_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    streamed_video_seconds: float | None = None,
    gpu_count: int = 1,
) -> dict[str, float]:
    durations = [
        float(row["monolith_prediction_seconds"])
        for row in rows
        if _finite_number(row.get("monolith_prediction_seconds"))
    ]
    ordered = sorted(durations)

    def percentile(q: float) -> float:
        if not ordered:
            return 0.0
        position = (len(ordered) - 1) * q
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    completed = [
        float(row["monolith_prediction_seconds"])
        for row in rows
        if row.get("completed") is True
        and _finite_number(row.get("monolith_prediction_seconds"))
    ]
    valid = sum(bool(row.get("parse_valid")) for row in rows)
    video_minutes = (
        float(streamed_video_seconds) / 60.0
        if streamed_video_seconds is not None and streamed_video_seconds > 0
        else 0.0
    )
    return {
        "prediction_runtime/monolith/count": float(len(durations)),
        "prediction_runtime/monolith/mean_seconds": (
            sum(durations) / len(durations) if durations else 0.0
        ),
        "prediction_runtime/monolith/median_seconds": median(durations) if durations else 0.0,
        "prediction_runtime/monolith/p90_seconds": percentile(0.9),
        "prediction_runtime/monolith/p95_seconds": percentile(0.95),
        "prediction_runtime/monolith/min_seconds": min(durations) if durations else 0.0,
        "prediction_runtime/monolith/max_seconds": max(durations) if durations else 0.0,
        "prediction_runtime/monolith/completed_event_mean_seconds": (
            sum(completed) / len(completed) if completed else 0.0
        ),
        "prediction_runtime/monolith/parse_rate": valid / len(rows) if rows else 0.0,
        "prediction_runtime/monolith/gpu_seconds_per_streamed_video_minute": (
            sum(durations) * gpu_count / video_minutes if video_minutes else 0.0
        ),
        "prediction_runtime/monolith/calls_per_streamed_video_minute": (
            len(rows) / video_minutes if video_minutes else 0.0
        ),
    }


REQUIRED_METRIC_PATHS = (
    "module_a",
    "module_a_step_id/accuracy",
    "module_a_step_id/macro_f1",
    "module_a_step_id/weighted_f1",
    "module_a_step_id/macro_auc",
    "module_a_step_id/gate/auc",
    "module_a_step_id/score_coverage",
    "module_b",
    "module_c",
    "pipeline",
    "num_gt_events",
    "num_predictions",
    "num_duplicates",
    "duplicate_detection_rate",
    "boundary_mae/start_seconds",
    "boundary_mae/end_seconds",
    "boundary_mae/center_seconds",
    "boundary_mae/duration_seconds",
    "reasoning/non_empty_rate",
    "reasoning/lexical_overlap",
    "temporal/precision_at_iou_0.1",
    "temporal/recall_at_iou_0.3",
    "temporal/f1_at_iou_0.5",
    "temporal/f1_at_iou_0.7",
    "temporal/f1_at_iou_0.9",
    "missed_step_rate_at_iou_0.5",
    "module_c_fine",
    "module_c_aggregate",
    "end_to_end_fine",
    "end_to_end_aggregate",
    "egoper_standard.tas",
    "egoper_standard.mistake_binary",
    "egoper_standard.mistake_fine",
    "egoper_standard.mistake_binary.error_f1_at_iou_0.1",
    "egoper_standard.mistake_binary.error_f1_at_iou_0.2",
    "egoper_standard.mistake_binary.error_f1_at_iou_0.3",
    "egoper_standard.mistake_binary.error_f1_at_iou_0.5",
    "egoper_standard.mistake_binary.error_f1_at_iou_0.7",
    "egoper_standard.mistake_binary.error_f1_at_iou_0.9",
    "egoper_standard.mistake_fine.macro_f1_at_iou_0.1",
    "egoper_standard.mistake_fine.macro_f1_at_iou_0.2",
    "egoper_standard.mistake_fine.macro_f1_at_iou_0.3",
    "egoper_standard.mistake_fine.macro_f1_at_iou_0.5",
    "egoper_standard.mistake_fine.macro_f1_at_iou_0.7",
    "egoper_standard.mistake_fine.macro_f1_at_iou_0.9",
    "egoper_standard.eda",
    "egoper_standard.framewise_auc",
    "egoper_standard.framewise_auc.per_task",
    "egoper_standard.framewise_auc.task_average_auc",
    "egoper_standard.framewise_auc.prediction_score_coverage",
    "egoper_standard.segment_ap",
    "egoper_standard.segment_ap.by_threshold.0.1",
    "egoper_standard.segment_ap.by_threshold.0.2",
    "egoper_standard.segment_ap.by_threshold.0.3",
    "egoper_standard.segment_ap.by_threshold.0.5",
    "egoper_standard.segment_ap.per_task",
    "egoper_standard.segment_ap.binary_score_coverage",
    "egoper_standard.segment_ap.fine_score_coverage",
    "egoper_standard.omission",
    "structural",
    "structural/order_f1",
    "structural/confirmed_omission_recall",
    "structural/provisional_alert_count",
    "structural/revision_event_count",
    "structural/retracted_alert_count",
    "runtime",
    "prediction_runtime/monolith/count",
    "prediction_runtime/monolith/mean_seconds",
    "prediction_runtime/monolith/median_seconds",
    "prediction_runtime/monolith/p90_seconds",
    "prediction_runtime/monolith/p95_seconds",
    "prediction_runtime/monolith/min_seconds",
    "prediction_runtime/monolith/max_seconds",
    "prediction_runtime/monolith/completed_event_mean_seconds",
    "prediction_runtime/monolith/parse_rate",
    "prediction_runtime/monolith/gpu_seconds_per_streamed_video_minute",
    "prediction_runtime/monolith/calls_per_streamed_video_minute",
)
REQUIRED_METRIC_KEYS = REQUIRED_METRIC_PATHS


def _has_path(payload: Mapping[str, Any], path: str) -> bool:
    if path in payload:
        return True
    current: Any = payload
    parts = path.split(".")
    for index, part in enumerate(parts):
        remainder = ".".join(parts[index:])
        if isinstance(current, Mapping) and remainder in current:
            return True
        if not isinstance(current, Mapping) or part not in current:
            return False
        current = current[part]
    return True


def _has_metric_family(metrics: Mapping[str, Any], family: str) -> bool:
    if _has_path(metrics, family):
        return True
    prefixes = {
        "module_a": ("module_a/", "module_a_step_id/"),
        "module_b": ("module_b_", "module_b/"),
        "module_c": ("module_c_", "module_c/"),
        "pipeline": ("temporal/", "end_to_end_"),
        "structural": ("structural/",),
        "runtime": ("prediction_runtime/",),
    }.get(family, ())
    return any(
        isinstance(key, str) and key.startswith(prefix)
        for key in metrics
        for prefix in prefixes
    )


def validate_metric_manifest(
    metrics: Mapping[str, Any],
    *,
    required: Iterable[str] = REQUIRED_METRIC_PATHS,
    raise_on_missing: bool = True,
) -> list[str]:
    missing = [path for path in required if not _has_metric_family(metrics, path)]
    for path in (
        "module_a_step_id/score_coverage",
        "egoper_standard.framewise_auc.prediction_score_coverage",
        "egoper_standard.segment_ap.binary_score_coverage",
        "egoper_standard.segment_ap.fine_score_coverage",
    ):
        if _has_metric_family(metrics, path):
            current: Any = metrics
            if path in metrics:
                value = metrics[path]
            else:
                for part in path.split("."):
                    current = current[part]
                value = current
            if not _finite_number(value) or float(value) != 1.0:
                missing.append(f"{path}=1.0")
    if missing and raise_on_missing:
        raise ValueError("Incomplete monolith metric artifact; missing: " + ", ".join(missing))
    return missing


def build_monolith_artifact(
    *,
    polls: Sequence[Mapping[str, Any]],
    events: Sequence[PipelineEvent],
    ground_truth: Sequence[Mapping[str, Any] | Any],
    metrics: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
    module_a_predictions: Sequence[Mapping[str, Any]] | None = None,
    args: Mapping[str, Any] | None = None,
    split: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a JSON-round-trippable artifact accepted by offline consumers."""

    gt_rows = [
        asdict(row) if is_dataclass(row) else dict(row)
        for row in ground_truth
    ]
    return {
        "schema_version": "egoper_abc_monolith_v1",
        "metadata": dict(metadata or {}),
        "args": dict(args or {}),
        "split": dict(split or {}),
        "polls": [dict(row) for row in polls],
        "events": [
            asdict(event) if is_dataclass(event) else dict(event)
            for event in events
        ],
        "ground_truth": gt_rows,
        # Presence of this field selects the current non-legacy offline summary
        # path. Callers may pass fully native rows for stage-projected metrics.
        "module_a_predictions": [
            dict(row) for row in (module_a_predictions or ())
        ],
        "metrics": dict(metrics),
    }


aggregate_monolith_runtime = monolith_runtime_metrics
validate_required_metric_manifest = validate_metric_manifest
create_monolith_artifact = build_monolith_artifact

