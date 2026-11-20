from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Iterable, Literal

from ..prompts.contracts import is_not_completed_text


@dataclass(frozen=True)
class TemporalPrediction:
    start: float | None
    end: float | None
    rejection_reason: str | None = None
    source_format: str | None = None
    raw_start: float | None = None
    raw_end: float | None = None
    was_clamped: bool = False
    clamp_reason: str | None = None

    @property
    def valid(self) -> bool:
        return (
            self.start is not None
            and self.end is not None
            and self.rejection_reason is None
        )

    @property
    def span(self) -> tuple[float, float] | None:
        if not self.valid:
            return None
        return float(self.start), float(self.end)

    @property
    def raw_span(self) -> tuple[float, float] | None:
        if (
            self.raw_start is None
            or self.raw_end is None
            or not math.isfinite(self.raw_start)
            or not math.isfinite(self.raw_end)
            or self.raw_end <= self.raw_start
        ):
            return None
        return float(self.raw_start), float(self.raw_end)


def _extract_numbers(payload: Any) -> tuple[float, float] | None:
    if isinstance(payload, dict):
        windows = payload.get("relevant_windows")
        if isinstance(windows, list) and windows and isinstance(windows[0], (list, tuple)):
            if len(windows[0]) >= 2:
                return float(windows[0][0]), float(windows[0][1])
        pairs = (
            ("start_time", "end_time"),
            ("start", "end"),
            ("start_seconds", "end_seconds"),
        )
        for start_key, end_key in pairs:
            if start_key in payload and end_key in payload:
                return float(payload[start_key]), float(payload[end_key])
    return None


def parse_temporal_prediction(
    text: str,
    *,
    clip_duration: float | None = None,
    allow_legacy_text: bool = True,
) -> TemporalPrediction:
    """Parse once, preserve the raw span, and clamp it to physical clip bounds."""

    cleaned = text.strip()
    if not cleaned or is_not_completed_text(cleaned):
        return TemporalPrediction(None, None, "no_action", "sentinel")

    parsed: tuple[float, float] | None = None
    source = "json"
    candidates = [cleaned]
    object_match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if object_match and object_match.group(0) != cleaned:
        candidates.append(object_match.group(0))
    for candidate in candidates:
        try:
            parsed = _extract_numbers(json.loads(candidate))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if parsed is not None:
            break

    if parsed is None and allow_legacy_text:
        source = "legacy"
        numbers = re.findall(r"-?\d+(?:\.\d+)?", cleaned)
        if len(numbers) >= 2:
            parsed = float(numbers[0]), float(numbers[1])

    if parsed is None:
        return TemporalPrediction(None, None, "parse_error", None)

    start, end = parsed
    if not math.isfinite(start) or not math.isfinite(end):
        return TemporalPrediction(
            start,
            end,
            "non_finite",
            source,
            raw_start=start,
            raw_end=end,
        )
    if end <= start:
        reason = "zero_length" if end == start else "reversed"
        return TemporalPrediction(
            start,
            end,
            reason,
            source,
            raw_start=start,
            raw_end=end,
        )
    if clip_duration is not None:
        if not math.isfinite(clip_duration) or clip_duration <= 0.0:
            raise ValueError("clip_duration must be a finite positive number")
        clamped_start = min(max(start, 0.0), clip_duration)
        clamped_end = min(max(end, 0.0), clip_duration)
        was_clamped = clamped_start != start or clamped_end != end
        if clamped_end <= clamped_start:
            return TemporalPrediction(
                clamped_start,
                clamped_end,
                "collapsed_after_clamp",
                source,
                raw_start=start,
                raw_end=end,
                was_clamped=was_clamped,
                clamp_reason="out_of_bounds" if was_clamped else None,
            )
        return TemporalPrediction(
            clamped_start,
            clamped_end,
            None,
            source,
            raw_start=start,
            raw_end=end,
            was_clamped=was_clamped,
            clamp_reason="out_of_bounds" if was_clamped else None,
        )
    if start < 0.0:
        return TemporalPrediction(
            start,
            end,
            "negative_start",
            source,
            raw_start=start,
            raw_end=end,
        )
    return TemporalPrediction(
        start,
        end,
        None,
        source,
        raw_start=start,
        raw_end=end,
    )


def temporal_iou(
    first: tuple[float, float],
    second: tuple[float, float],
) -> float:
    intersection = max(0.0, min(first[1], second[1]) - max(first[0], second[0]))
    union = max(first[1], second[1]) - min(first[0], second[0])
    return intersection / union if union > 0 else 0.0


def temporal_metrics(
    targets: Iterable[tuple[float, float]],
    predictions: Iterable[TemporalPrediction | tuple[float, float] | None],
    *,
    prefix: str = "eval_temporal",
    view: Literal["clamped", "raw"] = "clamped",
) -> dict[str, float]:
    target_list = list(targets)
    prediction_list = list(predictions)
    if len(target_list) != len(prediction_list):
        raise ValueError("targets and predictions must have equal lengths")

    valid: list[tuple[tuple[float, float], tuple[float, float]]] = []
    rejected = 0
    clamped = 0
    for target, prediction in zip(target_list, prediction_list):
        if isinstance(prediction, TemporalPrediction):
            span = prediction.span if view == "clamped" else prediction.raw_span
            if prediction.was_clamped:
                clamped += 1
        else:
            span = prediction
        if span is None:
            rejected += 1
        else:
            valid.append((target, span))

    count = len(target_list)
    ious = [temporal_iou(target, prediction) for target, prediction in valid]
    start_errors = [abs(target[0] - prediction[0]) for target, prediction in valid]
    end_errors = [abs(target[1] - prediction[1]) for target, prediction in valid]
    return {
        f"{prefix}/num_samples": float(count),
        f"{prefix}/valid_count": float(len(valid)),
        f"{prefix}/invalid_rate": rejected / count if count else 0.0,
        f"{prefix}/clamped_rate": clamped / count if count else 0.0,
        f"{prefix}/start_mae": sum(start_errors) / len(start_errors) if start_errors else 0.0,
        f"{prefix}/end_mae": sum(end_errors) / len(end_errors) if end_errors else 0.0,
        f"{prefix}/mean_iou": sum(ious) / len(ious) if ious else 0.0,
        f"{prefix}/recall_at_0.5": (
            sum(iou >= 0.5 for iou in ious) / count if count else 0.0
        ),
    }


def temporal_metric_views(
    targets: Iterable[tuple[float, float]],
    predictions: Iterable[TemporalPrediction | tuple[float, float] | None],
    *,
    prefix: str = "eval_temporal",
) -> dict[str, float]:
    target_list = list(targets)
    prediction_list = list(predictions)
    return {
        **temporal_metrics(
            target_list,
            prediction_list,
            prefix=f"{prefix}/raw",
            view="raw",
        ),
        **temporal_metrics(
            target_list,
            prediction_list,
            prefix=f"{prefix}/clamped",
            view="clamped",
        ),
    }
