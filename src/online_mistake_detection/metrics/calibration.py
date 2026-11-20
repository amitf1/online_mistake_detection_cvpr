"""Threshold-free scoring diagnostics and per-example calibration records."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..datasets.labels import (
    MODULE_A_STEP_ID_NONE_LETTER,
    is_module_a_completion_label,
)

CALIBRATION_FILE_STEM = "calibration"


def binary_roc_auc(y_true: list[int], y_score: list[float]) -> float | None:
    """Trapezoidal ROC-AUC for binary labels in {0,1}. Returns None if undefined."""
    if len(y_true) != len(y_score) or not y_true:
        return None
    positives = sum(y_true)
    negatives = len(y_true) - positives
    if positives == 0 or negatives == 0:
        return None
    ranked = sorted(zip(y_score, y_true), key=lambda item: item[0])
    tp = fp = 0
    prev_score = None
    points: list[tuple[float, float]] = [(0.0, 0.0)]
    for score, label in reversed(ranked):
        if prev_score is not None and score != prev_score:
            points.append((fp / negatives, tp / positives))
        if label == 1:
            tp += 1
        else:
            fp += 1
        prev_score = score
    points.append((fp / negatives, tp / positives))
    points.append((1.0, 1.0))
    auc = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        auc += (x1 - x0) * (y0 + y1) / 2.0
    return float(auc)


def scores_to_probabilities(scores: Mapping[str, float]) -> dict[str, float]:
    """Convert label NLLs to a simplex via softmax(-nll)."""
    return _softmax({label: -float(value) for label, value in scores.items()})


def logits_to_probabilities(
    logits: Sequence[float],
    labels: Sequence[str],
) -> dict[str, float]:
    """Convert candidate-letter logits to a simplex via softmax(+logit)."""
    if len(logits) != len(labels):
        raise ValueError(
            f"logits/labels length mismatch: {len(logits)} vs {len(labels)}."
        )
    return _softmax({label: float(value) for label, value in zip(labels, logits)})


def _softmax(values: Mapping[str, float]) -> dict[str, float]:
    if not values:
        return {}
    max_value = max(values.values())
    exps = {label: math.exp(value - max_value) for label, value in values.items()}
    total = sum(exps.values())
    if total <= 0:
        uniform = 1.0 / len(values)
        return {label: uniform for label in values}
    return {label: value / total for label, value in exps.items()}


def gate_score(probabilities: Mapping[str, float]) -> float:
    """Probability mass on every letter that means "a step completed"."""
    return float(
        sum(
            prob
            for letter, prob in probabilities.items()
            if letter != MODULE_A_STEP_ID_NONE_LETTER
        )
    )


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    position = q * (len(sorted_values) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return float(sorted_values[low])
    weight = position - low
    return float(sorted_values[low] * (1 - weight) + sorted_values[high] * weight)


def label_score_diagnostics(
    probabilities: Sequence[Mapping[str, float]],
    target_letters: Sequence[str],
) -> dict[str, float]:
    """Gate/per-class AUC plus the spread of the target-letter NLL."""
    if len(probabilities) != len(target_letters):
        raise ValueError(
            "probabilities/target_letters length mismatch: "
            f"{len(probabilities)} vs {len(target_letters)}."
        )
    if not probabilities:
        return {}

    binary_scores: list[float] = []
    binary_truth: list[int] = []
    class_scores: dict[str, list[float]] = {}
    class_truth: dict[str, list[int]] = {}
    target_nlls: list[float] = []
    all_letters: set[str] = set()

    for probs, target_letter in zip(probabilities, target_letters):
        all_letters.update(probs)
        binary_scores.append(gate_score(probs))
        binary_truth.append(
            1 if is_module_a_completion_label(target_letter, label_mode="step_id") else 0
        )
        target_probability = float(probs.get(target_letter, 0.0))
        target_nlls.append(
            -math.log(target_probability) if target_probability > 0.0 else float("inf")
        )
        for letter in sorted(set(probs) | {target_letter}):
            class_scores.setdefault(letter, []).append(float(probs.get(letter, 0.0)))
            class_truth.setdefault(letter, []).append(1 if target_letter == letter else 0)

    metrics: dict[str, float] = {}
    auc = binary_roc_auc(binary_truth, binary_scores)
    if auc is not None:
        metrics["eval_detection/auc"] = auc
        metrics["eval_step_id/gate/auc"] = auc
        metrics["eval_step_id/trigger/auc"] = auc

    per_class_aucs: list[float] = []
    for letter in sorted(all_letters):
        letter_auc = binary_roc_auc(class_truth[letter], class_scores[letter])
        if letter_auc is None:
            continue
        metrics[f"eval_step_id/auc_{letter}"] = letter_auc
        per_class_aucs.append(letter_auc)
    if per_class_aucs:
        metrics["eval_step_id/macro_auc"] = sum(per_class_aucs) / len(per_class_aucs)

    finite_nlls = sorted(value for value in target_nlls if math.isfinite(value))
    if finite_nlls:
        metrics["eval_step_id/target_nll_mean"] = sum(finite_nlls) / len(finite_nlls)
        metrics["eval_step_id/target_nll_median"] = _quantile(finite_nlls, 0.5)
        metrics["eval_step_id/target_nll_p90"] = _quantile(finite_nlls, 0.9)
    metrics["eval_step_id/target_nll_saturated_fraction"] = (
        sum(1 for value in target_nlls if not math.isfinite(value)) / len(target_nlls)
    )
    metrics["eval_step_id/gate_positive_rate"] = sum(binary_truth) / len(binary_truth)
    return metrics


def calibration_record(
    example: Mapping[str, Any],
    *,
    probabilities: Mapping[str, float],
    target_letter: str,
    prediction: str,
    row_index: int | None = None,
) -> dict[str, Any]:
    """One JSONL row: enough to sweep a threshold without the model."""
    return {
        "row_index": row_index,
        "video_id": example.get("video_id"),
        "task_id": example.get("task_id"),
        "step_index": example.get("step_index"),
        "window_start": example.get("window_start"),
        "window_end": example.get("window_end"),
        "target_letter": target_letter,
        "argmax_letter": prediction,
        "target_positive": bool(
            is_module_a_completion_label(target_letter, label_mode="step_id")
        ),
        "gate_score": gate_score(probabilities),
        "probabilities": {
            letter: float(prob) for letter, prob in sorted(probabilities.items())
        },
        "num_instructions": len(example.get("all_instructions") or []),
    }


def write_calibration_records(
    records: Sequence[Mapping[str, Any]],
    path: str | Path,
) -> Path:
    """Write one JSON object per line; overwrites any previous file at path."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, sort_keys=True))
            file.write("\n")
    return output_path


def calibration_output_path(
    output_dir: str | Path,
    *,
    global_step: int | None = None,
    tag: str | None = None,
) -> Path:
    parts = [CALIBRATION_FILE_STEM]
    if tag:
        parts.append(str(tag))
    if global_step is not None:
        parts.append(f"step{int(global_step)}")
    return Path(output_dir) / "calibration" / f"{'_'.join(parts)}.jsonl"


def sweep_thresholds(
    records: Sequence[Mapping[str, Any]],
    *,
    beta: float = 2.0,
) -> list[dict[str, float]]:
    """Precision/recall/F-beta at every distinct gate score in ``records``."""
    scored = sorted(
        (
            (float(record["gate_score"]), bool(record["target_positive"]))
            for record in records
        ),
        key=lambda item: item[0],
        reverse=True,
    )
    positives = sum(1 for _, truth in scored if truth)
    if not scored or positives == 0:
        return []
    beta_squared = beta * beta
    rows: list[dict[str, float]] = []
    tp = fp = 0
    for index, (score, truth) in enumerate(scored):
        if truth:
            tp += 1
        else:
            fp += 1
        next_score = scored[index + 1][0] if index + 1 < len(scored) else None
        if next_score is not None and math.isclose(next_score, score):
            continue
        precision = tp / (tp + fp)
        recall = tp / positives
        denominator = beta_squared * precision + recall
        fbeta = (
            (1 + beta_squared) * precision * recall / denominator
            if denominator > 0
            else 0.0
        )
        rows.append(
            {
                "threshold": score,
                "precision": precision,
                "recall": recall,
                "fbeta": fbeta,
                "predicted_positive": float(tp + fp),
            }
        )
    return rows
