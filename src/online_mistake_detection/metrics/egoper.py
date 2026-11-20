from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Hashable, Iterable

from ..paths import egoper_root
from ..pipeline.events import GroundTruthEvent, PipelineEvent, predicted_step_index
from ..pipeline.task_graph import TaskGraph, parse_task_graphs
from .omission import infer_omissions


IOU_THRESHOLD = 0.5
EXTRA_STEP = "EXTRA"
ERROR_CLASSES = ("CORRECTION", "MODIFICATION", "SLIP", "ADDITION")
SEGMENT_F1_THRESHOLDS = (0.1, 0.2, 0.3, 0.5, 0.7, 0.9)
SEGMENT_AP_THRESHOLDS = (0.1, 0.2, 0.3, 0.5)
# Events a module rejected or could not parse are not system outputs.
NON_FINAL_STATUSES = frozenset({"B_NOT_COMPLETED", "C_NOT_COMPLETED", "C_INVALID_PARSE"})


@dataclass(frozen=True)
class _Segment:
    video_id: str
    task_id: str
    start: float
    end: float
    label: Hashable | None


def _label_matches(prediction: Hashable | None, target: Hashable | None) -> bool:
    if prediction is None:
        return False
    if isinstance(target, frozenset):
        return prediction in target
    return prediction == target


def _class_key(label: Hashable | None) -> Hashable | None:
    if isinstance(label, frozenset):
        return EXTRA_STEP
    return label


def _iou(left: _Segment, right: _Segment) -> float:
    intersection = max(0.0, min(left.end, right.end) - max(left.start, right.start))
    union = max(left.end, right.end) - min(left.start, right.start)
    return intersection / union if union > 0.0 else 0.0


def _f1(tp: int, fp: int, fn: int) -> float:
    denominator = (2 * tp) + fp + fn
    return (2 * tp) / denominator if denominator else 0.0


def _threshold_key(threshold: float) -> str:
    return f"{threshold:g}"


def _segment_f1(
    predictions: Iterable[_Segment],
    targets: Iterable[_Segment],
    *,
    threshold: float = IOU_THRESHOLD,
) -> dict[str, Any]:
    """One-to-one, same-label temporal segment F1 used by TAS evaluators."""

    predicted = sorted(
        predictions,
        key=lambda segment: (segment.video_id, segment.start, segment.end),
    )
    expected = list(targets)
    used_targets: set[int] = set()
    per_class: dict[Hashable, list[int]] = defaultdict(lambda: [0, 0, 0])
    tp = 0
    fp = 0

    for prediction in predicted:
        candidates = [
            (index, _iou(prediction, target))
            for index, target in enumerate(expected)
            if index not in used_targets
            and prediction.video_id == target.video_id
            and _label_matches(prediction.label, target.label)
        ]
        if candidates:
            target_index, best_iou = max(candidates, key=lambda row: (row[1], -row[0]))
        else:
            target_index, best_iou = -1, 0.0
        if best_iou >= threshold:
            used_targets.add(target_index)
            tp += 1
            per_class[prediction.label][0] += 1
        else:
            fp += 1
            if prediction.label is not None:
                per_class[prediction.label][1] += 1

    for index, target in enumerate(expected):
        if index not in used_targets:
            per_class[_class_key(target.label)][2] += 1
    fn = len(expected) - len(used_targets)

    threshold_key = _threshold_key(threshold)
    f1_key = f"f1_at_{threshold_key}"
    macro_f1_key = f"macro_f1_at_{threshold_key}"
    class_metrics = {
        str(label): {
            "tp": float(counts[0]),
            "fp": float(counts[1]),
            "fn": float(counts[2]),
            f1_key: _f1(*counts),
        }
        for label, counts in sorted(per_class.items(), key=lambda row: str(row[0]))
    }
    macro_f1 = (
        sum(row[f1_key] for row in class_metrics.values()) / len(class_metrics)
        if class_metrics
        else 0.0
    )
    return {
        "threshold": threshold,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        f1_key: _f1(tp, fp, fn),
        macro_f1_key: macro_f1,
        "per_class": class_metrics,
    }


def _with_task_breakdown(
    predictions: list[_Segment],
    targets: list[_Segment],
    *,
    threshold: float = IOU_THRESHOLD,
) -> dict[str, Any]:
    result = _segment_f1(predictions, targets, threshold=threshold)
    threshold_key = _threshold_key(threshold)
    f1_key = f"f1_at_{threshold_key}"
    macro_f1_key = f"macro_f1_at_{threshold_key}"
    task_ids = sorted({segment.task_id for segment in predictions + targets})
    per_task = {
        task_id: _segment_f1(
            [segment for segment in predictions if segment.task_id == task_id],
            [segment for segment in targets if segment.task_id == task_id],
            threshold=threshold,
        )
        for task_id in task_ids
    }
    result["per_task"] = per_task
    result[f"task_average_{f1_key}"] = (
        sum(row[f1_key] for row in per_task.values()) / len(per_task)
        if per_task
        else 0.0
    )
    result[f"task_average_{macro_f1_key}"] = (
        sum(row[macro_f1_key] for row in per_task.values()) / len(per_task)
        if per_task
        else 0.0
    )
    return result


def _with_threshold_breakdown(
    predictions: list[_Segment],
    targets: list[_Segment],
    thresholds: Iterable[float] = SEGMENT_F1_THRESHOLDS,
) -> dict[str, Any]:
    result = _with_task_breakdown(
        predictions,
        targets,
        threshold=IOU_THRESHOLD,
    )
    result["by_threshold"] = {
        _threshold_key(threshold): _with_task_breakdown(
            predictions,
            targets,
            threshold=threshold,
        )
        for threshold in thresholds
    }
    for threshold, values in result["by_threshold"].items():
        result[f"f1_at_iou_{threshold}"] = values[f"f1_at_{threshold}"]
        result[f"macro_f1_at_iou_{threshold}"] = values[
            f"macro_f1_at_{threshold}"
        ]
    return result


def _predicted_segments(events: Iterable[PipelineEvent]) -> list[PipelineEvent]:
    return [
        event
        for event in events
        if event.stage_status not in NON_FINAL_STATUSES
        and event.pred_global_start is not None
        and event.pred_global_end is not None
        and float(event.pred_global_end) > float(event.pred_global_start)
    ]


def _predicted_step_index(event: PipelineEvent) -> int | None:
    return predicted_step_index(event)


def _predicted_error_label(
    event: PipelineEvent,
    error_classes: tuple[str, ...] = ERROR_CLASSES,
) -> bool | None:
    if event.module_c_classification in error_classes:
        return True
    if event.module_c_classification == "CORRECT":
        return False
    return event.module_c_mistake


def _ground_truth_error_label(event: GroundTruthEvent) -> bool:
    if event.gt_fine_class is not None:
        return event.gt_fine_class != "CORRECT"
    return event.gt_mistake


@lru_cache(maxsize=1)
def _annotated_addition_steps() -> dict[tuple[str, float, float], int]:
    """Step index EgoPER's annotation gives each ADDITION segment, when it gives one."""

    path = egoper_root() / "annotation.json"
    if not path.is_file():
        return {}
    steps: dict[tuple[str, float, float], int] = {}
    for task_data in json.loads(path.read_text(encoding="utf-8")).values():
        type_names = {int(index): name for name, index in task_data.get("actiontype2idx", {}).items()}
        for video in task_data.get("segments", []):
            labels = video["labels"]
            for action, action_type, stamp in zip(
                labels["action"], labels["action_type"], labels["time_stamp"]
            ):
                if type_names.get(int(action_type)) == "Error_Addition" and int(action) > 0:
                    key = (str(video["video_id"]), round(float(stamp[0]), 1), round(float(stamp[1]), 1))
                    steps[key] = int(action) - 1
    return steps


def _step_target(event: GroundTruthEvent) -> Hashable | None:
    if event.step_index >= 0:
        return event.step_index
    if event.gt_fine_class != "ADDITION":
        return None
    step = _annotated_addition_steps().get(
        (event.video_id, round(event.gt_start, 1), round(event.gt_end, 1))
    )
    return frozenset({EXTRA_STEP} if step is None else {EXTRA_STEP, step})


def tas_f1(
    events: list[PipelineEvent],
    ground_truth: list[GroundTruthEvent],
    *,
    include_additions: bool = True,
    threshold: float = IOU_THRESHOLD,
) -> dict[str, Any]:
    """Segmental step F1."""

    predictions = []
    for event in _predicted_segments(events):
        step = _predicted_step_index(event)
        if step is None or (step < 0 and not include_additions):
            continue
        predictions.append(
            _Segment(
                event.video_id,
                event.task_id,
                float(event.pred_global_start),
                float(event.pred_global_end),
                step if step >= 0 else EXTRA_STEP,
            )
        )
    targets = [
        _Segment(event.video_id, event.task_id, event.gt_start, event.gt_end, label)
        for event in ground_truth
        if event.eligible
        and (label := _step_target(event)) is not None
        and (include_additions or event.step_index >= 0)
    ]
    return _with_task_breakdown(predictions, targets, threshold=threshold)


def binary_mistake_f1(
    events: list[PipelineEvent],
    ground_truth: list[GroundTruthEvent],
    *,
    error_classes: tuple[str, ...] = ERROR_CLASSES,
    thresholds: Iterable[float] = SEGMENT_F1_THRESHOLDS,
) -> dict[str, Any]:
    predictions = [
        _Segment(
            event.video_id,
            event.task_id,
            float(event.pred_global_start),
            float(event.pred_global_end),
            (
                "ERROR"
                if _predicted_error_label(event, error_classes) is True
                else "NORMAL"
                if _predicted_error_label(event, error_classes) is False
                else None
            ),
        )
        for event in _predicted_segments(events)
    ]
    targets = [
        _Segment(
            event.video_id,
            event.task_id,
            event.gt_start,
            event.gt_end,
            "ERROR" if _ground_truth_error_label(event) else "NORMAL",
        )
        for event in ground_truth
        if event.eligible
    ]
    result = _with_threshold_breakdown(predictions, targets, thresholds)
    result["error_f1_at_0.5"] = result["per_class"].get("ERROR", {}).get(
        "f1_at_0.5",
        0.0,
    )
    result["error_f1_by_threshold"] = {
        threshold: values["per_class"].get("ERROR", {}).get(
            f"f1_at_{threshold}",
            0.0,
        )
        for threshold, values in result["by_threshold"].items()
    }
    result.update(
        {
            f"error_f1_at_iou_{threshold}": value
            for threshold, value in result["error_f1_by_threshold"].items()
        }
    )
    task_error_f1s = [
        row["per_class"].get("ERROR", {}).get("f1_at_0.5", 0.0)
        for row in result["per_task"].values()
    ]
    result["task_average_error_f1_at_0.5"] = (
        sum(task_error_f1s) / len(task_error_f1s)
        if task_error_f1s
        else 0.0
    )
    return result


def fine_mistake_f1(
    events: list[PipelineEvent],
    ground_truth: list[GroundTruthEvent],
    *,
    error_classes: tuple[str, ...] = ERROR_CLASSES,
    thresholds: Iterable[float] = SEGMENT_F1_THRESHOLDS,
) -> dict[str, Any]:
    predictions = [
        _Segment(
            event.video_id,
            event.task_id,
            float(event.pred_global_start),
            float(event.pred_global_end),
            event.module_c_classification,
        )
        for event in _predicted_segments(events)
        if event.module_c_classification in error_classes
    ]
    targets = [
        _Segment(
            event.video_id,
            event.task_id,
            event.gt_start,
            event.gt_end,
            event.gt_fine_class,
        )
        for event in ground_truth
        if event.eligible and event.gt_fine_class in error_classes
    ]
    return _with_threshold_breakdown(predictions, targets, thresholds)


def _segment_error_coverage(
    target: GroundTruthEvent,
    predicted_errors: list[PipelineEvent],
) -> float:
    """Fraction of a ground-truth segment covered by predicted error intervals."""

    span = float(target.gt_end) - float(target.gt_start)
    if span <= 0.0:
        return 0.0
    pieces: list[tuple[float, float]] = []
    for prediction in predicted_errors:
        if prediction.video_id != target.video_id:
            continue
        start = max(float(prediction.pred_global_start), float(target.gt_start))
        end = min(float(prediction.pred_global_end), float(target.gt_end))
        if end > start:
            pieces.append((start, end))
    if not pieces:
        return 0.0
    pieces.sort()
    covered = 0.0
    cursor_start, cursor_end = pieces[0]
    for start, end in pieces[1:]:
        if start > cursor_end:
            covered += cursor_end - cursor_start
            cursor_start, cursor_end = start, end
        else:
            cursor_end = max(cursor_end, end)
    covered += cursor_end - cursor_start
    return covered / span


def _eda_from_votes(
    votes: list[tuple[str, bool]],
) -> dict[str, Any]:
    per_task_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    correct = 0
    for task_id, is_correct in votes:
        correct += int(is_correct)
        per_task_counts[task_id][0] += int(is_correct)
        per_task_counts[task_id][1] += 1
    total = len(votes)
    per_task = {
        task_id: {
            "correct": float(counts[0]),
            "total": float(counts[1]),
            "eda": counts[0] / counts[1] if counts[1] else 0.0,
        }
        for task_id, counts in sorted(per_task_counts.items())
    }
    return {
        "correct": float(correct),
        "total": float(total),
        "eda": correct / total if total else 0.0,
        "per_task": per_task,
        "task_average_eda": (
            sum(row["eda"] for row in per_task.values()) / len(per_task)
            if per_task
            else 0.0
        ),
    }


def error_detection_accuracy(
    events: list[PipelineEvent],
    ground_truth: list[GroundTruthEvent],
    *,
    error_classes: tuple[str, ...] = ERROR_CLASSES,
) -> dict[str, Any]:
    """Segment accuracy for normal versus error."""

    predicted_errors = [
        event
        for event in _predicted_segments(events)
        if _predicted_error_label(event, error_classes) is True
    ]
    any_votes: list[tuple[str, bool]] = []
    majority_votes: list[tuple[str, bool]] = []
    any_confusion = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    majority_confusion = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    for target in ground_truth:
        if not target.eligible:
            continue
        coverage = _segment_error_coverage(target, predicted_errors)
        truth = _ground_truth_error_label(target)
        any_prediction = coverage > 0.0
        majority_prediction = coverage > 0.5
        any_votes.append((target.task_id, any_prediction == truth))
        majority_votes.append((target.task_id, majority_prediction == truth))
        for prediction, confusion in (
            (any_prediction, any_confusion),
            (majority_prediction, majority_confusion),
        ):
            key = (
                "tp"
                if truth and prediction
                else "fn"
                if truth
                else "fp"
                if prediction
                else "tn"
            )
            confusion[key] += 1
    result = _eda_from_votes(any_votes)
    result["confusion"] = {
        key: float(value) for key, value in any_confusion.items()
    }
    result["majority"] = _eda_from_votes(majority_votes)
    result["majority"]["confusion"] = {
        key: float(value) for key, value in majority_confusion.items()
    }
    result["majority_coverage_threshold"] = 0.5
    return result


def _roc_auc(targets: list[int], scores: list[float]) -> float | None:
    positives = sum(targets)
    negatives = len(targets) - positives
    if not positives or not negatives:
        return None
    ordered = sorted(zip(scores, targets), key=lambda row: row[0])
    positive_rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        positive_rank_sum += average_rank * sum(label for _score, label in ordered[index:end])
        index = end
    return (
        positive_rank_sum - (positives * (positives + 1) / 2.0)
    ) / (positives * negatives)


def framewise_auc(
    events: list[PipelineEvent],
    ground_truth: list[GroundTruthEvent],
    *,
    fps: float = 4.0,
) -> dict[str, Any]:
    scored = [
        event
        for event in _predicted_segments(events)
        if event.module_c_error_score is not None
    ]
    valid_predictions = _predicted_segments(events)
    if not scored:
        return {
            "available": False,
            "auc": None,
            "fps": fps,
            "num_scored_predictions": 0.0,
            "prediction_score_coverage": 0.0,
            "reason": (
                "Module C emitted hard labels only. Framewise AUC requires an "
                "error probability or confidence for each predicted segment."
            ),
            "per_task": {},
        }
    if len(scored) != len(valid_predictions):
        return {
            "available": False,
            "auc": None,
            "fps": fps,
            "num_scored_predictions": float(len(scored)),
            "prediction_score_coverage": (
                len(scored) / len(valid_predictions)
                if valid_predictions
                else 0.0
            ),
            "reason": (
                "Framewise AUC requires scores for every predicted segment; "
                f"found {len(scored)}/{len(valid_predictions)}."
            ),
            "per_task": {},
        }

    video_ids = sorted({event.video_id for event in ground_truth})
    all_targets: list[int] = []
    all_scores: list[float] = []
    per_task_samples: dict[str, tuple[list[int], list[float]]] = {}
    for video_id in video_ids:
        video_gt = [
            event for event in ground_truth
            if event.video_id == video_id and event.eligible
        ]
        if not video_gt:
            continue
        task_id = video_gt[0].task_id
        video_predictions = [event for event in scored if event.video_id == video_id]
        end_time = max(
            [event.gt_end for event in video_gt]
            + [float(event.pred_global_end) for event in video_predictions]
        )
        frame_targets: list[int] = []
        frame_scores: list[float] = []
        for frame_index in range(max(1, math.ceil(end_time * fps))):
            timestamp = (frame_index + 0.5) / fps
            frame_targets.append(
                int(any(
                    _ground_truth_error_label(event)
                    and event.gt_start <= timestamp < event.gt_end
                    for event in video_gt
                ))
            )
            frame_scores.append(max(
                [
                    float(event.module_c_error_score)
                    for event in video_predictions
                    if float(event.pred_global_start) <= timestamp < float(event.pred_global_end)
                ],
                default=0.0,
            ))
        all_targets.extend(frame_targets)
        all_scores.extend(frame_scores)
        task_targets, task_scores = per_task_samples.setdefault(task_id, ([], []))
        task_targets.extend(frame_targets)
        task_scores.extend(frame_scores)

    per_task = {
        task_id: {"auc": _roc_auc(targets, scores), "num_frames": float(len(targets))}
        for task_id, (targets, scores) in sorted(per_task_samples.items())
    }
    task_aucs = [row["auc"] for row in per_task.values() if row["auc"] is not None]
    return {
        "available": True,
        "auc": _roc_auc(all_targets, all_scores),
        "fps": fps,
        "num_frames": float(len(all_targets)),
        "num_scored_predictions": float(len(scored)),
        "prediction_score_coverage": len(scored) / len(valid_predictions) if valid_predictions else 0.0,
        "per_task": per_task,
        "task_average_auc": sum(task_aucs) / len(task_aucs) if task_aucs else None,
    }


def _average_precision(
    predictions: list[tuple[_Segment, float, int | None]],
    targets: list[tuple[_Segment, int | None]],
    *,
    threshold: float,
    require_step: bool,
) -> float:
    """Ranked segment AP with deterministic, one-to-one greedy matching."""

    ranked = sorted(
        enumerate(predictions),
        key=lambda row: (
            -row[1][1],
            row[1][0].video_id,
            row[1][0].start,
            row[1][0].end,
            str(row[1][0].label),
            row[0],
        ),
    )
    used: set[int] = set()
    tp_flags: list[int] = []
    fp_flags: list[int] = []
    for _original_index, (prediction, _score, predicted_step) in ranked:
        candidates = [
            (target_index, _iou(prediction, target), target.start, target.end)
            for target_index, (target, target_step) in enumerate(targets)
            if target_index not in used
            and prediction.video_id == target.video_id
            and prediction.label == target.label
            and (not require_step or predicted_step == target_step)
        ]
        if not candidates:
            tp_flags.append(0)
            fp_flags.append(1)
            continue
        target_index, overlap, _start, _end = max(
            candidates,
            key=lambda row: (row[1], -row[2], -row[3], -row[0]),
        )
        if overlap >= threshold:
            used.add(target_index)
            tp_flags.append(1)
            fp_flags.append(0)
        else:
            tp_flags.append(0)
            fp_flags.append(1)
    if not targets:
        return 0.0
    cumulative_tp = 0
    cumulative_fp = 0
    recalls: list[float] = []
    precisions: list[float] = []
    for true_positive, false_positive in zip(tp_flags, fp_flags):
        cumulative_tp += true_positive
        cumulative_fp += false_positive
        recalls.append(cumulative_tp / len(targets))
        precisions.append(cumulative_tp / (cumulative_tp + cumulative_fp))
    # Standard all-points interpolated AP: integrate the monotonic precision
    # envelope over each recall increase.
    recall_curve = [0.0, *recalls, recalls[-1] if recalls else 0.0]
    precision_curve = [0.0, *precisions, 0.0]
    for index in range(len(precision_curve) - 1, 0, -1):
        precision_curve[index - 1] = max(
            precision_curve[index - 1], precision_curve[index]
        )
    return sum(
        (recall_curve[index] - recall_curve[index - 1])
        * precision_curve[index]
        for index in range(1, len(recall_curve))
        if recall_curve[index] != recall_curve[index - 1]
    )


def segment_average_precision(
    events: list[PipelineEvent],
    ground_truth: list[GroundTruthEvent],
    *,
    thresholds: tuple[float, ...] = SEGMENT_AP_THRESHOLDS,
    error_classes: tuple[str, ...] = ERROR_CLASSES,
) -> dict[str, Any]:
    """Segment-level binary AP and fine mistake-type mAP."""

    valid = _predicted_segments(events)
    binary_candidates = [
        event for event in valid if _predicted_error_label(event) is True
    ]
    binary_scored = [
        event
        for event in binary_candidates
        if event.module_c_error_score is not None
    ]
    binary_coverage = (
        len(binary_scored) / len(binary_candidates) if binary_candidates else 1.0
    )
    fine_candidates = [
        event for event in valid if event.module_c_classification in error_classes
    ]
    fine_scored = [
        event
        for event in fine_candidates
        if isinstance(event.module_c_class_probabilities, dict)
        and event.module_c_classification in event.module_c_class_probabilities
    ]
    fine_coverage = len(fine_scored) / len(fine_candidates) if fine_candidates else 1.0

    binary_targets = [
        (
            _Segment(gt.video_id, gt.task_id, gt.gt_start, gt.gt_end, "ERROR"),
            gt.step_index,
        )
        for gt in ground_truth
        if gt.eligible and _ground_truth_error_label(gt)
    ]
    binary_predictions = [
        (
            _Segment(
                event.video_id,
                event.task_id,
                float(event.pred_global_start),
                float(event.pred_global_end),
                "ERROR",
            ),
            float(event.module_c_error_score),
            _predicted_step_index(event),
        )
        for event in binary_scored
    ]
    fine_targets_by_class = {
        class_name: [
            (
                _Segment(gt.video_id, gt.task_id, gt.gt_start, gt.gt_end, class_name),
                gt.step_index,
            )
            for gt in ground_truth
            if gt.eligible and gt.gt_fine_class == class_name
        ]
        for class_name in error_classes
    }
    fine_predictions_by_class = {
        class_name: [
            (
                _Segment(
                    event.video_id,
                    event.task_id,
                    float(event.pred_global_start),
                    float(event.pred_global_end),
                    class_name,
                ),
                float(event.module_c_class_probabilities[class_name]),
                _predicted_step_index(event),
            )
            for event in fine_scored
            if event.module_c_classification == class_name
        ]
        for class_name in error_classes
    }

    def evaluate(
        selected_events: list[PipelineEvent],
        selected_gt: list[GroundTruthEvent],
    ) -> dict[str, Any]:
        selected_videos = {gt.video_id for gt in selected_gt}
        selected_binary_predictions = [
            row for row in binary_predictions if row[0].video_id in selected_videos
        ]
        selected_binary_targets = [
            row for row in binary_targets if row[0].video_id in selected_videos
        ]
        by_threshold: dict[str, Any] = {}
        for threshold in thresholds:
            key = f"{threshold:.1f}"
            per_class = {
                class_name: _average_precision(
                    [
                        row
                        for row in fine_predictions_by_class[class_name]
                        if row[0].video_id in selected_videos
                    ],
                    [
                        row
                        for row in fine_targets_by_class[class_name]
                        if row[0].video_id in selected_videos
                    ],
                    threshold=threshold,
                    require_step=True,
                )
                for class_name in error_classes
            }
            supported = [
                per_class[class_name]
                for class_name in error_classes
                if any(
                    row[0].video_id in selected_videos
                    for row in fine_targets_by_class[class_name]
                )
            ]
            by_threshold[key] = {
                "binary_ap": (
                    _average_precision(
                        selected_binary_predictions,
                        selected_binary_targets,
                        threshold=threshold,
                        require_step=False,
                    )
                    if binary_coverage == 1.0
                    else None
                ),
                "fine_map": (
                    sum(supported) / len(supported)
                    if supported and fine_coverage == 1.0
                    else None
                ),
                "per_class_ap": per_class if fine_coverage == 1.0 else {},
            }
        mean_keys = [f"{value:.1f}" for value in thresholds if value <= 0.3]
        binary_values = [by_threshold[key]["binary_ap"] for key in mean_keys]
        fine_values = [by_threshold[key]["fine_map"] for key in mean_keys]
        return {
            "by_threshold": by_threshold,
            "binary_ap": {
                key: values["binary_ap"] for key, values in by_threshold.items()
            },
            "fine_map": {
                key: values["fine_map"] for key, values in by_threshold.items()
            },
            "per_class_ap": {
                key: values["per_class_ap"] for key, values in by_threshold.items()
            },
            "binary_ap_mean_0.1_0.3": (
                sum(binary_values) / len(binary_values)
                if binary_values and all(value is not None for value in binary_values)
                else None
            ),
            "fine_map_mean_0.1_0.3": (
                sum(fine_values) / len(fine_values)
                if fine_values and all(value is not None for value in fine_values)
                else None
            ),
        }

    result = evaluate(events, ground_truth)
    task_ids = sorted({gt.task_id for gt in ground_truth if gt.eligible})
    per_task = {
        task_id: evaluate(
            [event for event in events if event.task_id == task_id],
            [gt for gt in ground_truth if gt.task_id == task_id],
        )
        for task_id in task_ids
    }
    result.update(
        {
            "available": binary_coverage == 1.0 and fine_coverage == 1.0,
            "binary_score_coverage": binary_coverage,
            "fine_score_coverage": fine_coverage,
            "num_predictions": float(len(valid)),
            "num_binary_candidates": float(len(binary_candidates)),
            "num_binary_scored": float(len(binary_scored)),
            "num_fine_candidates": float(len(fine_candidates)),
            "num_fine_scored": float(len(fine_scored)),
            "per_task": per_task,
            "task_average_binary_ap_mean_0.1_0.3": (
                sum(
                    row["binary_ap_mean_0.1_0.3"]
                    for row in per_task.values()
                    if row["binary_ap_mean_0.1_0.3"] is not None
                )
                / sum(
                    row["binary_ap_mean_0.1_0.3"] is not None
                    for row in per_task.values()
                )
                if any(
                    row["binary_ap_mean_0.1_0.3"] is not None
                    for row in per_task.values()
                )
                else None
            ),
            "task_average_fine_map_mean_0.1_0.3": (
                sum(
                    row["fine_map_mean_0.1_0.3"]
                    for row in per_task.values()
                    if row["fine_map_mean_0.1_0.3"] is not None
                )
                / sum(
                    row["fine_map_mean_0.1_0.3"] is not None
                    for row in per_task.values()
                )
                if any(
                    row["fine_map_mean_0.1_0.3"] is not None
                    for row in per_task.values()
                )
                else None
            ),
        }
    )
    return result


segment_ap = segment_average_precision
segment_ap_metrics = segment_average_precision


def _task_graphs(
    ground_truth: list[GroundTruthEvent],
) -> tuple[dict[str, TaskGraph], dict[str, set[int]], str]:
    graph_path = egoper_root() / "task_graph.txt"
    if graph_path.is_file():
        return parse_task_graphs(graph_path), {}, str(graph_path)
    inferred: dict[str, set[int]] = defaultdict(set)
    for event in ground_truth:
        if event.step_index >= 0:
            inferred[event.task_id].add(event.step_index)
    return {}, dict(inferred), "inferred_from_ground_truth"


def path_omission_metrics(
    events: list[PipelineEvent],
    ground_truth: list[GroundTruthEvent],
) -> dict[str, Any]:
    """Legacy start-to-end path diagnostic; not an official EgoPER metric."""

    graphs, fallback_expected, expected_source = _task_graphs(ground_truth)
    gt_by_video: dict[str, list[GroundTruthEvent]] = defaultdict(list)
    pred_by_video: dict[str, list[PipelineEvent]] = defaultdict(list)
    for event in ground_truth:
        if event.eligible:
            gt_by_video[event.video_id].append(event)
    for event in _predicted_segments(events):
        pred_by_video[event.video_id].append(event)

    rows: list[dict[str, Any]] = []
    for video_id, targets in sorted(gt_by_video.items()):
        task_id = targets[0].task_id
        observed_gt = [
            event.step_index
            for event in sorted(targets, key=lambda event: (event.gt_start, event.gt_end))
            if event.step_index >= 0
        ]
        observed_pred = [
            predicted_step
            for event in sorted(
                pred_by_video.get(video_id, []),
                key=lambda event: (
                    float(event.pred_global_start),
                    float(event.pred_global_end),
                ),
            )
            if (predicted_step := _predicted_step_index(event)) is not None
            and predicted_step >= 0
        ]
        graph = graphs.get(task_id)
        if graph is not None:
            gt_omitted = set(infer_omissions(observed_gt, graph).omitted_steps)
            pred_omitted = set(infer_omissions(observed_pred, graph).omitted_steps)
        else:
            expected = fallback_expected.get(task_id, set())
            gt_omitted = expected.difference(observed_gt)
            pred_omitted = expected.difference(observed_pred)
        intersection = gt_omitted.intersection(pred_omitted)
        union = gt_omitted.union(pred_omitted)
        rows.append({
            "video_id": video_id,
            "task_id": task_id,
            "has_gt_omission": bool(gt_omitted),
            "o_acc": len(intersection) / len(gt_omitted) if gt_omitted else 1.0,
            "o_iou": len(intersection) / len(union) if union else 1.0,
        })

    def summarize(selected: list[dict[str, Any]]) -> dict[str, float]:
        return {
            "num_videos": float(len(selected)),
            "o_acc": (
                sum(float(row["o_acc"]) for row in selected) / len(selected)
                if selected else 0.0
            ),
            "o_iou": (
                sum(float(row["o_iou"]) for row in selected) / len(selected)
                if selected else 0.0
            ),
        }

    per_task = {
        task_id: summarize([row for row in rows if row["task_id"] == task_id])
        for task_id in sorted({str(row["task_id"]) for row in rows})
    }
    true_omission_rows = [row for row in rows if row["has_gt_omission"]]
    result: dict[str, Any] = {
        **summarize(rows),
        "num_true_omission_videos": float(len(true_omission_rows)),
        "true_omission_videos": summarize(true_omission_rows),
        "per_task": per_task,
        "expected_steps_source": expected_source,
    }
    result["task_average_o_acc"] = (
        sum(row["o_acc"] for row in per_task.values()) / len(per_task)
        if per_task else 0.0
    )
    result["task_average_o_iou"] = (
        sum(row["o_iou"] for row in per_task.values()) / len(per_task)
        if per_task else 0.0
    )
    return result


def _set_difference_omission_rows(
    events: list[PipelineEvent],
    ground_truth: list[GroundTruthEvent],
) -> tuple[list[dict[str, Any]], str]:
    graphs, fallback_expected, expected_source = _task_graphs(ground_truth)
    gt_by_video: dict[str, list[GroundTruthEvent]] = defaultdict(list)
    pred_by_video: dict[str, list[PipelineEvent]] = defaultdict(list)
    for event in ground_truth:
        if event.eligible:
            gt_by_video[event.video_id].append(event)
    for event in _predicted_segments(events):
        pred_by_video[event.video_id].append(event)

    rows: list[dict[str, Any]] = []
    for video_id, targets in sorted(gt_by_video.items()):
        task_id = targets[0].task_id
        graph = graphs.get(task_id)
        if graph is not None:
            expected = {
                instruction
                for node in graph.recipe_nodes
                if (instruction := graph.instruction_index_for_node(node)) is not None
            }
        else:
            expected = fallback_expected.get(task_id, set())
        observed_gt = {
            event.step_index for event in targets if event.step_index >= 0
        }
        observed_pred = {
            predicted_step
            for event in pred_by_video.get(video_id, [])
            if (predicted_step := _predicted_step_index(event)) is not None
            and predicted_step >= 0
        }
        gt_omitted = expected.difference(observed_gt)
        pred_omitted = expected.difference(observed_pred)
        intersection = gt_omitted.intersection(pred_omitted)
        union = gt_omitted.union(pred_omitted)
        rows.append(
            {
                "video_id": video_id,
                "task_id": task_id,
                "has_gt_omission": bool(gt_omitted),
                "intersection": float(len(intersection)),
                "target": float(len(gt_omitted)),
                "union": float(len(union)),
                "o_acc": len(intersection) / len(gt_omitted) if gt_omitted else 1.0,
                "o_iou": len(intersection) / len(union) if union else 1.0,
            }
        )
    return rows, expected_source


def egoper_omission_metrics(
    events: list[PipelineEvent],
    ground_truth: list[GroundTruthEvent],
) -> dict[str, Any]:
    """Reproduce the released EgoPER per-video omission evaluator."""

    rows, expected_source = _set_difference_omission_rows(events, ground_truth)

    def summarize(selected: list[dict[str, Any]]) -> dict[str, float]:
        return {
            "num_videos": float(len(selected)),
            "o_acc": (
                sum(float(row["o_acc"]) for row in selected) / len(selected)
                if selected
                else 0.0
            ),
            "o_iou": (
                sum(float(row["o_iou"]) for row in selected) / len(selected)
                if selected
                else 0.0
            ),
        }

    task_ids = sorted({event.task_id for event in ground_truth if event.eligible})
    per_task = {
        task_id: summarize([row for row in rows if row["task_id"] == task_id])
        for task_id in task_ids
    }
    true_omission_rows = [row for row in rows if row["has_gt_omission"]]
    result: dict[str, Any] = {
        **summarize(rows),
        "num_true_omission_videos": float(len(true_omission_rows)),
        "true_omission_videos": summarize(true_omission_rows),
        "per_task": per_task,
        "expected_steps_source": expected_source,
        "aggregation": "mean of per-video scores over all supplied videos",
    }
    valid_task_rows = [row for row in per_task.values() if row["num_videos"] > 0]
    result["num_valid_tasks"] = float(len(valid_task_rows))
    result["task_average_o_acc"] = (
        sum(row["o_acc"] for row in valid_task_rows) / len(valid_task_rows)
        if valid_task_rows
        else 0.0
    )
    result["task_average_o_iou"] = (
        sum(row["o_iou"] for row in valid_task_rows) / len(valid_task_rows)
        if valid_task_rows
        else 0.0
    )
    return result


def omission_metrics(
    events: list[PipelineEvent],
    ground_truth: list[GroundTruthEvent],
) -> dict[str, Any]:
    """Official EgoPER omission metric retained under the standard API name."""

    return egoper_omission_metrics(events, ground_truth)


def gtg2vid_omission_metrics(
    events: list[PipelineEvent],
    ground_truth: list[GroundTruthEvent],
) -> dict[str, Any]:
    """Reproduce the released GTG2Vid omission evaluator."""

    all_rows, expected_source = _set_difference_omission_rows(events, ground_truth)
    rows = [row for row in all_rows if row["has_gt_omission"]]

    def summarize(selected: list[dict[str, Any]]) -> dict[str, float]:
        intersection = sum(float(row["intersection"]) for row in selected)
        target = sum(float(row["target"]) for row in selected)
        union = sum(float(row["union"]) for row in selected)
        return {
            "num_videos": float(len(selected)),
            "intersection": intersection,
            "target_omissions": target,
            "union": union,
            "o_acc": intersection / target if target else 0.0,
            "o_iou": intersection / union if union else 0.0,
        }

    task_ids = sorted({event.task_id for event in ground_truth if event.eligible})
    per_task = {
        task_id: summarize([row for row in rows if row["task_id"] == task_id])
        for task_id in task_ids
    }
    result: dict[str, Any] = {
        **summarize(rows),
        "per_task": per_task,
        "expected_steps_source": expected_source,
        "aggregation": "micro over videos with at least one GT omission",
    }
    valid_task_rows = [row for row in per_task.values() if row["num_videos"] > 0]
    result["num_valid_tasks"] = float(len(valid_task_rows))
    result["task_average_o_acc"] = (
        sum(row["o_acc"] for row in valid_task_rows) / len(valid_task_rows)
        if valid_task_rows
        else 0.0
    )
    result["task_average_o_iou"] = (
        sum(row["o_iou"] for row in valid_task_rows) / len(valid_task_rows)
        if valid_task_rows
        else 0.0
    )
    return result


def paper_omission_metrics(
    events: list[PipelineEvent],
    ground_truth: list[GroundTruthEvent],
) -> dict[str, Any]:
    """Backward-compatible name for the GTG2Vid protocol."""

    return gtg2vid_omission_metrics(events, ground_truth)


def egoper_standard_metrics(
    events: list[PipelineEvent],
    ground_truth: list[GroundTruthEvent],
    *,
    error_classes: tuple[str, ...] = ERROR_CLASSES,
) -> dict[str, Any]:
    """Metrics used for EgoPER pipeline comparison, kept independent by task."""

    tas = tas_f1(events, ground_truth)
    mistake_binary = binary_mistake_f1(
        events, ground_truth, error_classes=error_classes
    )
    mistake_fine = fine_mistake_f1(
        events, ground_truth, error_classes=error_classes
    )
    eda = error_detection_accuracy(
        events, ground_truth, error_classes=error_classes
    )
    auc = framewise_auc(events, ground_truth)
    segment_ap = segment_average_precision(
        events, ground_truth, error_classes=error_classes
    )
    omission = omission_metrics(events, ground_truth)
    omission_gtg2vid = gtg2vid_omission_metrics(events, ground_truth)
    return {
        "tas": tas,
        "mistake_binary": mistake_binary,
        "mistake_fine": mistake_fine,
        "eda": eda,
        "framewise_auc": auc,
        "segment_ap": segment_ap,
        "omission": omission,
        "omission_gtg2vid": omission_gtg2vid,
    }
