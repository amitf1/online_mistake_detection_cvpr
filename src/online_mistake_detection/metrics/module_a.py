"""Pure Module A generated-label metric calculations."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from ..datasets.labels import (
    MODULE_A_STEP_ID_EXTRA_LETTER,
    MODULE_A_STEP_ID_FIRST_INSTRUCTION_LETTER,
    MODULE_A_STEP_ID_NONE_LETTER,
    is_module_a_completion_label,
)

SEMANTIC_NONE = "NONE"
SEMANTIC_EXTRA = "EXTRA"
LEGACY_LABELS = ("WAIT", "COMPLETE")


def extract_module_a_label(text: str, *, label_mode: str = "legacy") -> str:
    if label_mode == "completion_set":
        from ..prompts.module_a import format_completion_set_target, parse_completion_set

        return format_completion_set_target(parse_completion_set(text))
    normalized = text.strip().upper()
    if label_mode == "step_id":
        return next((char for char in normalized if "A" <= char <= "Z"), "A")
    return next((label for label in LEGACY_LABELS if label in normalized), "WAIT")


def binary_detection_metrics(
    targets: list[str],
    predictions: list[str],
    *,
    prefix: str,
    beta: float = 2.0,
    label_mode: str = "legacy",
) -> dict[str, float]:
    counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    for target, prediction in zip(targets, predictions):
        target_positive = is_module_a_completion_label(target, label_mode=label_mode)
        predicted_positive = is_module_a_completion_label(prediction, label_mode=label_mode)
        counts[
            "tp" if target_positive and predicted_positive
            else "fp" if predicted_positive
            else "tn" if not target_positive
            else "fn"
        ] += 1
    tp, fp, tn, fn = (counts[key] for key in ("tp", "fp", "tn", "fn"))
    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    beta_squared = beta * beta
    return {
        f"{prefix}/accuracy": (tp + tn) / total if total else 0.0,
        f"{prefix}/precision": precision,
        f"{prefix}/recall": recall,
        f"{prefix}/f1": f1,
        f"{prefix}/f_beta": (
            (1 + beta_squared) * precision * recall / (beta_squared * precision + recall)
            if precision + recall else 0.0
        ),
        f"{prefix}/specificity": specificity,
        f"{prefix}/balanced_accuracy": (recall + specificity) / 2.0,
        f"{prefix}/mcc": (
            (tp * tn - fp * fn)
            / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) else 0.0
        ),
        **{f"{prefix}/{key}": float(value) for key, value in counts.items()},
        f"{prefix}/num_samples": float(total),
    }


def multiclass_letter_metrics(
    targets: list[str],
    predictions: list[str],
    *,
    prefix: str = "eval_step_id",
    include_class_count_scalars: bool = False,
) -> dict[str, Any]:
    target_labels = [extract_module_a_label(value, label_mode="step_id") for value in targets]
    prediction_labels = [extract_module_a_label(value, label_mode="step_id") for value in predictions]
    labels = sorted(set(target_labels) | set(prediction_labels))
    metrics: dict[str, Any] = {
        f"{prefix}/accuracy": (
            sum(target == prediction for target, prediction in zip(target_labels, prediction_labels))
            / len(target_labels) if target_labels else 0.0
        ),
        f"{prefix}/num_samples": float(len(target_labels)),
        f"{prefix}/num_classes": float(len(labels)),
    }
    f1_values: list[float] = []
    weighted_f1 = support_total = 0
    micro_tp = micro_fp = micro_fn = 0
    for label in labels:
        tp = sum(target == label and prediction == label for target, prediction in zip(target_labels, prediction_labels))
        fp = sum(target != label and prediction == label for target, prediction in zip(target_labels, prediction_labels))
        fn = sum(target == label and prediction != label for target, prediction in zip(target_labels, prediction_labels))
        support = sum(target == label for target in target_labels)
        predicted_count = sum(prediction == label for prediction in prediction_labels)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        metrics.update({
            f"{prefix}/precision_{label}": precision,
            f"{prefix}/recall_{label}": recall,
            f"{prefix}/f1_{label}": f1,
        })
        if include_class_count_scalars:
            metrics.update({
                f"{prefix}/support_{label}": float(support),
                f"{prefix}/pred_{label}": float(predicted_count),
                f"{prefix}/tp_{label}": float(tp),
            })
        if support:
            f1_values.append(f1)
            weighted_f1 += f1 * support
            support_total += support
        micro_tp += tp
        micro_fp += fp
        micro_fn += fn
    micro_precision = micro_tp / (micro_tp + micro_fp) if micro_tp + micro_fp else 0.0
    micro_recall = micro_tp / (micro_tp + micro_fn) if micro_tp + micro_fn else 0.0
    metrics.update({
        f"{prefix}/macro_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        f"{prefix}/weighted_f1": weighted_f1 / support_total if support_total else 0.0,
        f"{prefix}/micro_f1": (
            2 * micro_precision * micro_recall / (micro_precision + micro_recall)
            if micro_precision + micro_recall else 0.0
        ),
        f"{prefix}/micro_precision": micro_precision,
        f"{prefix}/micro_recall": micro_recall,
        "confusion_matrix": {
            "labels": labels,
            "matrix": [
                [
                    sum(target == true_label and prediction == predicted_label for target, prediction in zip(target_labels, prediction_labels))
                    for predicted_label in labels
                ]
                for true_label in labels
            ],
        },
    })
    metrics.update(binary_detection_metrics(
        target_labels, prediction_labels, prefix=f"{prefix}/gate", label_mode="step_id"
    ))
    return metrics


def step_id_semantic_metrics(
    *,
    task_ids: list[str],
    all_instructions: list[list[str] | tuple[str, ...]],
    targets: list[str],
    predictions: list[str],
    prefix: str = "eval_step_id",
    beta: float = 2.0,
    include_class_count_scalars: bool = False,
) -> dict[str, Any]:
    """Average per-task F1 over classes with ground-truth support."""
    if not (len(task_ids) == len(all_instructions) == len(targets) == len(predictions)):
        raise ValueError("step_id_semantic_metrics requires aligned input lists.")

    def semantic(label: str, instructions: list[str] | tuple[str, ...]) -> str:
        letter = extract_module_a_label(label, label_mode="step_id")
        if letter == MODULE_A_STEP_ID_NONE_LETTER:
            return SEMANTIC_NONE
        if letter == MODULE_A_STEP_ID_EXTRA_LETTER:
            return SEMANTIC_EXTRA
        index = ord(letter) - ord(MODULE_A_STEP_ID_FIRST_INSTRUCTION_LETTER)
        return f"STEP_{index}" if 0 <= index < len(instructions) else f"UNKNOWN_{letter}"

    semantic_targets = [semantic(label, instructions) for label, instructions in zip(targets, all_instructions)]
    semantic_predictions = [semantic(label, instructions) for label, instructions in zip(predictions, all_instructions)]
    groups: dict[str, list[int]] = defaultdict(list)
    for index, task_id in enumerate(task_ids):
        groups[task_id].append(index)
    metrics: dict[str, Any] = {
        f"{prefix}/num_samples": float(len(targets)),
        f"{prefix}/num_tasks": float(len(groups)),
    }
    task_f1s: list[float] = []
    task_confusions: dict[str, Any] = {}
    for task_id, indices in sorted(groups.items()):
        task_targets = [semantic_targets[index] for index in indices]
        task_predictions = [semantic_predictions[index] for index in indices]
        instructions = all_instructions[indices[0]]
        labels = [SEMANTIC_NONE, SEMANTIC_EXTRA, *(f"STEP_{index}" for index in range(len(instructions)))]
        labels += sorted((set(task_targets) | set(task_predictions)) - set(labels))
        task_prefix = f"{prefix}/task_{task_id}"
        supported_class_f1s: list[float] = []
        for label in labels:
            tp = sum(target == label and prediction == label for target, prediction in zip(task_targets, task_predictions))
            fp = sum(target != label and prediction == label for target, prediction in zip(task_targets, task_predictions))
            fn = sum(target == label and prediction != label for target, prediction in zip(task_targets, task_predictions))
            support = tp + fn
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            if support > 0:
                supported_class_f1s.append(f1)
            metrics.update({
                f"{task_prefix}/precision_{label}": precision,
                f"{task_prefix}/recall_{label}": recall,
                f"{task_prefix}/f1_{label}": f1,
            })
            if include_class_count_scalars:
                metrics[f"{task_prefix}/support_{label}"] = float(support)
        task_f1s.append(
            sum(supported_class_f1s) / len(supported_class_f1s)
            if supported_class_f1s
            else 0.0
        )
        metrics[f"{task_prefix}/macro_f1"] = task_f1s[-1]
        metrics[f"{task_prefix}/num_samples"] = float(len(indices))
        metrics[f"{task_prefix}/num_classes"] = float(len(labels))
        metrics[f"{task_prefix}/num_supported_classes"] = float(
            len(supported_class_f1s)
        )
        task_confusions[task_id] = {
            "labels": [SEMANTIC_NONE, SEMANTIC_EXTRA, *(str(item) for item in instructions)],
            "label_keys": labels,
            "matrix": [
                [sum(target == left and prediction == right for target, prediction in zip(task_targets, task_predictions)) for right in labels]
                for left in labels
            ],
        }
    total_macro = sum(task_f1s) / len(task_f1s) if task_f1s else 0.0
    metrics[f"{prefix}/total_macro_f1"] = total_macro
    metrics[f"{prefix}/macro_f1"] = total_macro
    metrics["task_confusion_matrices"] = task_confusions
    trigger = binary_detection_metrics(targets, predictions, prefix=f"{prefix}/trigger", beta=beta, label_mode="step_id")
    metrics.update(trigger)
    metrics.update({
        key.replace(f"{prefix}/trigger", f"{prefix}/gate"): value
        for key, value in trigger.items()
    })
    return metrics


def _as_completion_set(value: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, (list, tuple, set)):
        return tuple(sorted({str(item).strip().upper() for item in value if str(item).strip()}))
    from ..prompts.module_a import parse_completion_set

    return tuple(sorted(set(parse_completion_set(str(value)))))


def completion_set_metrics(
    targets: list[str | Sequence[str]],
    predictions: list[str | Sequence[str]],
    *,
    prefix: str = "eval_completion_set",
    collision_masks: Sequence[bool] | None = None,
) -> dict[str, float]:
    if len(targets) != len(predictions):
        raise ValueError("targets and predictions must have equal lengths")
    target_sets = [_as_completion_set(item) for item in targets]
    predicted_sets = [_as_completion_set(item) for item in predictions]
    trigger_targets = ["COMPLETE" if item else "WAIT" for item in target_sets]
    trigger_predictions = ["COMPLETE" if item else "WAIT" for item in predicted_sets]
    metrics = binary_detection_metrics(
        trigger_targets,
        trigger_predictions,
        prefix=f"{prefix}/trigger",
        label_mode="legacy",
    )
    exact = [
        target == prediction
        for target, prediction in zip(target_sets, predicted_sets)
    ]
    metrics[f"{prefix}/exact_set_accuracy"] = sum(exact) / len(exact) if exact else 0.0
    labels = sorted({label for item in target_sets + predicted_sets for label in item})
    per_label_f1: list[float] = []
    micro_tp = micro_fp = micro_fn = 0
    for label in labels:
        tp = fp = fn = 0
        for target, prediction in zip(target_sets, predicted_sets):
            in_target = label in target
            in_prediction = label in prediction
            tp += int(in_target and in_prediction)
            fp += int(in_prediction and not in_target)
            fn += int(in_target and not in_prediction)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label_f1.append(f1)
        micro_tp += tp
        micro_fp += fp
        micro_fn += fn
        metrics[f"{prefix}/precision_{label}"] = precision
        metrics[f"{prefix}/recall_{label}"] = recall
        metrics[f"{prefix}/f1_{label}"] = f1
    micro_precision = micro_tp / (micro_tp + micro_fp) if micro_tp + micro_fp else 0.0
    micro_recall = micro_tp / (micro_tp + micro_fn) if micro_tp + micro_fn else 0.0
    metrics[f"{prefix}/micro_precision"] = micro_precision
    metrics[f"{prefix}/micro_recall"] = micro_recall
    metrics[f"{prefix}/micro_f1"] = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall
        else 0.0
    )
    metrics[f"{prefix}/macro_f1"] = (
        sum(per_label_f1) / len(per_label_f1) if per_label_f1 else 0.0
    )
    metrics[f"{prefix}/occurrence_recall"] = (
        micro_tp / (micro_tp + micro_fn) if micro_tp + micro_fn else 0.0
    )
    collisions = [
        (target, prediction, collision)
        for target, prediction, collision in zip(
            target_sets,
            predicted_sets,
            collision_masks or [len(item) > 1 for item in target_sets],
        )
        if collision
    ]
    metrics[f"{prefix}/collision_bucket_recall"] = (
        sum(1 for target, prediction, _ in collisions if set(target).intersection(prediction))
        / len(collisions)
        if collisions
        else 0.0
    )
    metrics[f"{prefix}/num_samples"] = float(len(target_sets))
    return metrics


def negative_sampling_metrics(
    stats: Mapping[str, Any] | None,
    *,
    prefix: str = "eval",
) -> dict[str, float]:
    """Report train-only negative-cap counts and that validation stays unsampled."""

    payload = dict(stats or {})
    natural = float(payload.get("natural_negative_windows", 0.0))
    sampled = float(payload.get("sampled_negative_windows", 0.0))
    validation_sampled = bool(payload.get("validation_windows_sampled", False))
    return {
        f"{prefix}/natural_negative_windows": natural,
        f"{prefix}/sampled_negative_windows": sampled,
        f"{prefix}/affected_episodes": float(payload.get("affected_episodes", 0.0)),
        f"{prefix}/negative_drop_rate": (1.0 - sampled / natural) if natural else 0.0,
        f"{prefix}/full_window_validation": 0.0 if validation_sampled else 1.0,
    }
