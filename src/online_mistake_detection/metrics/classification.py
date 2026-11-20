from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from ..prompts.contracts import NOT_COMPLETED_STATUS
from ..schemas import AggregateClass


@dataclass(frozen=True)
class ClassificationPrediction:
    classification: str | None
    reasoning: str | None
    valid: bool
    used_legacy_parser: bool = False
    classifications: tuple[str, ...] = ()
    reasoning_by_class: dict[str, tuple[str, ...]] = field(default_factory=dict)
    multi_label: bool = False


def _normalize_class(value: object, aliases: Mapping[str, str]) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = re.sub(r"[\s-]+", "_", value.strip()).upper()
    return aliases.get(normalized, normalized)


def parse_module_c_prediction(
    text: str,
    *,
    allowed_classes: Sequence[str],
    aliases: Mapping[str, str] | None = None,
    allow_legacy_binary: bool = False,
) -> ClassificationPrediction:
    canonical = {item.upper(): item for item in allowed_classes}
    normalized_aliases = {
        re.sub(r"[\s-]+", "_", key.strip()).upper(): value
        for key, value in (aliases or {}).items()
    }
    match = re.search(r"\{.*\}", text.strip(), flags=re.DOTALL)
    if not match:
        return ClassificationPrediction(None, None, False)
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return ClassificationPrediction(None, None, False)
    if not isinstance(payload, dict):
        return ClassificationPrediction(None, None, False)
    status = _normalize_class(payload.get("status"), {})
    classification = _normalize_class(payload.get("classification"), {})
    if status == NOT_COMPLETED_STATUS or classification == NOT_COMPLETED_STATUS:
        raw_reasoning = payload.get("reasoning")
        reasoning = (
            raw_reasoning.strip()
            if isinstance(raw_reasoning, str) and raw_reasoning.strip()
            else None
        )
        return ClassificationPrediction(NOT_COMPLETED_STATUS, reasoning, True)
    if "classifications" in payload:
        return _parse_multilabel_payload(payload, canonical, normalized_aliases)

    reasoning = payload.get("reasoning")
    if not isinstance(reasoning, str) or not reasoning.strip():
        return ClassificationPrediction(None, None, False)

    value = _normalize_class(payload.get("classification"), normalized_aliases)
    if value in canonical:
        return ClassificationPrediction(canonical[value], reasoning.strip(), True)

    if allow_legacy_binary and "mistake" in payload:
        raw = payload["mistake"]
        if isinstance(raw, str):
            raw = raw.strip().lower() in {"true", "1", "yes", "mistake"}
        if isinstance(raw, bool):
            legacy_class = "MISTAKE" if raw else "CORRECT"
            if legacy_class in canonical:
                return ClassificationPrediction(
                    canonical[legacy_class], reasoning.strip(), True, True
                )
    return ClassificationPrediction(None, reasoning.strip(), False)


def _parse_multilabel_payload(
    payload: dict[str, object],
    canonical: dict[str, str],
    aliases: Mapping[str, str],
) -> ClassificationPrediction:
    raw_labels = payload.get("classifications")
    if not isinstance(raw_labels, list) or not raw_labels:
        return ClassificationPrediction(None, None, False, multi_label=True)
    parsed: list[str] = []
    for item in raw_labels:
        value = _normalize_class(item, aliases)
        if value not in canonical:
            return ClassificationPrediction(None, None, False, multi_label=True)
        label = canonical[value]
        if label not in parsed:
            parsed.append(label)
    parsed = sorted(parsed)
    if "NO_VISUAL_ERROR" in parsed and len(parsed) > 1:
        return ClassificationPrediction(None, None, False, multi_label=True)
    raw_reasoning = payload.get("reasoning")
    reasoning_by_class: dict[str, tuple[str, ...]] = {}
    if isinstance(raw_reasoning, dict):
        for key, value in raw_reasoning.items():
            label = _normalize_class(key, aliases)
            if label not in canonical:
                continue
            if isinstance(value, str):
                descriptions = (value.strip(),) if value.strip() else ()
            elif isinstance(value, list):
                descriptions = tuple(
                    item.strip() for item in value if isinstance(item, str) and item.strip()
                )
            else:
                descriptions = ()
            reasoning_by_class[canonical[label]] = descriptions
    elif isinstance(raw_reasoning, str) and raw_reasoning.strip():
        if parsed and parsed != ["NO_VISUAL_ERROR"]:
            reasoning_by_class[parsed[0]] = (raw_reasoning.strip(),)
    if parsed != ["NO_VISUAL_ERROR"] and not reasoning_by_class:
        return ClassificationPrediction(
            parsed[0],
            None,
            False,
            classifications=tuple(parsed),
            multi_label=True,
        )
    reasoning_text = "; ".join(
        f"{label}: {'; '.join(descriptions)}"
        for label, descriptions in sorted(reasoning_by_class.items())
        if descriptions
    )
    return ClassificationPrediction(
        parsed[0],
        reasoning_text or None,
        True,
        classifications=tuple(parsed),
        reasoning_by_class=reasoning_by_class,
        multi_label=True,
    )


def multilabel_metrics(
    targets: Sequence[Sequence[str]],
    predictions: Sequence[Sequence[str] | None],
    *,
    classes: Sequence[str],
    prefix: str = "eval_multilabel",
) -> dict[str, object]:
    if len(targets) != len(predictions):
        raise ValueError("targets and predictions must have equal lengths")
    per_class: dict[str, dict[str, float]] = {}
    f1s: list[float] = []
    weighted_sum = 0.0
    total_support = 0
    micro_tp = micro_fp = micro_fn = 0
    exact = 0
    jaccard_sum = 0.0
    hamming_sum = 0.0
    invalid = 0
    any_tp = any_fp = any_fn = 0
    for target, prediction in zip(targets, predictions):
        target_set = {item for item in target if item in classes}
        if prediction is None:
            invalid += 1
            predicted_set: set[str] = set()
        else:
            predicted_set = {item for item in prediction if item in classes}
            if not predicted_set:
                invalid += 1
        exact += int(target_set == predicted_set)
        union = target_set | predicted_set
        jaccard_sum += (len(target_set & predicted_set) / len(union)) if union else 1.0
        hamming_sum += len(target_set ^ predicted_set) / len(classes) if classes else 0.0
        target_error = bool(target_set.difference({"NO_VISUAL_ERROR"}))
        predicted_error = bool(predicted_set.difference({"NO_VISUAL_ERROR"}))
        any_tp += int(target_error and predicted_error)
        any_fp += int(predicted_error and not target_error)
        any_fn += int(target_error and not predicted_error)
    for class_name in classes:
        tp = fp = fn = 0
        for target, prediction in zip(targets, predictions):
            in_target = class_name in target
            in_prediction = prediction is not None and class_name in prediction
            tp += int(in_target and in_prediction)
            fp += int(in_prediction and not in_target)
            fn += int(in_target and not in_prediction)
        support = sum(class_name in target for target in targets)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": float(support),
        }
        f1s.append(f1)
        weighted_sum += f1 * support
        total_support += support
        micro_tp += tp
        micro_fp += fp
        micro_fn += fn
    micro_precision = micro_tp / (micro_tp + micro_fp) if micro_tp + micro_fp else 0.0
    micro_recall = micro_tp / (micro_tp + micro_fn) if micro_tp + micro_fn else 0.0
    any_precision = any_tp / (any_tp + any_fp) if any_tp + any_fp else 0.0
    any_recall = any_tp / (any_tp + any_fn) if any_tp + any_fn else 0.0
    total = len(targets)
    present = [values["f1"] for values in per_class.values() if values["support"] > 0]
    return {
        f"{prefix}/per_class": per_class,
        f"{prefix}/macro_f1": sum(f1s) / len(f1s) if f1s else 0.0,
        f"{prefix}/present_class_macro_f1": sum(present) / len(present) if present else 0.0,
        f"{prefix}/weighted_f1": weighted_sum / total_support if total_support else 0.0,
        f"{prefix}/micro_precision": micro_precision,
        f"{prefix}/micro_recall": micro_recall,
        f"{prefix}/micro_f1": (
            2 * micro_precision * micro_recall / (micro_precision + micro_recall)
            if micro_precision + micro_recall
            else 0.0
        ),
        f"{prefix}/exact_set_accuracy": exact / total if total else 0.0,
        f"{prefix}/jaccard": jaccard_sum / total if total else 0.0,
        f"{prefix}/hamming_loss": hamming_sum / total if total else 0.0,
        f"{prefix}/invalid_rate": invalid / total if total else 0.0,
        f"{prefix}/any_visual_error_precision": any_precision,
        f"{prefix}/any_visual_error_recall": any_recall,
        f"{prefix}/any_visual_error_f1": (
            2 * any_precision * any_recall / (any_precision + any_recall)
            if any_precision + any_recall
            else 0.0
        ),
        f"{prefix}/num_samples": float(total),
    }


def _class_metrics(
    targets: Sequence[str],
    predictions: Sequence[str | None],
    classes: Sequence[str],
) -> dict[str, object]:
    confusion = {
        target: {prediction: 0 for prediction in [*classes, "INVALID"]}
        for target in classes
    }
    for target, prediction in zip(targets, predictions):
        if target not in confusion:
            continue
        confusion[target][prediction if prediction in classes else "INVALID"] += 1

    per_class: dict[str, dict[str, float]] = {}
    f1s: list[float] = []
    weighted_sum = 0.0
    for class_name in classes:
        tp = confusion[class_name][class_name]
        fp = sum(
            confusion[other][class_name]
            for other in classes
            if other != class_name
        )
        fn = sum(
            count
            for prediction, count in confusion[class_name].items()
            if prediction != class_name
        )
        support = sum(confusion[class_name].values())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_class[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": float(support),
        }
        f1s.append(f1)
        weighted_sum += f1 * support

    total = len(targets)
    present = [
        values["f1"] for values in per_class.values() if values["support"] > 0
    ]
    return {
        "per_class": per_class,
        "macro_f1": sum(f1s) / len(f1s) if f1s else 0.0,
        "present_class_macro_f1": (
            sum(present) / len(present) if present else 0.0
        ),
        "weighted_f1": weighted_sum / total if total else 0.0,
        "invalid_rate": (
            sum(prediction not in classes for prediction in predictions) / total
            if total
            else 0.0
        ),
        "confusion_matrix": confusion,
    }


def _binary_prf(truth: Sequence[bool], predicted: Sequence[bool]) -> dict[str, float]:
    tp = sum(t and p for t, p in zip(truth, predicted))
    fp = sum(not t and p for t, p in zip(truth, predicted))
    fn = sum(t and not p for t, p in zip(truth, predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": float(sum(truth)),
    }


def module_c_metrics(
    targets: Sequence[str],
    predictions: Sequence[str | None],
    *,
    fine_classes: Sequence[str],
    aggregate_mapping: Mapping[str, AggregateClass],
) -> dict[str, object]:
    if len(targets) != len(predictions):
        raise ValueError("targets and predictions must have equal lengths")
    not_completed = _binary_prf(
        [target == NOT_COMPLETED_STATUS for target in targets],
        [prediction == NOT_COMPLETED_STATUS for prediction in predictions],
    )
    all_classes = [*fine_classes, NOT_COMPLETED_STATUS]
    all_class = _class_metrics(targets, predictions, all_classes)
    completed_pairs = [
        (target, prediction)
        for target, prediction in zip(targets, predictions)
        if target != NOT_COMPLETED_STATUS
    ]
    completed_targets = [target for target, _prediction in completed_pairs]
    completed_predictions = [prediction for _target, prediction in completed_pairs]
    valid_completed = [
        (target, prediction)
        for target, prediction in completed_pairs
        if prediction not in {None, NOT_COMPLETED_STATUS}
    ]
    completed_fine = (
        _class_metrics(completed_targets, completed_predictions, fine_classes)
        if completed_targets
        else _class_metrics([], [], fine_classes)
    )
    fine = completed_fine
    conditional = _class_metrics(
        [target for target, _prediction in valid_completed],
        [prediction for _target, prediction in valid_completed],
        fine_classes,
    ) if valid_completed else _class_metrics([], [], fine_classes)
    aggregate_classes = ["MISTAKE", "CORRECTION", "CORRECT"]
    aggregate_targets = [
        aggregate_mapping[target] if target in aggregate_mapping else target
        for target in targets
        if target != NOT_COMPLETED_STATUS
    ]
    aggregate_predictions = [
        aggregate_mapping[prediction]
        if prediction is not None and prediction in aggregate_mapping
        else None
        for target, prediction in zip(targets, predictions)
        if target != NOT_COMPLETED_STATUS
    ]
    aggregate = _class_metrics(
        aggregate_targets, aggregate_predictions, aggregate_classes
    )

    true_mistake = [target == "MISTAKE" for target in aggregate_targets]
    pred_mistake = [prediction == "MISTAKE" for prediction in aggregate_predictions]
    tp = sum(t and p for t, p in zip(true_mistake, pred_mistake))
    fp = sum(not t and p for t, p in zip(true_mistake, pred_mistake))
    fn = sum(t and not p for t, p in zip(true_mistake, pred_mistake))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    legacy_f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "all": all_class,
        "fine": fine,
        "aggregate": aggregate,
        "completed": completed_fine,
        "completed_conditional": conditional,
        "not_completed": not_completed,
        "deprecated_binary": {
            "precision": precision,
            "recall": recall,
            "f1": legacy_f1,
        },
    }
