from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from ..pipeline.task_graph import TaskGraph
from ..schemas import VideoRecord


def collapse_consecutive_duplicates(sequence: Iterable[int]) -> tuple[int, ...]:
    collapsed: list[int] = []
    for item in sequence:
        if not collapsed or item != collapsed[-1]:
            collapsed.append(item)
    return tuple(collapsed)


def edit_distance(first: Sequence[int], second: Sequence[int]) -> int:
    previous = list(range(len(second) + 1))
    for row, left in enumerate(first, start=1):
        current = [row]
        for column, right in enumerate(second, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (left != right),
                )
            )
        previous = current
    return previous[-1]


@dataclass(frozen=True)
class OmissionResult:
    omitted_steps: frozenset[int]
    tied_references: tuple[tuple[int, ...], ...]
    edit_distance: int


def infer_omissions(
    detected_steps: Iterable[int],
    task_graph: TaskGraph,
    *,
    reference_sequences: Iterable[Sequence[int]] | None = None,
) -> OmissionResult:
    detected = collapse_consecutive_duplicates(
        step for step in detected_steps if step >= 0
    )
    references = [
        tuple(reference)
        for reference in (
            reference_sequences
            if reference_sequences is not None
            else task_graph.recipe_paths()
        )
    ]
    if not references:
        return OmissionResult(frozenset(), (), 0)
    distances = [edit_distance(detected, reference) for reference in references]
    best_distance = min(distances)
    tied = tuple(
        reference
        for reference, distance in zip(references, distances)
        if distance == best_distance
    )
    detected_set = set(detected)
    omitted_per_reference = [
        set(reference).difference(detected_set) for reference in tied
    ]
    common = set.intersection(*omitted_per_reference) if omitted_per_reference else set()
    return OmissionResult(frozenset(common), tied, best_distance)


def derive_ground_truth_omissions(record: VideoRecord) -> OmissionResult:
    """Derive an EgoPER omission set from annotated observed events and its DAG."""

    if not isinstance(record.task_graph, TaskGraph):
        raise ValueError(f"{record.video_id}: record has no TaskGraph")
    observed = [
        segment.instruction_index
        for segment in record.segments
        if segment.instruction_index >= 0 and segment.event_type != "BACKGROUND"
    ]
    return infer_omissions(observed, record.task_graph)


def omission_metrics(
    predicted: Iterable[set[int] | frozenset[int]],
    targets: Iterable[set[int] | frozenset[int]],
) -> dict[str, float]:
    predictions = list(predicted)
    target_list = list(targets)
    if len(predictions) != len(target_list):
        raise ValueError("predicted and targets must have equal lengths")
    exact = 0
    ious: list[float] = []
    for prediction, target in zip(predictions, target_list):
        exact += prediction == target
        union = set(prediction).union(target)
        ious.append(
            len(set(prediction).intersection(target)) / len(union)
            if union
            else 1.0
        )
    count = len(predictions)
    return {
        "omission/num_videos": float(count),
        "omission/accuracy": exact / count if count else 0.0,
        "omission/mean_iou": sum(ious) / count if count else 0.0,
    }


def online_omission_metrics(
    predicted_possibilities: Iterable[Iterable[set[int] | frozenset[int]]],
    targets: Iterable[set[int] | frozenset[int]],
) -> dict[str, float]:
    """Score candidate omission alerts without penalizing unresolved branches."""

    predicted = [
        frozenset().union(*(frozenset(candidate) for candidate in possibilities))
        for possibilities in predicted_possibilities
    ]
    target_list = [frozenset(target) for target in targets]
    if len(predicted) != len(target_list):
        raise ValueError("predicted_possibilities and targets must have equal lengths")

    alert_tp = alert_fp = 0
    covered_targets: set[int] = set()
    all_targets: set[int] = set()
    for candidates, target in zip(predicted, target_list):
        all_targets.update(target)
        overlap = candidates.intersection(target)
        covered_targets.update(overlap)
        if not candidates:
            continue
        if overlap:
            alert_tp += 1
        else:
            alert_fp += 1
    alert_fn = len(all_targets.difference(covered_targets))
    precision = alert_tp / (alert_tp + alert_fp) if alert_tp + alert_fp else 0.0
    recall = len(covered_targets) / len(all_targets) if all_targets else 1.0
    return {
        "online_omission/num_alerts": float(sum(bool(items) for items in predicted)),
        "online_omission/alert_tp": float(alert_tp),
        "online_omission/alert_fp": float(alert_fp),
        "online_omission/step_fn": float(alert_fn),
        "online_omission/alert_precision": precision,
        "online_omission/step_recall": recall,
        "online_omission/step_f1": (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        ),
    }
