from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .task_graph import TaskGraph


@dataclass(frozen=True)
class DetectedOccurrence:
    occurrence_id: str
    instruction_index: int
    start: float | None
    end: float | None
    trigger_time: float
    localized: bool = False


@dataclass(frozen=True)
class StructuralAlert:
    kind: str
    step_index: int
    candidates: tuple[int, ...]
    status: str
    reason: str
    trigger_time: float
    occurrence_id: str = ""
    alert_id: str = ""


@dataclass
class DependencyGraphTracker:
    """Online order/omission tracker for dependency DAGs."""

    task_graph: TaskGraph
    occurrences: list[DetectedOccurrence] = field(default_factory=list)
    alerts: list[StructuralAlert] = field(default_factory=list)
    _alert_serial: int = 0

    def update(self, occurrence: DetectedOccurrence) -> list[StructuralAlert]:
        existing = next(
            (
                index
                for index, item in enumerate(self.occurrences)
                if item.occurrence_id == occurrence.occurrence_id
            ),
            None,
        )
        if existing is None:
            self.occurrences.append(occurrence)
        else:
            self.occurrences[existing] = occurrence
        return self.recompute(trigger_time=occurrence.trigger_time)

    def finalize(self, *, trigger_time: float) -> list[StructuralAlert]:
        confirmed: list[StructuralAlert] = []
        for alert in list(self.alerts):
            if alert.kind == "MISSING_STEP" and alert.status == "provisional":
                replacement = StructuralAlert(
                    kind=alert.kind,
                    step_index=alert.step_index,
                    candidates=alert.candidates,
                    status="confirmed",
                    reason="end_of_task",
                    trigger_time=trigger_time,
                    occurrence_id=alert.occurrence_id,
                    alert_id=alert.alert_id,
                )
                self._replace_alert(alert, replacement)
                confirmed.append(replacement)
            elif alert.kind == "ORDER_ERROR" and alert.status == "provisional":
                replacement = StructuralAlert(
                    kind=alert.kind,
                    step_index=alert.step_index,
                    candidates=alert.candidates,
                    status="confirmed",
                    reason="end_of_task",
                    trigger_time=trigger_time,
                    occurrence_id=alert.occurrence_id,
                    alert_id=alert.alert_id,
                )
                self._replace_alert(alert, replacement)
                confirmed.append(replacement)
        required = self._required_end_to_end()
        observed = self._satisfied_instructions()
        for step_index in sorted(required.difference(observed)):
            if any(
                alert.kind == "MISSING_STEP" and alert.step_index == step_index
                for alert in self.alerts
                if alert.status != "retracted"
            ):
                continue
            confirmed.append(
                self._emit(
                    kind="MISSING_STEP",
                    step_index=step_index,
                    candidates=(step_index,),
                    status="confirmed",
                    reason="end_of_task_unobserved_required_step",
                    trigger_time=trigger_time,
                )
            )
        return confirmed

    def recompute(self, *, trigger_time: float) -> list[StructuralAlert]:
        updates: list[StructuralAlert] = []
        satisfied = self._satisfied_instructions()
        for occurrence in self._ordered_occurrences():
            if occurrence.instruction_index < 0:
                continue
            prerequisites = self.task_graph.recipe_prerequisites(occurrence.instruction_index)
            missing = tuple(
                sorted(
                    step
                    for step in prerequisites
                    if step not in satisfied
                    or self._start_of(step) is None
                    or (
                        occurrence.start is not None
                        and self._start_of(step) is not None
                        and self._start_of(step) > occurrence.start
                    )
                )
            )
            order_id = f"order:{occurrence.occurrence_id}"
            if missing:
                pending_overlap = self._pending_overlap(occurrence)
                status = "provisional" if pending_overlap else "confirmed"
                reason = (
                    "unmet_prerequisites_pending_overlap"
                    if pending_overlap
                    else "unmet_prerequisites"
                )
                updates.append(
                    self._upsert(
                        alert_id=order_id,
                        kind="ORDER_ERROR",
                        step_index=occurrence.instruction_index,
                        candidates=missing,
                        status=status,
                        reason=reason,
                        trigger_time=trigger_time,
                        occurrence_id=occurrence.occurrence_id,
                    )
                )
            else:
                retracted = self._retract(order_id, trigger_time=trigger_time)
                if retracted is not None:
                    updates.append(retracted)
            if occurrence.start is not None or occurrence.localized:
                node = self.task_graph.node_for_instruction(occurrence.instruction_index)
                if node is None:
                    continue
                prerequisite_nodes = self.task_graph.ancestors(node)
                for ancestor in sorted(self.task_graph.recipe_nodes):
                    ancestor_index = self.task_graph.instruction_index_for_node(ancestor)
                    if ancestor_index is None or ancestor_index in satisfied:
                        continue
                    if ancestor not in prerequisite_nodes:
                        continue
                    missing_id = f"missing:{ancestor_index}:{occurrence.occurrence_id}"
                    updates.append(
                        self._upsert(
                            alert_id=missing_id,
                            kind="MISSING_STEP",
                            step_index=ancestor_index,
                            candidates=(ancestor_index,),
                            status="provisional",
                            reason="unobserved_prerequisite",
                            trigger_time=trigger_time,
                            occurrence_id=occurrence.occurrence_id,
                        )
                    )
        for alert in list(self.alerts):
            if alert.kind != "MISSING_STEP" or alert.status == "retracted":
                continue
            if alert.step_index in satisfied:
                retracted = self._retract(alert.alert_id, trigger_time=trigger_time)
                if retracted is not None:
                    updates.append(retracted)
        return updates

    def active_alerts(self) -> list[StructuralAlert]:
        return [alert for alert in self.alerts if alert.status != "retracted"]

    def fused_labels(self, visual_labels: SequenceLike) -> tuple[str, ...]:
        labels = [str(item) for item in visual_labels if item]
        for alert in self.active_alerts():
            if alert.status == "retracted":
                continue
            if alert.kind == "ORDER_ERROR" and "ORDER" not in labels:
                labels.append("ORDER")
            if alert.kind == "MISSING_STEP" and alert.status == "confirmed":
                if "MISSING_STEP" not in labels:
                    labels.append("MISSING_STEP")
        return tuple(labels)

    def _ordered_occurrences(self) -> list[DetectedOccurrence]:
        def sort_key(item: DetectedOccurrence) -> tuple[float, float]:
            start = item.start if item.start is not None else item.trigger_time
            return (start, item.trigger_time)

        unique: dict[int, DetectedOccurrence] = {}
        for item in sorted(self.occurrences, key=sort_key):
            if item.instruction_index < 0:
                continue
            unique.setdefault(item.instruction_index, item)
        return list(unique.values())

    def _satisfied_instructions(self) -> set[int]:
        return {
            item.instruction_index
            for item in self.occurrences
            if item.instruction_index >= 0
        }

    def _start_of(self, instruction_index: int) -> float | None:
        starts = [
            item.start
            for item in self.occurrences
            if item.instruction_index == instruction_index and item.start is not None
        ]
        return min(starts) if starts else None

    def _pending_overlap(self, occurrence: DetectedOccurrence) -> bool:
        if occurrence.start is None:
            return True
        for other in self.occurrences:
            if other.occurrence_id == occurrence.occurrence_id:
                continue
            if other.start is None:
                other_end = other.end if other.end is not None else other.trigger_time
                if other_end >= occurrence.start:
                    return True
        return False

    def _required_end_to_end(self) -> set[int]:
        required: set[int] = set()
        for node in self.task_graph.recipe_nodes:
            index = self.task_graph.instruction_index_for_node(node)
            if index is not None:
                required.add(index)
        return required

    def _emit(
        self,
        *,
        kind: str,
        step_index: int,
        candidates: tuple[int, ...],
        status: str,
        reason: str,
        trigger_time: float,
        occurrence_id: str = "",
    ) -> StructuralAlert:
        self._alert_serial += 1
        alert = StructuralAlert(
            kind=kind,
            step_index=step_index,
            candidates=candidates,
            status=status,
            reason=reason,
            trigger_time=trigger_time,
            occurrence_id=occurrence_id,
            alert_id=f"{kind.lower()}:{self._alert_serial}",
        )
        self.alerts.append(alert)
        return alert

    def _upsert(
        self,
        *,
        alert_id: str,
        kind: str,
        step_index: int,
        candidates: tuple[int, ...],
        status: str,
        reason: str,
        trigger_time: float,
        occurrence_id: str,
    ) -> StructuralAlert:
        existing = next((item for item in self.alerts if item.alert_id == alert_id), None)
        alert = StructuralAlert(
            kind=kind,
            step_index=step_index,
            candidates=candidates,
            status=status,
            reason=reason,
            trigger_time=trigger_time,
            occurrence_id=occurrence_id,
            alert_id=alert_id,
        )
        if existing is None:
            self.alerts.append(alert)
            return alert
        self._replace_alert(existing, alert)
        return alert

    def _retract(self, alert_id: str, *, trigger_time: float) -> StructuralAlert | None:
        existing = next((item for item in self.alerts if item.alert_id == alert_id), None)
        if existing is None or existing.status == "retracted":
            return None
        retracted = StructuralAlert(
            kind=existing.kind,
            step_index=existing.step_index,
            candidates=existing.candidates,
            status="retracted",
            reason="resolved_by_later_localization",
            trigger_time=trigger_time,
            occurrence_id=existing.occurrence_id,
            alert_id=existing.alert_id,
        )
        self._replace_alert(existing, retracted)
        return retracted

    def _replace_alert(self, existing: StructuralAlert, replacement: StructuralAlert) -> None:
        self.alerts = [
            replacement if item.alert_id == existing.alert_id else item
            for item in self.alerts
        ]


SequenceLike = Any


class DependencyGraphPipeline:
    def __init__(self) -> None:
        self._trackers: dict[str, DependencyGraphTracker] = {}

    def tracker(self, video_id: str, graph: TaskGraph) -> DependencyGraphTracker:
        if video_id not in self._trackers:
            self._trackers[video_id] = DependencyGraphTracker(task_graph=graph)
        return self._trackers[video_id]


def structural_metrics(
    *,
    predicted_alerts: list[StructuralAlert],
    target_order_steps: set[int],
    target_missing_steps: set[int],
    prefix: str = "eval_structural",
) -> dict[str, float]:
    predicted_order = {
        alert.step_index
        for alert in predicted_alerts
        if alert.kind == "ORDER_ERROR" and alert.status != "retracted"
    }
    predicted_missing = {
        alert.step_index
        for alert in predicted_alerts
        if alert.kind == "MISSING_STEP" and alert.status == "confirmed"
    }
    provisional = [
        alert
        for alert in predicted_alerts
        if alert.status == "provisional"
    ]
    confirmed_missing_hits = predicted_missing.intersection(target_missing_steps)
    order_tp = len(predicted_order.intersection(target_order_steps))
    order_fp = len(predicted_order.difference(target_order_steps))
    order_fn = len(target_order_steps.difference(predicted_order))
    order_precision = order_tp / (order_tp + order_fp) if order_tp + order_fp else 0.0
    order_recall = order_tp / (order_tp + order_fn) if order_tp + order_fn else 0.0
    missing_recall = (
        len(confirmed_missing_hits) / len(target_missing_steps)
        if target_missing_steps
        else 0.0
    )
    return {
        f"{prefix}/order_precision": order_precision,
        f"{prefix}/order_recall": order_recall,
        f"{prefix}/order_f1": (
            2 * order_precision * order_recall / (order_precision + order_recall)
            if order_precision + order_recall
            else 0.0
        ),
        f"{prefix}/provisional_alert_count": float(len(provisional)),
        f"{prefix}/confirmed_omission_recall": missing_recall,
        f"{prefix}/predicted_order_steps": float(len(predicted_order)),
        f"{prefix}/predicted_missing_steps": float(len(predicted_missing)),
    }
