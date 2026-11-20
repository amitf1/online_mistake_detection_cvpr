from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .task_graph import TaskGraph


@dataclass(frozen=True)
class OnlineOmissionUpdate:
    """Omission possibilities emitted after a newly completed Module A step."""

    observed_steps: tuple[int, ...]
    anchor_step: int | None
    detected_step: int
    possibilities: tuple[frozenset[int], ...]
    confirmed_omitted_steps: frozenset[int]
    transition_valid: bool

    @property
    def candidate_omitted_steps(self) -> frozenset[int]:
        return frozenset().union(*self.possibilities) if self.possibilities else frozenset()


class OnlineOmissionTracker:
    """Maintain online omissions for an all-nodes dependency DAG."""

    def __init__(self, task_graph: TaskGraph) -> None:
        self.task_graph = task_graph
        self._observed: list[int] = []

    @property
    def observed_steps(self) -> tuple[int, ...]:
        return tuple(self._observed)

    def update(self, detected_step: int) -> OnlineOmissionUpdate:
        if detected_step < 0:
            raise ValueError("Online omission tracking requires a non-negative STEP_k index.")
        if self._observed and detected_step == self._observed[-1]:
            return self._result(
                anchor_step=self._observed[-1],
                detected_step=detected_step,
                possibilities=(),
                transition_valid=True,
            )

        node = self.task_graph.node_for_instruction(detected_step)
        if node not in self.task_graph.recipe_nodes:
            return self._result(
                anchor_step=self._observed[-1] if self._observed else None,
                detected_step=detected_step,
                possibilities=(),
                transition_valid=False,
            )

        anchor = self._observed[-1] if self._observed else None
        missing = frozenset(
            self.task_graph.recipe_prerequisites(detected_step).difference(
                self._observed
            )
        )
        self._observed.append(detected_step)
        return self._result(
            anchor_step=anchor,
            detected_step=detected_step,
            possibilities=(missing,) if missing else (),
            transition_valid=True,
        )

    def finalize(self) -> OnlineOmissionUpdate | None:
        """Return possible remaining omissions after a reliable task-end signal."""

        if not self._observed:
            return None
        anchor = self._observed[-1]
        expected = {
            instruction
            for node in self.task_graph.recipe_nodes
            if (instruction := self.task_graph.instruction_index_for_node(node))
            is not None
        }
        omitted = frozenset(expected.difference(self._observed))
        possibilities = (omitted,) if omitted else ()
        return self._result(
            anchor_step=anchor,
            detected_step=anchor,
            possibilities=possibilities,
            transition_valid=True,
            confirmed=omitted,
        )

    def _result(
        self,
        *,
        anchor_step: int | None,
        detected_step: int,
        possibilities: tuple[frozenset[int], ...],
        transition_valid: bool,
        confirmed: frozenset[int] = frozenset(),
    ) -> OnlineOmissionUpdate:
        return OnlineOmissionUpdate(
            observed_steps=tuple(self._observed),
            anchor_step=anchor_step,
            detected_step=detected_step,
            possibilities=possibilities,
            confirmed_omitted_steps=confirmed,
            transition_valid=transition_valid,
        )


class OnlineOmissionPipeline:
    """Per-video tracker registry called after each completed Module A STEP_k."""

    def __init__(self, task_graphs: Mapping[str, TaskGraph]) -> None:
        self.task_graphs = {task_id.lower(): graph for task_id, graph in task_graphs.items()}
        self._trackers: dict[str, OnlineOmissionTracker] = {}
        self._task_ids: dict[str, str] = {}

    def on_step_completion(
        self,
        *,
        video_id: str,
        task_id: str,
        detected_step: int,
    ) -> OnlineOmissionUpdate:
        normalized_task = task_id.lower()
        graph = self.task_graphs.get(normalized_task)
        if graph is None:
            raise KeyError(f"No task graph configured for task {task_id!r}.")
        tracker = self._trackers.get(video_id)
        if tracker is None:
            tracker = OnlineOmissionTracker(graph)
            self._trackers[video_id] = tracker
            self._task_ids[video_id] = normalized_task
        elif self._task_ids[video_id] != normalized_task:
            raise ValueError(
                f"{video_id}: task changed from {self._task_ids[video_id]!r} to {normalized_task!r}."
            )
        return tracker.update(detected_step)

    def finalize_video(self, video_id: str) -> OnlineOmissionUpdate | None:
        tracker = self._trackers.get(video_id)
        return None if tracker is None else tracker.finalize()
