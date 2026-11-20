from __future__ import annotations

import ast
import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TaskGraph:
    task_id: str
    edges: tuple[tuple[int, int], ...]
    start_node: int
    end_node: int
    step_labels: tuple[tuple[int, str], ...] = ()
    node_to_instruction: tuple[tuple[int, int], ...] = ()
    graph_kind: str = "path"

    @property
    def nodes(self) -> set[int]:
        labeled = {node for node, _ in self.step_labels}
        return labeled.union(node for edge in self.edges for node in edge)

    @property
    def recipe_nodes(self) -> set[int]:
        return self.nodes.difference({self.start_node, self.end_node})

    @property
    def instruction_lookup(self) -> dict[int, int]:
        return dict(self.node_to_instruction)

    @property
    def labels(self) -> dict[int, str]:
        return dict(self.step_labels)

    def successors(self, node: int) -> tuple[int, ...]:
        return tuple(end for start, end in self.edges if start == node)

    def predecessors(self, node: int) -> tuple[int, ...]:
        return tuple(start for start, end in self.edges if end == node)

    def instruction_index_for_node(self, node: int) -> int | None:
        mapping = self.instruction_lookup
        if mapping:
            return mapping.get(node)
        if node in {self.start_node, self.end_node}:
            return None
        return node - 1

    def node_for_instruction(self, index: int) -> int | None:
        mapping = self.instruction_lookup
        if mapping:
            reverse = {instruction: node for node, instruction in mapping.items()}
            return reverse.get(index)
        return index + 1

    def ancestors(self, node: int) -> set[int]:
        seen: set[int] = set()
        queue = deque(self.predecessors(node))
        while queue:
            current = queue.popleft()
            if current in seen:
                continue
            seen.add(current)
            queue.extend(self.predecessors(current))
        return seen

    def descendants(self, node: int) -> set[int]:
        seen: set[int] = set()
        queue = deque(self.successors(node))
        while queue:
            current = queue.popleft()
            if current in seen:
                continue
            seen.add(current)
            queue.extend(self.successors(current))
        return seen

    def reachable_from(self, start: int, *, blocked: set[int] | None = None) -> set[int]:
        blocked = blocked or set()
        seen: set[int] = set()
        queue = deque([start])
        while queue:
            current = queue.popleft()
            if current in seen or current in blocked:
                continue
            seen.add(current)
            queue.extend(self.successors(current))
        return seen

    def is_mandatory_ancestor(self, ancestor: int, node: int) -> bool:
        if ancestor == node:
            return False
        if ancestor not in self.ancestors(node):
            return False
        return node not in self.reachable_from(self.start_node, blocked={ancestor})

    def recipe_prerequisites(self, instruction_index: int) -> set[int]:
        node = self.node_for_instruction(instruction_index)
        if node is None:
            return set()
        required: set[int] = set()
        for ancestor in self.ancestors(node):
            if ancestor in {self.start_node, self.end_node}:
                continue
            index = self.instruction_index_for_node(ancestor)
            if index is not None:
                required.add(index)
        return required

    def assert_acyclic(self) -> None:
        visiting: set[int] = set()
        visited: set[int] = set()

        def visit(node: int) -> None:
            if node in visited:
                return
            if node in visiting:
                raise ValueError(f"{self.task_id}: task graph contains a cycle")
            visiting.add(node)
            for successor in self.successors(node):
                visit(successor)
            visiting.remove(node)
            visited.add(node)

        for node in sorted(self.nodes):
            visit(node)

    def all_paths(self, *, max_paths: int = 10_000) -> list[tuple[int, ...]]:
        paths: list[tuple[int, ...]] = []

        def visit(node: int, path: tuple[int, ...]) -> None:
            if len(paths) >= max_paths:
                raise ValueError(f"{self.task_id}: path limit {max_paths} exceeded")
            if node == self.end_node:
                paths.append(path)
                return
            for successor in self.successors(node):
                if successor in path:
                    raise ValueError(f"{self.task_id}: task graph contains a cycle")
                visit(successor, (*path, successor))

        visit(self.start_node, (self.start_node,))
        return paths

    def recipe_paths(self) -> list[tuple[int, ...]]:
        """Return paths in the zero-based STEP_k identifier domain."""
        if self.graph_kind == "dependency" and self.node_to_instruction:
            return [
                tuple(
                    index
                    for node in path
                    if (index := self.instruction_index_for_node(node)) is not None
                )
                for path in self.all_paths()
            ]
        return [
            tuple(
                node - 1
                for node in path
                if node not in {self.start_node, self.end_node}
            )
            for path in self.all_paths()
        ]


def parse_task_graphs(path: str | Path) -> dict[str, TaskGraph]:
    text = Path(path).read_text(encoding="utf-8")
    pattern = re.compile(
        r"(?P<task>[A-Za-z]+):\s*\n"
        r"Edges:\s*(?P<edges>\[[^\n]+\])\s*\n"
        r"Start node:\s*(?P<start>-?\d+)\s*\n"
        r"End node:\s*(?P<end>-?\d+)",
        flags=re.MULTILINE,
    )
    graphs: dict[str, TaskGraph] = {}
    for match in pattern.finditer(text):
        task_id = match.group("task").lower()
        raw_edges = ast.literal_eval(match.group("edges"))
        edges = tuple((int(start), int(end)) for start, end in raw_edges)
        graphs[task_id] = TaskGraph(
            task_id=task_id,
            edges=edges,
            start_node=int(match.group("start")),
            end_node=int(match.group("end")),
        )
    if not graphs:
        raise ValueError(f"No task graphs found in {path}")
    return graphs


def recipe_slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())
