from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

ModuleName = Literal["A", "B", "C"]
AggregateClass = Literal["MISTAKE", "CORRECTION", "CORRECT"]
EventType = Literal[
    "NORMAL",
    "MISTAKE",
    "CORRECTION",
    "ADDITION",
    "OMISSION",
    "BACKGROUND",
]


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    instruction_index: int
    instruction: str
    is_mistake: bool
    error_label: str | None = None
    caption: str = ""
    event_type: EventType = "NORMAL"
    fine_class: str = "CORRECT"
    aggregate_class: AggregateClass = "CORRECT"
    observed_description: str = ""
    occurrence_id: str = ""
    annotation_position: int = 0
    recipe_step_id: int | None = None
    fine_classes: tuple[str, ...] = ()
    descriptions_by_class: dict[str, tuple[str, ...]] = field(default_factory=dict)
    structural_labels: tuple[str, ...] = ()
    is_missing: bool = False
    modified_description: str = ""
    dataset_metadata: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def resolved_fine_classes(self) -> tuple[str, ...]:
        if self.fine_classes:
            return self.fine_classes
        return (self.fine_class,) if self.fine_class else ()


@dataclass(frozen=True)
class VideoRecord:
    dataset: str
    task_id: str
    video_id: str
    video_path: Path
    instructions: list[str]
    segments: list[Segment]
    task_description: str = ""
    split: str | None = None
    task_graph: Any | None = None
    dataset_metadata: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WindowSpec:
    video_path: Path
    video_id: str
    task_id: str
    step_index: int
    current_step: str
    window_start: float
    window_end: float
    gt_start: float | None
    gt_end: float | None
    label: str
    completed_steps: tuple[str, ...] = ()
    pending_steps: tuple[str, ...] = ()
    all_instructions: tuple[str, ...] = ()
    label_mode: str = "legacy"
    event_type: EventType = "NORMAL"
    fine_class: str | None = None
    aggregate_class: AggregateClass | None = None
    observed_description: str = ""
    occurrence_id: str = ""
    annotation_position: int = -1
    completed_step_ids: tuple[str, ...] = ()
    fine_classes: tuple[str, ...] = ()
    dataset_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return max(0.0, self.window_end - self.window_start)


@dataclass(frozen=True)
class TrainingExample:
    module: ModuleName
    source_dataset: str
    video_path: Path
    video_id: str
    task_id: str
    step_index: int
    window_start: float
    window_end: float
    prompt_text: str
    target_text: str
    gt_start: float | None = None
    gt_end: float | None = None
    label: str | None = None
    mistake: bool | None = None
    reasoning: str | None = None
    event_type: EventType = "NORMAL"
    fine_class: str | None = None
    aggregate_class: AggregateClass | None = None
    observed_description: str = ""
    occurrence_id: str = ""
    fine_classes: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
