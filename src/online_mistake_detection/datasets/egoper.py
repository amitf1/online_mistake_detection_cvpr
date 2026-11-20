from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ..pipeline.task_graph import TaskGraph, parse_task_graphs
from ..schemas import AggregateClass, Segment, VideoRecord
from ..text import normalize_instruction_text, normalize_instructions


_NORMAL_ACTION_LINE = re.compile(r"^Action_(\d+)\s+(\S.*)$")


def humanize_egoper_script_label(text: str) -> str:
    """Turn ``Fold_semi-circle_paper_filter_...`` into readable prompt text."""
    return " ".join(str(text).replace("_", " ").split())


def parse_egoper_normal_actions(path: str | Path) -> dict[int, str]:
    """Parse 1-based ``Action_N`` lines from ``*_normal_actions.txt``."""
    source = Path(path)
    mapping: dict[int, str] = {}
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        match = _NORMAL_ACTION_LINE.match(stripped)
        if match is None:
            raise ValueError(f"{source}:{line_number}: expected 'Action_N ...', got {stripped!r}")
        index = int(match.group(1))
        if index in mapping:
            raise ValueError(f"{source}: duplicate Action_{index}")
        mapping[index] = humanize_egoper_script_label(match.group(2))
    if not mapping:
        raise ValueError(f"{source}: no Action_N lines found")
    return mapping


def overlay_egoper_script_labels(
    action_by_index: Mapping[int, str],
    script_labels: Mapping[int, str],
    *,
    task_id: str,
    source: str | Path,
) -> dict[int, str]:
    """Keep action IDs from ``action2idx``; replace recipe text with the script."""
    recipe_ids = sorted(index for index in action_by_index if index != 0)
    script_ids = sorted(script_labels)
    if recipe_ids != script_ids:
        raise ValueError(
            f"{task_id}: Action_N ids in {source} {script_ids} do not match "
            f"annotation.json action2idx {recipe_ids}"
        )
    merged = dict(action_by_index)
    merged.update(script_labels)
    return merged


EGOPER_FINE_CLASSES = (
    "CORRECT",
    "CORRECTION",
    "MODIFICATION",
    "SLIP",
    "ADDITION",
)
EGOPER_MODULE_C_CLASSES = (
    "CORRECT",
    "CORRECTION",
    "MODIFICATION",
    "SLIP",
)
EGOPER_AGGREGATE_MAPPING: dict[str, AggregateClass] = {
    "CORRECT": "CORRECT",
    "CORRECTION": "CORRECTION",
    "MODIFICATION": "MISTAKE",
    "SLIP": "MISTAKE",
    "ADDITION": "MISTAKE",
}
ACTION_TYPE_TO_FINE = {
    "Normal": "CORRECT",
    "Error_Correction": "CORRECTION",
    "Error_Modification": "MODIFICATION",
    "Error_Slip": "SLIP",
    "Error_Addition": "ADDITION",
}


class EgoPERProvider:
    dataset_name = "egoper"

    def __init__(
        self,
        root: str | Path,
        *,
        split: str | None = None,
        task_ids: set[str] | None = None,
        video_ids: set[str] | None = None,
        require_existing_videos: bool = False,
    ) -> None:
        self.root = Path(root)
        self.annotation_path = self.root / "annotation.json"
        self.task_graph_path = self.root / "task_graph.txt"
        if not self.annotation_path.is_file():
            raise FileNotFoundError(f"EgoPER annotation missing: {self.annotation_path}")
        if split not in {None, "train", "validation", "test"}:
            raise ValueError("split must be train, validation, test, or None")
        self.split = split
        self.task_ids = {task.lower() for task in task_ids} if task_ids else None
        self.video_ids = video_ids
        self.require_existing_videos = require_existing_videos
        self.annotation: dict[str, Any] = json.loads(
            self.annotation_path.read_text(encoding="utf-8")
        )
        self.graphs = parse_task_graphs(self.task_graph_path)
        self._splits = self._load_splits()
        self._trim_bounds = self._load_trim_bounds()

    @property
    def fine_taxonomy(self) -> Sequence[str]:
        return EGOPER_FINE_CLASSES

    @property
    def aggregate_mapping(self) -> Mapping[str, AggregateClass]:
        return EGOPER_AGGREGATE_MAPPING

    def official_split(self, video_id: str) -> str | None:
        for split_name, ids in self._splits.items():
            if video_id in ids:
                return split_name
        return None

    def iter_video_records(self) -> Iterable[VideoRecord]:
        for task_id, task_data in self.annotation.items():
            normalized_task = task_id.lower()
            if self.task_ids and normalized_task not in self.task_ids:
                continue
            action2idx = {
                str(action): int(index)
                for action, index in task_data["action2idx"].items()
            }
            action_by_index, instruction_source = self._recipe_labels(
                normalized_task, action2idx
            )
            instructions = normalize_instructions(
                [
                    action_by_index[index]
                    for index in sorted(action_by_index)
                    if index != 0
                ]
            )
            type_by_index = {
                int(index): name
                for name, index in task_data["actiontype2idx"].items()
            }
            for video_data in task_data["segments"]:
                video_id = str(video_data["video_id"])
                official_split = self.official_split(video_id)
                if self.split and official_split != self.split:
                    continue
                if self.video_ids and video_id not in self.video_ids:
                    continue
                video_path = self._video_path(normalized_task, video_id)
                if self.require_existing_videos and not video_path.is_file():
                    raise FileNotFoundError(f"EgoPER video missing: {video_path}")
                labels = video_data["labels"]
                arrays = [
                    labels["action"],
                    labels["action_type"],
                    labels["time_stamp"],
                    labels["error_description"],
                ]
                if len({len(array) for array in arrays}) != 1:
                    raise ValueError(f"{video_id}: annotation arrays are not aligned")
                segments = [
                    self._normalize_segment(
                        action=int(action),
                        action_type=type_by_index[int(action_type_index)],
                        timestamp=timestamp,
                        description=str(description),
                        action_by_index=action_by_index,
                        row_index=row_index,
                        video_id=video_id,
                    )
                    for row_index, (action, action_type_index, timestamp, description)
                    in enumerate(zip(*arrays))
                ]
                segments = [
                    segment for segment in segments
                    if segment.event_type != "BACKGROUND"
                ]
                yield VideoRecord(
                    dataset=self.dataset_name,
                    task_id=normalized_task,
                    video_id=video_id,
                    video_path=video_path,
                    instructions=instructions,
                    segments=segments,
                    task_description=f"EgoPER procedural task: {normalized_task}",
                    split=official_split,
                    task_graph=self.graphs.get(normalized_task),
                    dataset_metadata={
                        "trim_bounds": self._trim_bounds.get(video_id),
                        "source": str(self.annotation_path),
                        "instruction_source": instruction_source,
                    },
                    raw=video_data,
                )

    def _normalize_segment(
        self,
        *,
        action: int,
        action_type: str,
        timestamp: Sequence[float],
        description: str,
        action_by_index: Mapping[int, str],
        row_index: int,
        video_id: str,
    ) -> Segment:
        start, end = float(timestamp[0]), float(timestamp[1])
        if action == 0 or description.strip().upper() == "BG":
            return Segment(
                start=start,
                end=end,
                instruction_index=-1,
                instruction="background",
                is_mistake=False,
                event_type="BACKGROUND",
                fine_class="CORRECT",
                aggregate_class="CORRECT",
                observed_description=description,
                occurrence_id=f"{video_id}:{row_index}",
                annotation_position=row_index,
                dataset_metadata={"row_index": row_index, "raw_action": action},
            )
        fine_class = ACTION_TYPE_TO_FINE[action_type]
        instruction_index = -1 if fine_class == "ADDITION" else action - 1
        instruction = (
            "undefined / extra step"
            if fine_class == "ADDITION"
            else normalize_instruction_text(
                action_by_index.get(action, f"unknown action {action}")
            )
        )
        event_type = {
            "CORRECT": "NORMAL",
            "CORRECTION": "CORRECTION",
            "ADDITION": "ADDITION",
        }.get(fine_class, "MISTAKE")
        return Segment(
            start=start,
            end=end,
            instruction_index=instruction_index,
            instruction=instruction,
            is_mistake=EGOPER_AGGREGATE_MAPPING[fine_class] == "MISTAKE",
            error_label=None if fine_class == "CORRECT" else fine_class,
            caption=description,
            event_type=event_type,
            fine_class=fine_class,
            aggregate_class=EGOPER_AGGREGATE_MAPPING[fine_class],
            observed_description=description,
            occurrence_id=f"{video_id}:{row_index}",
            annotation_position=row_index,
            dataset_metadata={
                "row_index": row_index,
                "raw_action": action,
                "raw_action_type": action_type,
            },
        )

    def _load_splits(self) -> dict[str, set[str]]:
        split_files = {
            "train": "training.txt",
            "validation": "validation.txt",
            "test": "test.txt",
        }
        result = {name: set() for name in split_files}
        for task_id in self.annotation:
            for split_name, filename in split_files.items():
                path = self.root / task_id / filename
                if not path.is_file():
                    raise FileNotFoundError(f"EgoPER split missing: {path}")
                result[split_name].update(
                    line.strip()
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                )
        return result

    def _load_trim_bounds(self) -> dict[str, tuple[float, float]]:
        bounds: dict[str, tuple[float, float]] = {}
        for task_id in self.annotation:
            path = self.root / task_id / "trim_start_end.txt"
            if not path.is_file():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) >= 3:
                    bounds[parts[0]] = (float(parts[1]), float(parts[2]))
        return bounds

    def _video_path(self, task_id: str, video_id: str) -> Path:
        candidates = (
            self.root / task_id / f"{video_id}.mp4",
            self.root / task_id / "videos" / f"{video_id}.mp4",
            self.root / task_id / "videos" / "trim_videos" / f"{video_id}.mp4",
            self.root / "videos" / f"{video_id}.mp4",
        )
        return next((path for path in candidates if path.is_file()), candidates[0])

    def _recipe_labels(
        self, task_id: str, action2idx: Mapping[str, int]
    ) -> tuple[dict[int, str], str]:
        action_by_index = {index: action for action, index in action2idx.items()}
        script_path = self.root / task_id / f"{task_id}_normal_actions.txt"
        if not script_path.is_file():
            return action_by_index, "action2idx"
        script_labels = parse_egoper_normal_actions(script_path)
        return (
            overlay_egoper_script_labels(
                action_by_index,
                script_labels,
                task_id=task_id,
                source=script_path,
            ),
            str(script_path),
        )

    def validation_report(self) -> dict[str, Any]:
        annotation_ids = {
            str(video["video_id"])
            for task in self.annotation.values()
            for video in task["segments"]
        }
        split_sets = list(self._splits.values())
        duplicate_split_ids = set()
        for index, left in enumerate(split_sets):
            for right in split_sets[index + 1 :]:
                duplicate_split_ids.update(left.intersection(right))
        records = list(self.iter_video_records())
        class_counts = Counter(
            segment.fine_class
            for record in records
            for segment in record.segments
        )
        missing_videos = [
            record.video_id for record in records if not record.video_path.is_file()
        ]
        durations = [
            segment.duration for record in records for segment in record.segments
        ]
        monotonic_timestamps = True
        graph_action_mappings_valid = True
        for record in records:
            previous_start = -1.0
            graph = self.graphs.get(record.task_id)
            for segment in record.segments:
                monotonic_timestamps &= (
                    segment.start >= previous_start and segment.end > segment.start
                )
                previous_start = segment.start
                raw_action = int(segment.dataset_metadata.get("raw_action", -1))
                if (
                    graph is not None
                    and raw_action > 0
                    and segment.fine_class != "ADDITION"
                    and raw_action not in graph.recipe_nodes
                ):
                    graph_action_mappings_valid = False
        return {
            "annotation_video_count": len(annotation_ids),
            "split_video_count": len(set().union(*split_sets)),
            "expected_local_video_count": 389,
            "paper_video_count": 386,
            "split_disjoint": not duplicate_split_ids,
            "split_complete": set().union(*split_sets) == annotation_ids,
            "duplicate_split_ids": sorted(duplicate_split_ids),
            "missing_split_ids": sorted(annotation_ids.difference(set().union(*split_sets))),
            "missing_video_count": len(missing_videos),
            "timestamps_monotonic": monotonic_timestamps,
            "graph_action_mappings_valid": graph_action_mappings_valid,
            "class_counts": dict(sorted(class_counts.items())),
            "duration_seconds": {
                "min": min(durations) if durations else 0.0,
                "max": max(durations) if durations else 0.0,
                "mean": sum(durations) / len(durations) if durations else 0.0,
            },
            "tasks": sorted(self.annotation),
        }
