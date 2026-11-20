from __future__ import annotations

import string
from dataclasses import replace
from typing import Any, Iterable

from ..config import ModuleAConfig, ModuleBConfig, ModuleCConfig
from ..prompts.module_a import (
    build_module_a_completion_set_prompt,
    build_module_a_step_id_user_prompt,
)
from ..schemas import TrainingExample, VideoRecord, WindowSpec
from ..windowing.module_a import build_module_a_windows
from ..windowing.module_b import build_module_b_examples
from ..windowing.module_c import build_module_c_examples


def _step_letter(instruction_index: int) -> str:
    alphabet_index = instruction_index + 2
    if alphabet_index >= len(string.ascii_uppercase):
        raise ValueError("Module A supports at most 24 task instructions.")
    return string.ascii_uppercase[alphabet_index]


def _letter_label(label: str) -> str:
    if label == "NONE":
        return "A"
    if label == "EXTRA":
        return "B"
    if label.startswith("STEP_"):
        return _step_letter(int(label.removeprefix("STEP_")))
    raise ValueError(f"Unsupported dataset-neutral Module A label: {label!r}")


def _module_a_prompt(record: VideoRecord, window: WindowSpec, taxonomy: tuple[str, ...]) -> str:
    del taxonomy
    return build_module_a_step_id_user_prompt(
        task_name=record.task_id,
        task_description=record.task_description or record.task_id,
        instructions=record.instructions,
        completed_steps=window.completed_steps,
        dataset_name=record.dataset,
    )


class ModuleTrainingDataset:
    def __init__(self, records: Iterable[VideoRecord], examples: list[TrainingExample]) -> None:
        self.records = list(records)
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> TrainingExample:
        return self.examples[index]

    def _window_to_example(self, window: WindowSpec) -> TrainingExample:
        for example in self.examples:
            if (
                example.video_id == window.video_id
                and abs(example.window_start - window.window_start) < 1e-6
                and abs(example.window_end - window.window_end) < 1e-6
                and example.label == window.label
            ):
                return example
        raise KeyError(
            f"No training example for {window.video_id} [{window.window_start}, {window.window_end}]"
        )


def _records(provider: Any, max_videos: int | None) -> list[VideoRecord]:
    records = list(provider.iter_video_records())
    return records if max_videos is None else records[:max_videos]


def build_module_a_training_dataset(
    provider: Any,
    *,
    config: ModuleAConfig,
    max_videos: int | None = None,
) -> ModuleTrainingDataset:
    records = _records(provider, max_videos)
    records_by_video = {record.video_id: record for record in records}
    taxonomy = tuple(provider.fine_taxonomy)
    raw_windows = build_module_a_windows(records, config=config)
    if config.label_mode == "completion_set":
        windows = raw_windows
    else:
        windows = [replace(window, label=_letter_label(window.label)) for window in raw_windows]
    examples: list[TrainingExample] = []
    for window in windows:
        record = records_by_video[window.video_id]
        if config.label_mode == "completion_set":
            prompt = build_module_a_completion_set_prompt(
                task_name=record.task_id,
                task_description=record.task_description or record.task_id,
                instructions=record.instructions,
                completed_steps=window.completed_steps,
            )
            semantic_label = window.label
            label_mode = "completion_set"
        else:
            prompt = _module_a_prompt(record, window, taxonomy)
            semantic_label = (
                "NONE"
                if window.label == "A"
                else "EXTRA"
                if window.label == "B"
                else f"STEP_{string.ascii_uppercase.index(window.label) - 2}"
            )
            label_mode = "step_id"
        examples.append(
            TrainingExample(
                module="A",
                source_dataset=record.dataset,
                video_path=window.video_path,
                video_id=window.video_id,
                task_id=window.task_id,
                step_index=window.step_index,
                window_start=window.window_start,
                window_end=window.window_end,
                gt_start=window.gt_start,
                gt_end=window.gt_end,
                prompt_text=prompt,
                target_text=window.label,
                label=window.label,
                event_type=window.event_type,
                fine_class=window.fine_class,
                aggregate_class=window.aggregate_class,
                observed_description=window.observed_description,
                occurrence_id=window.occurrence_id,
                fine_classes=window.fine_classes,
                metadata={
                    "completed_steps": list(window.completed_steps),
                    "completed_step_ids": list(window.completed_step_ids),
                    "completed_occurrences": window.dataset_metadata.get("completed_occurrences", []),
                    "all_instructions": list(window.all_instructions),
                    "label_mode": label_mode,
                    "semantic_label": semantic_label,
                    "collision_bucket": bool(window.dataset_metadata.get("collision_bucket")),
                },
            )
        )
    dataset = ModuleTrainingDataset(records, examples)
    dataset.windows = windows
    return dataset


def build_module_b_training_dataset(
    provider: Any,
    *,
    config: ModuleBConfig,
    module_a_config: ModuleAConfig | None = None,
    max_videos: int | None = None,
) -> ModuleTrainingDataset:
    records = _records(provider, max_videos)
    return ModuleTrainingDataset(
        records,
        build_module_b_examples(
            records,
            config=config,
            module_a_config=module_a_config,
        ),
    )


def build_module_c_training_dataset(
    provider: Any,
    *,
    config: ModuleCConfig,
    max_videos: int | None = None,
) -> ModuleTrainingDataset:
    records = _records(provider, max_videos)
    return ModuleTrainingDataset(
        records,
        build_module_c_examples(
            records,
            config=config,
            fine_taxonomy=provider.fine_taxonomy,
            aggregate_mapping=provider.aggregate_mapping,
        ),
    )
