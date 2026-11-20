from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

from online_mistake_detection.config import ModuleAConfig, ModuleBConfig, ModuleCConfig
from online_mistake_detection.datasets.egoper import (
    EgoPERProvider,
    overlay_egoper_script_labels,
    parse_egoper_normal_actions,
)
from online_mistake_detection.datasets.training import (
    build_module_a_training_dataset,
    build_module_b_training_dataset,
    build_module_c_training_dataset,
)
from online_mistake_detection.metrics.omission import derive_ground_truth_omissions
from online_mistake_detection.schemas import Segment, VideoRecord
from online_mistake_detection.windowing.module_a import build_module_a_windows
from online_mistake_detection.windowing.module_b import build_module_b_examples
from online_mistake_detection.windowing.module_c import build_module_c_examples
from online_mistake_detection.training.module_a import (
    apply_video_split_subset,
    default_split_file as module_a_split_file,
)
from online_mistake_detection.training.module_b import default_split_file as module_b_split_file
from online_mistake_detection.training.module_c import default_split_file as module_c_split_file


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def build_root(tmp_path: Path, *, normal_actions: str | None = None) -> Path:
    root = tmp_path / "EgoPER"
    task = root / "tea"
    task.mkdir(parents=True)
    (root / "annotation.json").write_text(
        (FIXTURES / "egoper_minimal.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (root / "task_graph.txt").write_text(
        "Tea:\nEdges: [(0, 1), (1, 2), (2, 3)]\n"
        "Start node: 0\nEnd node: 3\n",
        encoding="utf-8",
    )
    (task / "training.txt").write_text("tea_fixture\n", encoding="utf-8")
    (task / "validation.txt").write_text("", encoding="utf-8")
    (task / "test.txt").write_text("", encoding="utf-8")
    (task / "trim_start_end.txt").write_text(
        "tea_fixture 0 9\n", encoding="utf-8"
    )
    if normal_actions is not None:
        (task / "tea_normal_actions.txt").write_text(normal_actions, encoding="utf-8")
    return root


def test_all_modules_resolve_one_dataset_split_file() -> None:
    args = Namespace(dataset="egoper", seed=3407)
    paths = {
        module_a_split_file(args),
        module_b_split_file(args),
        module_c_split_file(args),
    }
    assert len(paths) == 1


def test_run_subset_does_not_modify_canonical_split() -> None:
    records = [
        VideoRecord(
            dataset="egoper",
            task_id="tea",
            video_id=f"video_{index}",
            video_path=Path("/missing.mp4"),
            instructions=["step"],
            segments=[
                Segment(0.0, 1.0, 0, "step", True, fine_class="SLIP")
            ],
        )
        for index in range(5)
    ]
    canonical = {
        "train_video_ids": ["video_0", "video_1", "video_2"],
        "val_video_ids": ["video_3", "video_4"],
        "test_video_ids": [],
        "tasks": {
            "tea": {
                "train": ["video_0", "video_1", "video_2"],
                "val": ["video_3", "video_4"],
                "test": [],
            }
        },
    }
    subset = apply_video_split_subset(
        canonical,
        SimpleNamespace(records=records),
        Namespace(
            seed=3407,
            train_videos_per_task=2,
            val_videos_per_task=1,
            train_validation_erroneous_only=True,
        ),
    )
    assert len(subset["train_video_ids"]) == 2
    assert len(subset["val_video_ids"]) == 1
    assert canonical["train_video_ids"] == ["video_0", "video_1", "video_2"]
    assert canonical["val_video_ids"] == ["video_3", "video_4"]


def test_egoper_normalization_and_taxonomy(tmp_path: Path) -> None:
    provider = EgoPERProvider(build_root(tmp_path), split="train")
    records = list(provider.iter_video_records())
    assert len(records) == 1
    record = records[0]
    assert [segment.fine_class for segment in record.segments] == [
        "CORRECT",
        "MODIFICATION",
        "ADDITION",
        "CORRECTION",
    ]
    assert [segment.aggregate_class for segment in record.segments] == [
        "CORRECT",
        "MISTAKE",
        "MISTAKE",
        "CORRECTION",
    ]
    assert record.segments[2].instruction_index == -1
    assert record.segments[2].observed_description == "Wipe the table"
    assert record.instructions == ["Measure water", "Transfer water"]
    assert record.dataset_metadata["instruction_source"] == "action2idx"


def test_egoper_uses_normal_actions_text_with_zero_based_step_ids(tmp_path: Path) -> None:
    root = build_root(
        tmp_path,
        normal_actions=(
            "Action_1 Measure_water_using_measuring_cup\n"
            "Action_2 Transfer_water_from_measuring_cup_to_kettle\n"
        ),
    )
    record = next(EgoPERProvider(root, split="train").iter_video_records())
    assert record.instructions == [
        "Measure water using measuring cup",
        "Transfer water from measuring cup to kettle",
    ]
    assert record.dataset_metadata["instruction_source"].endswith("tea_normal_actions.txt")
    first_recipe = next(
        segment for segment in record.segments if segment.instruction_index == 0
    )
    second_recipe = next(
        segment for segment in record.segments if segment.instruction_index == 1
    )
    assert first_recipe.instruction == "Measure water using measuring cup"
    assert second_recipe.instruction == "Transfer water from measuring cup to kettle"
    assert first_recipe.dataset_metadata["raw_action"] == 1
    assert second_recipe.dataset_metadata["raw_action"] == 2

    windows = build_module_a_windows(
        [record],
        config=ModuleAConfig(stride_seconds=2.0),
    )
    assert [window.label for window in windows if window.label != "NONE"] == [
        "STEP_0",
        "STEP_0",
        "EXTRA",
        "STEP_1",
    ]
    second_attempt = next(
        window for window in windows
        if window.label == "STEP_0" and window.window_end == 6.0
    )
    assert second_attempt.completed_steps == ("Measure water using measuring cup",)


def test_parse_egoper_normal_actions_rejects_id_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "coffee_normal_actions.txt"
    path.write_text("Action_1 Measure_water\nAction_3 Transfer_water\n", encoding="utf-8")
    parsed = parse_egoper_normal_actions(path)
    assert parsed == {1: "Measure water", 3: "Transfer water"}
    try:
        overlay_egoper_script_labels(
            {0: "BG", 1: "Measure water", 2: "Transfer water"},
            parsed,
            task_id="coffee",
            source=path,
        )
    except ValueError as error:
        assert "do not match" in str(error)
    else:
        raise AssertionError("expected Action_N mismatch to raise")


def test_egoper_normal_actions_mismatch_raises_at_load(tmp_path: Path) -> None:
    root = build_root(tmp_path, normal_actions="Action_1 Measure_water_using_measuring_cup\n")
    try:
        list(EgoPERProvider(root).iter_video_records())
    except ValueError as error:
        assert "do not match" in str(error)
    else:
        raise AssertionError("expected missing Action_2 to raise")


def test_egoper_module_windows_are_dataset_neutral(tmp_path: Path) -> None:
    provider = EgoPERProvider(build_root(tmp_path))
    records = list(provider.iter_video_records())
    module_a = build_module_a_windows(
        records,
        config=ModuleAConfig(stride_seconds=2.0),
    )
    completion_labels = [window.label for window in module_a if window.label != "NONE"]
    assert completion_labels == ["STEP_0", "STEP_0", "EXTRA", "STEP_1"]
    second_attempt = next(
        window for window in module_a
        if window.label == "STEP_0" and window.window_end == 6.0
    )
    assert second_attempt.completed_steps == ("Measure water",)

    module_b = build_module_b_examples(
        records,
        config=ModuleBConfig(
            pre_pad_ratio_min=0.1,
            pre_pad_ratio_max=0.1,
            post_pad_ratio_min=0.1,
            post_pad_ratio_max=0.1,
            include_incomplete_negatives=False,
        ),
    )
    assert len(module_b) == 4
    for example in module_b:
        target = json.loads(example.target_text)
        assert 0.0 <= float(target["start"]) < float(target["end"]) <= example.metadata[
            "clip_duration"
        ]

    module_c = build_module_c_examples(
        records,
        fine_taxonomy=provider.fine_taxonomy,
        aggregate_mapping=provider.aggregate_mapping,
        config=ModuleCConfig(
            jitter_ratio_min=0.0,
            jitter_ratio_max=0.0,
            include_not_completed_examples=False,
        ),
    )
    assert [example.label for example in module_c] == [
        "CORRECT",
        "MODIFICATION",
        "CORRECTION",
    ]
    for example in module_c:
        target = json.loads(example.target_text)
        assert target["classification"] == example.label
        assert target["reasoning"]
        outcomes = set(
            example.prompt_text.split("Allowed outcomes: ")[1].split(".")[0].split(", ")
        )
        assert "CORRECT" in outcomes
        assert "ADDITION" not in outcomes
        assert "CORRECTION" in outcomes
        assert "MODIFICATION" in outcomes
        assert "SLIP" in outcomes
        assert "NOT_COMPLETED" in outcomes


def test_module_a_completion_uses_the_next_poll_boundary() -> None:
    record = VideoRecord(
        dataset="egoper",
        task_id="coffee",
        video_id="poll_boundary",
        video_path=Path("/missing.mp4"),
        instructions=["measure water", "transfer water"],
        segments=[
            Segment(
                start=0.0,
                end=17.6,
                instruction_index=0,
                instruction="measure water",
                is_mistake=False,
                fine_class="CORRECT",
            ),
            Segment(
                start=17.6,
                end=30.0,
                instruction_index=1,
                instruction="transfer water",
                is_mistake=False,
                fine_class="CORRECT",
            ),
        ],
    )
    windows = build_module_a_windows([record], config=ModuleAConfig(stride_seconds=5.0))
    completion = next(window for window in windows if window.label == "STEP_0")
    following_poll = next(
        window
        for window in windows
        if window.label == "NONE" and window.window_start == 17.6
    )
    assert completion.window_start == 0.0
    assert completion.window_end == 20.0
    assert following_poll.window_end == 22.6


def test_egoper_module_b_anchors_to_module_a_and_augments_only_the_end() -> None:
    record = VideoRecord(
        dataset="egoper",
        task_id="coffee",
        video_id="module_b_anchor",
        video_path=Path("/missing.mp4"),
        instructions=["measure water", "transfer water"],
        segments=[
            Segment(0.0, 17.6, 0, "measure water", False, fine_class="CORRECT"),
            Segment(17.6, 30.0, 1, "transfer water", False, fine_class="CORRECT"),
        ],
    )
    module_a_config = ModuleAConfig(stride_seconds=5.0)
    normal = build_module_b_examples(
        [record],
        config=ModuleBConfig(late_trigger_probability=0.0),
        module_a_config=module_a_config,
    )
    delayed = build_module_b_examples(
        [record],
        config=ModuleBConfig(late_trigger_probability=1.0),
        module_a_config=module_a_config,
    )
    assert (normal[0].window_start, normal[0].window_end) == (0.0, 20.0)
    assert (delayed[0].window_start, delayed[0].window_end) == (0.0, 25.0)
    assert normal[0].target_text == delayed[0].target_text
    assert normal[1].window_start == delayed[1].window_start == 17.6
    assert normal[0].metadata["late_trigger_augmented"] is False
    assert delayed[0].metadata["late_trigger_augmented"] is True


def test_egoper_training_adapters_feed_all_modules(tmp_path: Path) -> None:
    provider = EgoPERProvider(build_root(tmp_path))
    module_a = build_module_a_training_dataset(
        provider,
        config=ModuleAConfig(stride_seconds=2.0),
    )
    module_b = build_module_b_training_dataset(
        provider,
        config=ModuleBConfig(),
    )
    module_c = build_module_c_training_dataset(
        provider,
        config=ModuleCConfig(jitter_ratio_min=0.0, jitter_ratio_max=0.0),
    )
    assert {example.target_text for example in module_a} >= {"A", "B", "C", "D"}
    assert all(example.source_dataset == "egoper" for example in module_a)
    assert len(module_b) >= 4
    assert {example.label for example in module_c} == {
        "CORRECT",
        "MODIFICATION",
        "CORRECTION",
    }


def test_egoper_validation_checks_splits(tmp_path: Path) -> None:
    provider = EgoPERProvider(build_root(tmp_path))
    report = provider.validation_report()
    assert report["annotation_video_count"] == 1
    assert report["split_video_count"] == 1
    assert report["split_disjoint"] is True
    assert report["split_complete"] is True


def test_egoper_omission_ground_truth_is_derived_from_graph(tmp_path: Path) -> None:
    record = next(EgoPERProvider(build_root(tmp_path)).iter_video_records())
    result = derive_ground_truth_omissions(record)
    assert result.omitted_steps == frozenset()
