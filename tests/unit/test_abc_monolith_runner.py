import json
from pathlib import Path
from typing import Any

from evaluate.abc_monolith.run import (
    EvaluationInterrupted,
    run_sequential_evaluation,
    shard_output_path,
)
from online_mistake_detection.pipeline.runner import RunControl
from evaluate.abc_monolith.evaluate import evaluate_artifact
from online_mistake_detection.metrics.abc_monolith import validate_metric_manifest
from online_mistake_detection.schemas import Segment, VideoRecord


def test_sequential_runner_uses_predicted_history_and_preserves_every_poll(
    tmp_path: Path,
) -> None:
    record = VideoRecord(
        dataset="egoper",
        task_id="tea",
        video_id="video-1",
        video_path=Path("/does/not/exist.mp4"),
        instructions=["boil water", "pour water"],
        segments=[
            Segment(
                start=1.0,
                end=4.0,
                instruction_index=0,
                instruction="boil water",
                is_mistake=False,
                fine_class="CORRECT",
                occurrence_id="first",
            ),
            Segment(
                start=6.0,
                end=9.0,
                instruction_index=1,
                instruction="pour water",
                is_mistake=True,
                fine_class="SLIP",
                aggregate_class="MISTAKE",
                occurrence_id="second",
            ),
        ],
    )
    outputs = iter(
        [
            (
                '{"step_id":"STEP_0","start":1.0,"end":4.0,'
                '"classification":"CORRECT","reasoning":"Water boiled."}'
            ),
            (
                '{"step_id":"STEP_1","start":2.0,"end":5.0,'
                '"classification":"SLIP","reasoning":"Water was spilled."}'
            ),
        ]
    )
    prompts: list[str] = []

    def generate(example: dict) -> str:
        prompts.append(example["prompt_text"])
        return next(outputs)

    artifact = run_sequential_evaluation([record], generate=generate)

    assert artifact["schema_version"] == "egoper_abc_monolith_v1"
    assert len(artifact["polls"]) == 2
    assert len(artifact["events"]) == 2
    assert artifact["events"][1]["accumulation_start"] == 4.0
    assert artifact["events"][1]["module_c_history"][0]["step_id"] == "STEP_0"
    assert "Predicted prior eligible actions" in prompts[1]
    assert "Water boiled." in prompts[1]
    assert artifact["metadata"]["history_source"] == "predicted"
    assert artifact["metadata"]["candidate_scoring"] is False
    assert "egoper_standard" in artifact["metrics"]
    assert "prediction_runtime/monolith/count" in artifact["metrics"]
    recomputed = evaluate_artifact(artifact)
    assert "module_a_step_id/macro_auc" in validate_metric_manifest(
        recomputed["metrics"], raise_on_missing=False
    )
    assert recomputed["metrics"]["temporal/f1_at_iou_0.5"] == artifact["metrics"][
        "temporal/f1_at_iou_0.5"
    ]
    assert recomputed["metrics"]["egoper_standard"] == artifact["metrics"][
        "egoper_standard"
    ]


def _two_step_record() -> VideoRecord:
    return VideoRecord(
        dataset="egoper",
        task_id="tea",
        video_id="video-1",
        video_path=Path("/does/not/exist.mp4"),
        instructions=["boil water", "pour water"],
        segments=[
            Segment(
                start=1.0,
                end=4.0,
                instruction_index=0,
                instruction="boil water",
                is_mistake=False,
                fine_class="CORRECT",
                occurrence_id="first",
            ),
            Segment(
                start=6.0,
                end=9.0,
                instruction_index=1,
                instruction="pour water",
                is_mistake=True,
                fine_class="SLIP",
                aggregate_class="MISTAKE",
                occurrence_id="second",
            ),
        ],
    )


def test_history_modes_change_only_the_prompt_history() -> None:
    outputs = [
        (
            '{"step_id":"STEP_0","start":1.0,"end":4.0,'
            '"classification":"CORRECT","reasoning":"Water boiled."}'
        ),
        (
            '{"step_id":"STEP_1","start":2.0,"end":5.0,'
            '"classification":"SLIP","reasoning":"Water was spilled."}'
        ),
    ]

    def run(mode: str) -> tuple[dict, list[str]]:
        prompts: list[str] = []
        stream = iter(outputs)

        def generate(example: dict) -> str:
            prompts.append(example["prompt_text"])
            return next(stream)

        artifact = run_sequential_evaluation(
            [_two_step_record()],
            generate=generate,
            history_mode=mode,
        )
        return artifact, prompts

    predicted, predicted_prompts = run("predicted")
    none, none_prompts = run("none")
    gold, gold_prompts = run("gold")

    assert predicted["metadata"]["evaluation_view"] == (
        "realistic_predicted_history_sequential"
    )
    assert "Predicted prior eligible actions" in predicted_prompts[1]
    assert "Water boiled." in predicted_prompts[1]
    assert predicted["events"][1]["accumulation_start"] == 4.0
    assert predicted["polls"][1]["history_mode"] == "predicted"
    assert predicted["polls"][1]["history_cutoff_seconds"] is None

    assert none["metadata"]["evaluation_view"] == "sequential_no_history"
    assert none["events"][1]["accumulation_start"] == 4.0
    assert all("No prior actions supplied" in prompt for prompt in none_prompts)
    assert all("Water boiled." not in prompt for prompt in none_prompts)
    assert none["polls"][1]["supplied_history"] == []

    assert gold["metadata"]["evaluation_view"] == "sequential_gold_causal_history"
    assert gold["events"][1]["accumulation_start"] == 4.0
    assert "Gold prior eligible actions" in gold_prompts[1]
    assert "Water boiled." not in gold_prompts[1]
    assert "1. STEP_0 | boil water | CORRECT |" in gold_prompts[1]
    assert "1. STEP_0 | boil water | CORRECT |" not in gold_prompts[0]
    assert gold["polls"][0]["history_occurrence_ids"] == []
    assert gold["polls"][1]["history_cutoff_kind"] == "target_segment_start"
    assert gold["polls"][1]["history_cutoff_seconds"] == 6.0
    assert gold["polls"][1]["history_occurrence_ids"] == ["first"]
    assert "second" not in gold["polls"][1]["history_occurrence_ids"]


def test_gold_history_excludes_the_target_and_additions() -> None:
    from online_mistake_detection.windowing.abc_monolith import gold_history_for_poll

    record = _two_step_record()
    record.segments.append(
        Segment(
            start=4.2,
            end=5.0,
            instruction_index=-1,
            instruction="wipe the counter",
            is_mistake=True,
            fine_class="ADDITION",
            occurrence_id="extra",
        )
    )
    history, cutoff, kind = gold_history_for_poll(
        record,
        window_start=4.0,
        window_end=9.0,
        target=record.segments[1],
    )
    assert cutoff == 6.0
    assert kind == "target_segment_start"
    assert [item["occurrence_id"] for item in history] == ["first"]


def test_stop_between_polls_discards_the_in_flight_video() -> None:
    record = VideoRecord(
        dataset="egoper",
        task_id="tea",
        video_id="video-1",
        video_path=Path("/does/not/exist.mp4"),
        instructions=["boil water", "pour water"],
        segments=[
            Segment(
                start=1.0,
                end=4.0,
                instruction_index=0,
                instruction="boil water",
                is_mistake=False,
                fine_class="CORRECT",
                occurrence_id="first",
            ),
            Segment(
                start=6.0,
                end=9.0,
                instruction_index=1,
                instruction="pour water",
                is_mistake=True,
                fine_class="SLIP",
                aggregate_class="MISTAKE",
                occurrence_id="second",
            ),
        ],
    )
    calls: list[int] = []

    def generate(example: dict) -> str:
        del example
        calls.append(1)
        RunControl.stop_requested = True
        return (
            '{"step_id":"STEP_0","start":1.0,"end":4.0,'
            '"classification":"CORRECT","reasoning":"Water boiled."}'
        )

    try:
        try:
            run_sequential_evaluation([record], generate=generate)
        except EvaluationInterrupted as exc:
            assert str(exc) == "video-1"
        else:
            raise AssertionError("expected EvaluationInterrupted")
        assert calls == [1]
    finally:
        RunControl.stop_requested = False


def test_score_pass_preserves_generated_decisions_and_is_timed_separately() -> None:
    record = VideoRecord(
        dataset="egoper",
        task_id="tea",
        video_id="video-1",
        video_path=Path("/does/not/exist.mp4"),
        instructions=["pour water"],
        segments=[
            Segment(
                start=6.0,
                end=9.0,
                instruction_index=0,
                instruction="pour water",
                is_mistake=True,
                fine_class="SLIP",
                aggregate_class="MISTAKE",
            )
        ],
    )
    generated = (
        '{"step_id":"STEP_0","start":6.0,"end":9.0,'
        '"classification":"SLIP","reasoning":"Water was spilled."}'
    )
    outputs = iter(['{"status":"NOT_COMPLETED"}', generated])

    def score(_example: dict, prediction: Any):
        if prediction.completed:
            return (
                {"A": 2.0, "C": 0.1},
                {
                    "CORRECT": 2.0,
                    "SLIP": 0.1,
                    "MODIFICATION": 3.0,
                    "CORRECTION": 4.0,
                },
                0.25,
            )
        return ({"A": 0.1, "C": 2.0}, {}, 0.25)

    artifact = run_sequential_evaluation(
        [record],
        generate=lambda _example: next(outputs),
        score=score,
    )

    assert artifact["events"][0]["module_c_classification"] == "SLIP"
    assert artifact["polls"][1]["raw_json"] == generated
    assert artifact["metadata"]["score_pass_seconds"] == 0.5
    assert artifact["metadata"]["candidate_scoring"] is True
    assert artifact["metrics"]["egoper_standard"]["segment_ap"]["available"] is True
    score_wrapper = json.loads(artifact["module_a_predictions"][1]["raw_output"])
    assert score_wrapper["generated_text"] == generated
    assert score_wrapper["candidate_nlls"]["C"] == 0.1
    assert validate_metric_manifest(
        artifact["metrics"], raise_on_missing=False
    ) == []


def test_distributed_shard_output_names_are_deterministic() -> None:
    path = Path("/tmp/generated.json")
    assert shard_output_path(path, rank=0, world_size=1) == path
    assert shard_output_path(path, rank=3, world_size=8).name == (
        "generated.rank-003-of-008.json"
    )


def test_cluster_collection_can_defer_all_metrics() -> None:
    record = VideoRecord(
        dataset="egoper",
        task_id="tea",
        video_id="video-1",
        video_path=Path("/does/not/exist.mp4"),
        instructions=["pour water"],
        segments=[
            Segment(
                start=1.0,
                end=4.0,
                instruction_index=0,
                instruction="pour water",
                is_mistake=False,
                fine_class="CORRECT",
            )
        ],
    )
    artifact = run_sequential_evaluation(
        [record],
        generate=lambda _example: '{"status":"NOT_COMPLETED"}',
        compute_metrics=False,
    )
    assert artifact["polls"]
    assert artifact["metrics"] == {}
    assert artifact["metadata"]["metrics_deferred"] is True
