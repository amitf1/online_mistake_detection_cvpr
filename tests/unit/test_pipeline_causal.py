from __future__ import annotations

from pathlib import Path

from online_mistake_detection.pipeline.causal import (
    causal_module_a_example,
    causal_window,
    run_causal_video,
    target_for_window,
)
from online_mistake_detection.pipeline.events import PipelineEvent
from online_mistake_detection.schemas import Segment, VideoRecord, WindowSpec

INSTRUCTIONS = ["Pour water", "Grind beans", "Stir coffee"]


def _record() -> VideoRecord:
    return VideoRecord(
        dataset="egoper",
        task_id="coffee",
        video_id="coffee_test",
        video_path=Path("/tmp/coffee_test.mp4"),
        instructions=list(INSTRUCTIONS),
        segments=[
            Segment(start=1.0, end=8.0, instruction_index=0, instruction="Pour water", is_mistake=False, annotation_position=0),
            Segment(start=8.0, end=17.0, instruction_index=1, instruction="Grind beans", is_mistake=False, annotation_position=1),
            Segment(start=17.0, end=19.0, instruction_index=-1, instruction="Extra", is_mistake=True, event_type="ADDITION", annotation_position=2),
            Segment(start=19.0, end=28.0, instruction_index=2, instruction="Stir coffee", is_mistake=False, annotation_position=3),
        ],
    )


def _fake_b(spans: dict[float, tuple[float, float] | None]):
    calls: list[WindowSpec] = []

    def run_b(window: WindowSpec, prediction: str, raw: str, seconds: float) -> PipelineEvent:
        calls.append(window)
        letter = prediction[:1]
        index = ord(letter) - ord("C")
        instruction = INSTRUCTIONS[index] if 0 <= index < len(INSTRUCTIONS) else "Extra step"
        event = PipelineEvent(
            video_id=window.video_id,
            task_id=window.task_id,
            step_index=window.step_index,
            instruction=instruction,
            trigger_time=window.window_end,
            accumulation_start=window.window_start,
            accumulation_end=window.window_end,
            module_a_prediction=prediction,
            occurrence_id="gold-occurrence",
        )
        span = spans.get(window.window_end)
        if span is None:
            event.stage_status = "B_NOT_COMPLETED"
        else:
            event.pred_global_start, event.pred_global_end = span
        return event

    return run_b, calls


def _fake_a(answers: dict[float, str]):
    prompts: list[str] = []

    def score_a(window: WindowSpec, example) -> tuple[str, str, float]:
        prompts.append(example.prompt_text)
        return answers.get(window.window_end, "A"), "{}", 0.0

    return score_a, prompts


def test_not_completed_keeps_clip_start_and_grows() -> None:
    score_a, _ = _fake_a({15.0: "C"})
    run_b, _ = _fake_b({15.0: (1.0, 8.5)})
    result = run_causal_video(_record(), duration=30.0, score_a=score_a, run_b=run_b)
    windows = [(row["window"]["window_start"], row["window"]["window_end"]) for row in result.poll_rows]
    assert windows[:3] == [(0.0, 5.0), (0.0, 10.0), (0.0, 15.0)]


def test_letter_resets_to_module_b_end() -> None:
    score_a, _ = _fake_a({10.0: "C"})
    run_b, _ = _fake_b({10.0: (1.0, 8.5)})
    result = run_causal_video(_record(), duration=30.0, score_a=score_a, run_b=run_b)
    windows = [(row["window"]["window_start"], row["window"]["window_end"]) for row in result.poll_rows]
    assert windows[2:4] == [(8.5, 15.0), (8.5, 20.0)]
    assert result.poll_rows[1]["disposition"] == "reset_to_module_b_end"


def test_module_b_not_completed_keeps_growing() -> None:
    score_a, _ = _fake_a({10.0: "C", 15.0: "C"})
    run_b, calls = _fake_b({10.0: None, 15.0: (1.0, 9.0)})
    result = run_causal_video(_record(), duration=30.0, score_a=score_a, run_b=run_b)
    assert [call.window_start for call in calls] == [0.0, 0.0]
    assert result.poll_rows[1]["disposition"] == "module_b_not_completed"
    assert result.poll_rows[3]["window"]["window_start"] == 9.0


def _fake_c(answers: dict[float, str]):
    seen: list[list[float]] = []

    def run_c(event: PipelineEvent, video_events: list[PipelineEvent]) -> None:
        seen.append([item.accumulation_end for item in video_events])
        answer = answers.get(event.accumulation_end, "CORRECT")
        if answer == "NOT_COMPLETED":
            event.stage_status = "C_NOT_COMPLETED"
        else:
            event.module_c_classification = answer
            event.stage_status = "C_CLASSIFIED"

    return run_c, seen


def test_module_c_not_completed_keeps_cursor_and_history() -> None:
    score_a, _ = _fake_a({10.0: "C", 15.0: "C"})
    run_b, calls = _fake_b({10.0: (1.0, 7.0), 15.0: (1.0, 8.5)})
    run_c, seen = _fake_c({10.0: "NOT_COMPLETED"})
    result = run_causal_video(_record(), duration=30.0, score_a=score_a, run_b=run_b, run_c=run_c)
    assert [call.window_start for call in calls] == [0.0, 0.0]
    assert result.poll_rows[1]["disposition"] == "module_c_not_completed"
    assert result.poll_rows[2]["completed_steps_in_prompt"] == []
    assert result.poll_rows[2]["disposition"] == "reset_to_module_b_end"
    assert result.poll_rows[3]["window"]["window_start"] == 8.5
    assert result.poll_rows[3]["completed_steps_in_prompt"] == ["Pour water"]
    assert seen == [[10.0], [10.0, 15.0]]
    assert [event.stage_status for event in result.events] == ["C_NOT_COMPLETED", "C_CLASSIFIED"]


def test_module_c_is_not_called_without_a_span() -> None:
    score_a, _ = _fake_a({10.0: "C"})
    run_b, _ = _fake_b({10.0: None})
    run_c, seen = _fake_c({})
    result = run_causal_video(_record(), duration=30.0, score_a=score_a, run_b=run_b, run_c=run_c)
    assert not seen
    assert result.poll_rows[1]["disposition"] == "module_b_not_completed"


def test_extra_is_not_added_to_history() -> None:
    score_a, _ = _fake_a({10.0: "B", 20.0: "D"})
    run_b, _ = _fake_b({10.0: (2.0, 9.0), 20.0: (9.0, 18.0)})
    result = run_causal_video(_record(), duration=30.0, score_a=score_a, run_b=run_b)
    assert result.poll_rows[2]["completed_steps_in_prompt"] == []
    assert result.poll_rows[-1]["completed_steps_in_prompt"] == ["Grind beans"]
    assert result.events[0].step_index == -1
    assert result.events[1].step_index == 1


def test_prompt_never_contains_gold_history() -> None:
    score_a, prompts = _fake_a({})
    run_b, calls = _fake_b({})
    result = run_causal_video(_record(), duration=30.0, score_a=score_a, run_b=run_b)
    assert not calls
    assert all(row["completed_steps_in_prompt"] == [] for row in result.poll_rows)
    assert [row["gold_label"] for row in result.poll_rows] == ["A", "C", "C", "C", "C", "C"]
    empty = causal_module_a_example(
        _record(),
        causal_window(_record(), window_start=0.0, window_end=30.0, completed_steps=()),
    ).prompt_text
    assert prompts == [empty] * len(prompts)


def test_events_carry_module_a_step_not_gold() -> None:
    score_a, _ = _fake_a({10.0: "E"})
    run_b, _ = _fake_b({10.0: (1.0, 8.5)})
    result = run_causal_video(_record(), duration=30.0, score_a=score_a, run_b=run_b)
    event = result.events[0]
    assert event.step_index == 2
    assert event.occurrence_id == ""
    assert event.instruction == "Stir coffee"


def test_runner_queue_checkpoints_ab_then_c(tmp_path: Path, monkeypatch) -> None:
    import argparse
    from types import SimpleNamespace

    from online_mistake_detection.pipeline import runner
    from online_mistake_detection.pipeline.shards import ensure_video_queue
    from online_mistake_detection.pipeline.video_checkpoints import (
        load_event_rows,
        load_module_a_rows,
    )

    record = _record()
    dataset = SimpleNamespace(records=[record])
    queue_dir = tmp_path / "work_queue"
    checkpoint_root = tmp_path / "videos"
    ensure_video_queue(queue_dir, [record.video_id])
    args = argparse.Namespace(
        video_queue_dir=str(queue_dir),
        video_checkpoint_dir=str(checkpoint_root),
        video_shard_rank=0,
        causal_max_videos=-1,
        stride_seconds=5.0,
        module_a_checkpoint=None,
        module_b_checkpoint=None,
        max_seq_length_a=1,
        max_seq_length_b=1,
        vision_resize_a=1,
        vision_resize_b=1,
        module_a_max_new_tokens=1,
        module_b_max_new_tokens=1,
        fbeta_beta=2.0,
        label_score_eval=True,
    )
    fake_run_b, _calls = _fake_b({10.0: (1.0, 8.5)})
    monkeypatch.setattr(runner, "load_generation_eval_model", lambda _args: (object(), object()))
    monkeypatch.setattr(runner, "make_model_args", lambda *a, **k: None)
    monkeypatch.setattr(runner, "make_collator", lambda *a, **k: None)
    monkeypatch.setattr(runner, "ModuleADetectionMetricsCallback", lambda **k: None)
    monkeypatch.setattr(runner, "ModuleBTemporalMetricsCallback", lambda **k: None)
    monkeypatch.setattr(runner, "release_cuda_memory", lambda: None)
    monkeypatch.setattr(runner, "record_duration", lambda _record: 30.0)
    monkeypatch.setattr(
        runner,
        "_score_module_a_example",
        lambda *, window, **k: (window, "C" if window.window_end == 10.0 else "A", "{}", 0.0),
    )
    monkeypatch.setattr(
        runner,
        "run_module_b_one",
        lambda _args, *, window, module_a_prediction, module_a_raw, module_a_elapsed, **k: fake_run_b(
            window, module_a_prediction, module_a_raw, module_a_elapsed
        ),
    )
    claimed: list[str] = []
    _predictions, events, poll_rows = runner.run_causal_ab(
        args, dataset=dataset, video_ids=[], claimed_video_ids=claimed
    )
    assert claimed == [record.video_id]
    assert len(events) == 1
    saved_polls = load_module_a_rows(checkpoint_root, record.video_id)
    assert saved_polls == poll_rows
    assert saved_polls[2]["window"]["window_start"] == 8.5

    def fake_c(_args, c_events, _paths) -> None:
        for event in c_events:
            event.module_c_classification = "CORRECT"

    monkeypatch.setattr(runner, "run_module_c", fake_c)
    monkeypatch.setattr(runner, "apply_graph_stage", lambda *a: None)
    finished = runner.run_causal_c_from_queue(args, dataset=dataset, checkpoint_root=checkpoint_root)
    assert [event.module_c_classification for event in finished] == ["CORRECT"]
    saved_events = load_event_rows(checkpoint_root, record.video_id)
    assert saved_events[0]["pred_global_end"] == 8.5
    assert saved_events[0]["step_index"] == 0


def test_runner_module_c_in_loop_checkpoints_final_events(tmp_path: Path, monkeypatch) -> None:
    import argparse
    from types import SimpleNamespace

    from online_mistake_detection.pipeline import runner
    from online_mistake_detection.pipeline.shards import ensure_video_queue
    from online_mistake_detection.pipeline.video_checkpoints import (
        causal_ab_events_path,
        load_event_rows,
    )

    record = _record()
    dataset = SimpleNamespace(records=[record])
    queue_dir = tmp_path / "work_queue"
    checkpoint_root = tmp_path / "videos"
    ensure_video_queue(queue_dir, [record.video_id])
    args = argparse.Namespace(
        video_queue_dir=str(queue_dir),
        video_checkpoint_dir=str(checkpoint_root),
        video_shard_rank=0,
        causal_max_videos=-1,
        causal_module_c_in_loop=True,
        stride_seconds=5.0,
        max_seq_length_a=1,
        max_seq_length_b=1,
        max_seq_length_c=1,
        vision_resize_a=1,
        vision_resize_b=1,
        vision_resize_c=1,
        module_a_max_new_tokens=1,
        module_b_max_new_tokens=1,
        module_c_max_new_tokens=1,
        fbeta_beta=2.0,
        label_score_eval=True,
        dataset="egoper",
        module_a_label_mode="step_id",
    )
    active: list[str] = []
    bank = SimpleNamespace(activate=active.append, states={"A": {}, "B": {}, "C": {}})
    fake_run_b, _calls = _fake_b({10.0: (1.0, 7.0), 15.0: (1.0, 8.5)})
    monkeypatch.setattr(runner, "load_shared_abc_model", lambda _args: (object(), object(), bank))
    monkeypatch.setattr(runner, "make_collator", lambda *a, **k: None)
    monkeypatch.setattr(runner, "ModuleADetectionMetricsCallback", lambda **k: None)
    monkeypatch.setattr(runner, "ModuleBTemporalMetricsCallback", lambda **k: None)
    monkeypatch.setattr(runner, "ModuleCMistakeMetricsCallback", lambda **k: None)
    monkeypatch.setattr(runner, "release_cuda_memory", lambda: None)
    monkeypatch.setattr(runner, "record_duration", lambda _record: 30.0)
    monkeypatch.setattr(runner, "apply_graph_stage", lambda *a: None)
    monkeypatch.setattr(
        runner,
        "_score_module_a_example",
        lambda *, window, **k: (window, "C" if window.window_end in (10.0, 15.0) else "A", "{}", 0.0),
    )
    monkeypatch.setattr(
        runner,
        "run_module_b_one",
        lambda _args, *, window, module_a_prediction, module_a_raw, module_a_elapsed, **k: fake_run_b(
            window, module_a_prediction, module_a_raw, module_a_elapsed
        ),
    )

    def fake_classify(_args, *, events, index, video_path, **k) -> None:
        assert video_path == record.video_path
        event = events[index]
        if event.accumulation_end == 10.0:
            event.stage_status = "C_NOT_COMPLETED"
        else:
            event.module_c_classification = "CORRECT"
            event.stage_status = "C_CLASSIFIED"

    monkeypatch.setattr(runner, "classify_module_c_event", fake_classify)
    _predictions, events, poll_rows = runner.run_causal_ab(
        args, dataset=dataset, video_ids=[], claimed_video_ids=[]
    )
    assert [event.stage_status for event in events] == ["C_NOT_COMPLETED", "C_CLASSIFIED"]
    assert poll_rows[3]["window"]["window_start"] == 8.5
    assert active[:5] == ["A", "A", "B", "C", "A"]
    saved = load_event_rows(checkpoint_root, record.video_id)
    assert [row["module_c_classification"] for row in saved] == [None, "CORRECT"]
    assert not causal_ab_events_path(checkpoint_root, record.video_id).exists()
    assert not (queue_dir / "bc_remaining.json").exists()


def test_target_for_window_uses_first_gold_end_inside_clip() -> None:
    record = _record()
    assert target_for_window(record, window_start=0.0, window_end=5.0) is None
    assert target_for_window(record, window_start=0.0, window_end=20.0).instruction_index == 0
    assert target_for_window(record, window_start=8.0, window_end=20.0).instruction_index == 1


def test_module_c_clip_selects_span_or_whole_window() -> None:
    import pytest

    from online_mistake_detection.pipeline.runner import c_example_from_event

    event = PipelineEvent(
        video_id="vid",
        task_id="tea",
        step_index=0,
        instruction="Boil water",
        trigger_time=20.0,
        accumulation_start=7.3,
        accumulation_end=20.0,
        module_a_prediction="1",
        pred_global_start=9.0,
        pred_global_end=15.0,
    )
    span = c_example_from_event(event, dataset_name="egoper")
    window = c_example_from_event(event, dataset_name="egoper", clip="window")
    assert span is not None and window is not None
    assert (span.window_start, span.window_end) == (9.0, 15.0)
    assert (window.window_start, window.window_end) == (7.3, 20.0)
    assert window.metadata["module_c_clip"] == "window"
    assert window.prompt_text == span.prompt_text
    with pytest.raises(ValueError):
        c_example_from_event(event, dataset_name="egoper", clip="full")
