from online_mistake_detection.metrics.abc_monolith import SequentialMonolithState


def test_wait_retains_accumulation_and_completion_resets_cursor() -> None:
    state = SequentialMonolithState()
    assert state.consume(
        '{"status":"NOT_COMPLETED"}',
        video_id="v",
        task_id="t",
        instruction="step",
        window_end=5.0,
    ) is None
    assert state.accumulation_start == 0.0
    event = state.consume(
        '{"step_id":"STEP_0","start":2,"end":7,'
        '"classification":"CORRECT","reasoning":"done"}',
        video_id="v",
        task_id="t",
        instruction="step",
        window_end=10.0,
    )
    assert event is not None
    assert event.pred_global_end == 7.0
    assert state.cursor == 7.0
    assert not state.should_poll(6.0)


def test_addition_is_excluded_from_predicted_history() -> None:
    state = SequentialMonolithState()
    state.consume(
        '{"step_id":"EXTRA","start":0,"end":1,'
        '"classification":"ADDITION","reasoning":"extra action"}',
        video_id="v",
        task_id="t",
        instruction="extra",
        window_end=5.0,
    )
    assert state.history == []

