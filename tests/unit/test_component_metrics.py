from __future__ import annotations

from evaluate.egoper.component_metrics import breakdown, load


def _poll(video_id: str, start: float, end: float, prediction: str) -> dict:
    return {"prediction": prediction, "window": {"video_id": video_id, "window_start": start, "window_end": end}}


def _event(video_id: str, start: float, end: float, status: str, span: tuple[float, float] | None, label: str | None) -> dict:
    return {
        "video_id": video_id,
        "accumulation_start": start,
        "accumulation_end": end,
        "stage_status": status,
        "pred_global_start": None if span is None else span[0],
        "pred_global_end": None if span is None else span[1],
        "module_c_classification": label,
    }


def test_rejected_span_is_retried_on_a_longer_clip() -> None:
    payload = {
        "ground_truth": [
            {"video_id": "v", "gt_start": 0.0, "gt_end": 8.0, "step_index": 0, "gt_fine_class": "SLIP", "eligible": True},
        ],
        "module_a_predictions": [
            _poll("v", 0.0, 5.0, "A"),
            _poll("v", 0.0, 10.0, "C"),
            _poll("v", 0.0, 15.0, "C"),
        ],
        "events": [
            _event("v", 0.0, 10.0, "C_NOT_COMPLETED", (0.0, 8.0), None),
            _event("v", 0.0, 15.0, "C_CLASSIFIED", (0.5, 8.0), "SLIP"),
        ],
    }
    gold, polls = load(payload)
    report = breakdown(gold, polls, ["v"])

    assert report["module_a"]["polls"] == 3
    assert report["module_a"]["not_completed"] == 1
    assert report["module_a"]["retries"] == 1
    assert report["module_b"]["retries"] == 1
    assert report["module_b"]["spans_rejected_by_module_c"] == 1
    assert report["module_b"]["every_call"]["f1_at_0.5"] == 1.0
    module_c = report["module_c"]["every_call"]
    assert module_c["calls"] == 2
    assert module_c["not_completed"] == 1
    assert module_c["per_class"]["SLIP"]["tp"] == 1
    assert module_c["per_class"]["NO_MATCH"]["predicted"] == 1
    assert report["module_c"]["upstream_correct"]["calls"] == 2
