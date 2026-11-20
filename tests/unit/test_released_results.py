from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluate.egoper.evaluate_final import evaluate_scope
from online_mistake_detection.paths import egoper_root


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
NAMES = ("ground_truth", "pipeline", "monolith")


def _load(name: str) -> dict:
    return json.loads((RESULTS / "final_predictions" / f"{name}_test.json").read_text(encoding="utf-8"))["videos"]


@pytest.mark.parametrize("scope", ["all_test", "erroneous_videos"])
def test_released_metrics_follow_from_released_predictions(scope: str) -> None:
    expected = json.loads((RESULTS / "metrics.json").read_text(encoding="utf-8"))[scope]
    sources = {name: _load(name) for name in NAMES}
    videos = sorted(sources["ground_truth"])
    if scope == "erroneous_videos":
        videos = [
            video_id
            for video_id in videos
            if any(segment["classification"] != "CORRECT" for segment in sources["ground_truth"][video_id]["segments"])
        ]
    report = evaluate_scope(sources, videos)
    assert report["videos"] == expected["videos"]
    with_task_graph = (egoper_root() / "task_graph.txt").is_file()
    for method in ("pipeline", "monolith"):
        got = report["methods"][method]
        want = expected["methods"][method]
        for metric in ("predicted_segments", "predicted_by_class", "eda", "binary_f1_at_0.5", "tas_f1_at_0.5"):
            assert got[metric] == pytest.approx(want[metric])
        assert got["fine_at_0.5"]["macro"] == pytest.approx(want["fine_at_0.5"]["macro"])
        for name, row in want["fine_at_0.5"]["per_class"].items():
            assert got["fine_at_0.5"]["per_class"][name] == pytest.approx(row)
        if with_task_graph:
            assert got["o_acc"] == pytest.approx(want["o_acc"])
            assert got["o_iou"] == pytest.approx(want["o_iou"])


def test_released_predictions_cover_the_test_split() -> None:
    split = json.loads((ROOT / "configs/splits/egoper_video_split_seed3407.json").read_text(encoding="utf-8"))
    test_ids = {str(video) for task in split["tasks"].values() for video in task["test"]}
    for name in NAMES:
        assert set(_load(name)) == test_ids


def test_released_predictions_hold_only_final_classes() -> None:
    allowed = {"CORRECT", "CORRECTION", "MODIFICATION", "SLIP", "ADDITION"}
    for name in ("pipeline", "monolith"):
        for video in _load(name).values():
            for segment in video["segments"]:
                assert segment["classification"] in allowed
                assert segment["end"] > segment["start"]
