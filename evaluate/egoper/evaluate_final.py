#!/usr/bin/env python3
"""System-level scores of the final prediction files against the annotation.

    PYTHONPATH=src:. python evaluate/egoper/evaluate_final.py \
        --predictions results/final_predictions --output results/metrics.json

Metrics (percent, one-to-one matching at temporal IoU >= 0.5), on all test
videos and on the erroneous videos (at least one annotated mistake):
  Predicted segments   final segments, by class
  EDA                  error detection accuracy (strict-majority coverage), recipe mean
  Binary F1@0.5        error/normal macro F1, recipe mean
  Fine P/R/F1@0.5      per mistake class, pooled over videos; macro = mean over the four classes
  TAS F1@0.5           segmental step F1, recipe mean
  O-Acc, O-IoU         EgoPER omission metrics, recipe mean (needs <EGOPER_ROOT>/task_graph.txt)
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from online_mistake_detection.metrics.egoper import (
    ERROR_CLASSES,
    binary_mistake_f1,
    egoper_omission_metrics,
    error_detection_accuracy,
    fine_mistake_f1,
    tas_f1,
)
from online_mistake_detection.paths import egoper_root
from online_mistake_detection.pipeline.events import GroundTruthEvent, PipelineEvent

ROOT = Path(__file__).resolve().parents[2]
METHODS = (("pipeline", "Modular"), ("monolith", "Joint"))
CLASSES = ("CORRECT", *ERROR_CLASSES)
THRESHOLD = 0.5


def _step_index(segment: dict[str, Any]) -> int:
    return -1 if segment["step_id"] == "EXTRA" else int(segment["step_id"].removeprefix("STEP_"))


def ground_truth_events(video_id: str, video: dict[str, Any]) -> list[GroundTruthEvent]:
    return [
        GroundTruthEvent(
            video_id=video_id,
            task_id=video["task"],
            step_index=_step_index(segment),
            instruction=segment["step"] or "undefined / extra step",
            gt_start=segment["start"],
            gt_end=segment["end"],
            gt_fine_class=segment["classification"],
            gt_mistake=segment["classification"] in {"MODIFICATION", "SLIP", "ADDITION"},
            gt_reasoning=segment["reasoning"],
            eligible=True,
        )
        for segment in video["segments"]
    ]


def predicted_events(video_id: str, video: dict[str, Any]) -> list[PipelineEvent]:
    return [
        PipelineEvent(
            video_id=video_id,
            task_id=video["task"],
            step_index=_step_index(segment),
            instruction=segment["step"] or "",
            trigger_time=segment["decided_at"],
            accumulation_start=segment["clip"][0],
            accumulation_end=segment["clip"][1],
            module_a_prediction=segment["step_id"],
            pred_global_start=segment["start"],
            pred_global_end=segment["end"],
            module_c_classification=segment["classification"],
            module_c_reasoning=segment.get("reasoning"),
            module_c_mistake=segment["classification"] != "CORRECT",
            stage_status="C_SKIPPED_EXTRA" if segment["step_id"] == "EXTRA" else "C_CLASSIFIED",
        )
        for segment in video["segments"]
    ]


def _pct(value: float) -> float:
    return 100.0 * value


def fine_scores(events: list[PipelineEvent], ground_truth: list[GroundTruthEvent]) -> dict[str, Any]:
    by_class = fine_mistake_f1(events, ground_truth, thresholds=[THRESHOLD])["by_threshold"]["0.5"]["per_class"]
    per_class = {}
    for name in ERROR_CLASSES:
        row = by_class.get(name, {"tp": 0.0, "fp": 0.0, "fn": 0.0})
        tp, fp, fn = row["tp"], row["fp"], row["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        per_class[name] = {
            "support": int(tp + fn),
            "precision": _pct(precision),
            "recall": _pct(recall),
            "f1": _pct(2 * tp / (2 * tp + fp + fn)) if tp + fp + fn else 0.0,
        }
    macro = {
        key: sum(row[key] for row in per_class.values()) / len(per_class) for key in ("precision", "recall", "f1")
    }
    return {"per_class": per_class, "macro": macro}


def score(events: list[PipelineEvent], ground_truth: list[GroundTruthEvent]) -> dict[str, Any]:
    classes = Counter(event.module_c_classification for event in events)
    binary = binary_mistake_f1(events, ground_truth, thresholds=[THRESHOLD])["by_threshold"]["0.5"]
    omission = egoper_omission_metrics(events, ground_truth)
    return {
        "predicted_segments": len(events),
        "predicted_error_segments": sum(bool(event.module_c_mistake) for event in events),
        "predicted_by_class": {name: classes.get(name, 0) for name in CLASSES},
        "eda": _pct(error_detection_accuracy(events, ground_truth)["majority"]["task_average_eda"]),
        "binary_f1_at_0.5": _pct(binary["task_average_macro_f1_at_0.5"]),
        "fine_at_0.5": fine_scores(events, ground_truth),
        "tas_f1_at_0.5": _pct(tas_f1(events, ground_truth, threshold=THRESHOLD)["task_average_f1_at_0.5"]),
        "o_acc": _pct(omission["task_average_o_acc"]),
        "o_iou": _pct(omission["task_average_o_iou"]),
    }


def evaluate_scope(sources: dict[str, dict[str, Any]], video_ids: list[str]) -> dict[str, Any]:
    ground_truth = [
        event for video_id in video_ids for event in ground_truth_events(video_id, sources["ground_truth"][video_id])
    ]
    classes = Counter(event.gt_fine_class for event in ground_truth)
    report: dict[str, Any] = {
        "videos": len(video_ids),
        "annotated_segments": len(ground_truth),
        "annotated_error_segments": sum(event.gt_fine_class != "CORRECT" for event in ground_truth),
        "annotated_by_class": {name: classes.get(name, 0) for name in CLASSES},
        "methods": {},
    }
    for name, _label in METHODS:
        if name not in sources:
            continue
        missing = sorted(set(video_ids) - set(sources[name]))
        if missing:
            raise SystemExit(f"{name}: no predictions for {len(missing)} videos, e.g. {missing[:3]}")
        events = [event for video_id in video_ids for event in predicted_events(video_id, sources[name][video_id])]
        report["methods"][name] = score(events, ground_truth)
    return report


def table(scope: str, report: dict[str, Any]) -> str:
    header = ["System", "Segments", "EDA", "Binary F1", "Fine P", "Fine R", "Fine F1", "TAS F1", "O-Acc", "O-IoU"]
    lines = [
        f"{scope}: {report['videos']} videos, {report['annotated_segments']} segments "
        f"({report['annotated_error_segments']} mistakes)",
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]
    for name, label in METHODS:
        row = report["methods"].get(name)
        if row is None:
            continue
        macro = row["fine_at_0.5"]["macro"]
        cells = [label, str(row["predicted_segments"])] + [
            f"{value:.2f}"
            for value in (
                row["eda"],
                row["binary_f1_at_0.5"],
                macro["precision"],
                macro["recall"],
                macro["f1"],
                row["tas_f1_at_0.5"],
                row["o_acc"],
                row["o_iou"],
            )
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", type=Path, default=ROOT / "outputs/final_predictions")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/results/metrics.json")
    args = parser.parse_args()

    sources: dict[str, dict[str, Any]] = {}
    for name in ("ground_truth", *(name for name, _label in METHODS)):
        path = args.predictions / f"{name}_test.json"
        if path.is_file():
            sources[name] = json.loads(path.read_text(encoding="utf-8"))["videos"]
        elif name == "ground_truth":
            raise SystemExit(f"Missing {path}")
    if not any(name in sources for name, _label in METHODS):
        raise SystemExit(f"No pipeline_test.json or monolith_test.json in {args.predictions}")
    if not (egoper_root() / "task_graph.txt").is_file():
        print(f"warning: {egoper_root() / 'task_graph.txt'} not found; O-Acc and O-IoU use steps seen in the annotation")

    all_test = sorted(sources["ground_truth"])
    erroneous = sorted(
        video_id
        for video_id, video in sources["ground_truth"].items()
        if any(segment["classification"] != "CORRECT" for segment in video["segments"])
    )
    report = {
        "iou_threshold": THRESHOLD,
        "all_test": evaluate_scope(sources, all_test),
        "erroneous_videos": evaluate_scope(sources, erroneous),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for scope in ("all_test", "erroneous_videos"):
        print(table(scope, report[scope]))
        print()
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
