#!/usr/bin/env python3
"""Component-level scores of Modules A, B and C from the Modular test artifact.

    PYTHONPATH=src:. python evaluate/egoper/component_metrics.py \
        --pipeline-artifact outputs/online_pipeline_eval/pipeline_egoper_27b_test/pipeline_artifacts.json \
        --output outputs/results/component_metrics.json

Every model call is scored, including "not completed" answers and calls that
were repeated on a longer clip.

Poll target: the first annotated segment that ends inside the poll's clip and
is not already covered (IoU >= 0.5) by a span the stream accepted at an
earlier poll. Without one, the target letter is A (no step completed).

Module A: step-ID macro-F1 and accuracy over all polls.
Module B: a span is a hit at IoU >= 0.5 with the poll's target. Precision is
over returned spans, recall over calls with a target.
Module C: the gold class is the fine class of the annotated segment the span
overlaps most, or NO_MATCH below IoU 0.5. A "not completed" answer is NO_MATCH.

"Upstream correct" keeps calls whose inputs were right: for Module B, Module A
named the target step; for Module C, also Module B's IoU with the target >= 0.5.
"""

from __future__ import annotations

import argparse
import json
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from online_mistake_detection.metrics.module_a import multiclass_letter_metrics
from online_mistake_detection.metrics.temporal import temporal_iou

ROOT = Path(__file__).resolve().parents[2]
PIPELINE = ROOT / "outputs/online_pipeline_eval/pipeline_egoper_27b_test/pipeline_artifacts.json"
OUTPUT = ROOT / "outputs/results/component_metrics.json"
FULL_MATCH_IOU = 0.5
FINE_CLASSES = ("CORRECT", "CORRECTION", "MODIFICATION", "SLIP")
C_CLASSES = (*FINE_CLASSES, "NO_MATCH")
C_CALLED = frozenset({"C_CLASSIFIED", "C_NOT_COMPLETED", "C_INVALID_PARSE"})
NOT_ACCEPTED = frozenset({"B_NOT_COMPLETED", "C_NOT_COMPLETED", "C_INVALID_PARSE"})
EPS = 1e-9


def letter_for_step(step_index: int) -> str:
    return "B" if step_index < 0 else string.ascii_uppercase[step_index + 2]


def normalize_letter(label: str | None) -> str:
    text = str(label or "").strip().upper()
    if text in {"", "NONE", "WAIT", "NOT_COMPLETED"}:
        return "A"
    if text == "EXTRA":
        return "B"
    if text.startswith("STEP_"):
        return letter_for_step(int(text.removeprefix("STEP_")))
    return text[:1]


def _key(video_id: Any, start: Any, end: Any) -> tuple[str, float, float]:
    return (str(video_id), round(float(start), 4), round(float(end), 4))


@dataclass
class Gold:
    start: float
    end: float
    letter: str
    fine_class: str
    position: int


@dataclass
class Poll:
    video_id: str
    start: float
    end: float
    prediction: str
    event: dict[str, Any] | None


def load(payload: dict[str, Any]) -> tuple[dict[str, list[Gold]], dict[str, list[Poll]]]:
    gold: dict[str, list[Gold]] = {}
    for row in payload["ground_truth"]:
        if row.get("eligible", True):
            gold.setdefault(str(row["video_id"]), []).append(
                Gold(
                    start=float(row["gt_start"]),
                    end=float(row["gt_end"]),
                    letter=letter_for_step(int(row["step_index"])),
                    fine_class=str(row.get("gt_fine_class") or ""),
                    position=int(row.get("annotation_position") or 0),
                )
            )
    events = {
        _key(event["video_id"], event["accumulation_start"], event["accumulation_end"]): event
        for event in payload["events"]
    }
    if len(events) != len(payload["events"]):
        raise ValueError("two events share one clip")
    polls: dict[str, list[Poll]] = {}
    for row in payload["module_a_predictions"]:
        window = row["window"]
        video_id = str(window["video_id"])
        polls.setdefault(video_id, []).append(
            Poll(
                video_id=video_id,
                start=float(window["window_start"]),
                end=float(window["window_end"]),
                prediction=normalize_letter(row.get("prediction")),
                event=events.get(_key(video_id, window["window_start"], window["window_end"])),
            )
        )
    joined = sum(poll.event is not None for rows in polls.values() for poll in rows)
    if joined != len(events):
        raise ValueError(f"joined {joined} of {len(events)} events to polls")
    for rows in polls.values():
        rows.sort(key=lambda poll: (poll.start, poll.end))
    return gold, polls


def _accepted(event: dict[str, Any] | None) -> bool:
    return (
        event is not None
        and event.get("stage_status") not in NOT_ACCEPTED
        and event.get("pred_global_end") is not None
    )


def _span(event: dict[str, Any]) -> tuple[float, float] | None:
    if event.get("stage_status") == "B_NOT_COMPLETED" or event.get("pred_global_end") is None:
        return None
    return float(event["pred_global_start"]), float(event["pred_global_end"])


def _best_gold(span: tuple[float, float], gold: list[Gold]) -> tuple[Gold | None, float]:
    best: tuple[Gold | None, float] = (None, 0.0)
    for row in gold:
        iou = temporal_iou(span, (row.start, row.end))
        if iou > best[1]:
            best = (row, iou)
    return best


def poll_rows(polls: list[Poll], gold: list[Gold]) -> list[dict[str, Any]]:
    accepted = [(poll.end, _span(poll.event)) for poll in polls if _accepted(poll.event)]
    ordered = sorted(gold, key=lambda row: (row.end, row.start, row.position))
    rejected_clips = [
        (poll.start, poll.end)
        for poll in polls
        if poll.event is not None and poll.event.get("stage_status") == "C_NOT_COMPLETED"
    ]
    rows = []
    for poll in polls:
        earlier = [span for placed_at, span in accepted if placed_at < poll.end - EPS]
        target = next(
            (
                row
                for row in ordered
                if poll.start + EPS < row.end <= poll.end + EPS
                and not any(temporal_iou(span, (row.start, row.end)) >= FULL_MATCH_IOU for span in earlier)
            ),
            None,
        )
        retry = any(abs(start - poll.start) < 1e-4 and end < poll.end - EPS for start, end in rejected_clips)
        rows.append({"poll": poll, "target": target, "letter": target.letter if target else "A", "retry": retry})
    return rows


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def module_b_scores(calls: list[dict[str, Any]]) -> dict[str, Any]:
    spans = [call for call in calls if call["span"] is not None]
    with_target = [call for call in calls if call["target"] is not None]
    scored = [call for call in spans if call["target"] is not None]
    hits = sum(call["iou"] >= FULL_MATCH_IOU for call in scored)
    precision = _ratio(hits, len(spans)) or 0.0
    recall = _ratio(hits, len(with_target)) or 0.0
    return {
        "calls": len(calls),
        "spans": len(spans),
        "not_completed": len(calls) - len(spans),
        "calls_with_target": len(with_target),
        "precision_at_0.5": precision,
        "recall_at_0.5": recall,
        "f1_at_0.5": _f1(precision, recall),
        "mean_iou": _ratio(sum(call["iou"] for call in scored), len(scored)),
        "boundary_mae_seconds": _ratio(
            sum(
                (abs(call["span"][0] - call["target"].start) + abs(call["span"][1] - call["target"].end)) / 2
                for call in scored
            ),
            len(scored),
        ),
    }


def module_c_scores(calls: list[dict[str, Any]]) -> dict[str, Any]:
    per_class = {}
    for label in C_CLASSES:
        tp = sum(call["gold"] == label and call["predicted"] == label for call in calls)
        support = sum(call["gold"] == label for call in calls)
        predicted = sum(call["predicted"] == label for call in calls)
        precision = _ratio(tp, predicted) or 0.0
        recall = _ratio(tp, support) or 0.0
        per_class[label] = {
            "support": support,
            "predicted": predicted,
            "tp": tp,
            "f1": _f1(precision, recall) if support or predicted else None,
        }

    def macro(labels: tuple[str, ...]) -> float | None:
        values = [per_class[label]["f1"] for label in labels if per_class[label]["f1"] is not None]
        return sum(values) / len(values) if values else None

    return {
        "calls": len(calls),
        "not_completed": sum(call["status"] == "C_NOT_COMPLETED" for call in calls),
        "accuracy": _ratio(sum(call["gold"] == call["predicted"] for call in calls), len(calls)),
        "fine_macro_f1": macro(FINE_CLASSES),
        "macro_f1_with_no_match": macro(C_CLASSES),
        "per_class": per_class,
    }


def breakdown(gold: dict[str, list[Gold]], polls: dict[str, list[Poll]], videos: list[str]) -> dict[str, Any]:
    rows = [row for video_id in videos for row in poll_rows(polls.get(video_id, []), gold.get(video_id, []))]
    a = multiclass_letter_metrics([row["letter"] for row in rows], [row["poll"].prediction for row in rows], prefix="a")
    b_calls, c_calls = [], []
    for row in rows:
        event = row["poll"].event
        if event is None:
            continue
        target = row["target"]
        span = _span(event)
        iou = temporal_iou(span, (target.start, target.end)) if span and target else 0.0
        a_correct = target is not None and row["poll"].prediction == row["letter"]
        b_calls.append({"target": target, "span": span, "iou": iou, "a_correct": a_correct, "retry": row["retry"]})
        if event.get("stage_status") not in C_CALLED:
            continue
        overlap, overlap_iou = _best_gold(span, gold.get(row["poll"].video_id, []))
        predicted = event.get("module_c_classification")
        c_calls.append(
            {
                "gold": overlap.fine_class if overlap and overlap_iou >= FULL_MATCH_IOU else "NO_MATCH",
                "predicted": predicted if predicted in FINE_CLASSES else "NO_MATCH",
                "status": event.get("stage_status"),
                "upstream_correct": a_correct and iou >= FULL_MATCH_IOU,
                "retry": row["retry"],
            }
        )
    return {
        "videos": len(videos),
        "module_a": {
            "polls": len(rows),
            "not_completed": sum(row["poll"].prediction == "A" for row in rows),
            "retries": sum(row["retry"] for row in rows),
            "step_id_macro_f1": float(a["a/macro_f1"]),
            "accuracy": float(a["a/accuracy"]),
        },
        "module_b": {
            "retries": sum(call["retry"] for call in b_calls),
            "spans_rejected_by_module_c": sum(call["status"] == "C_NOT_COMPLETED" for call in c_calls),
            "every_call": module_b_scores(b_calls),
            "upstream_correct": module_b_scores([call for call in b_calls if call["a_correct"]]),
        },
        "module_c": {
            "retries": sum(call["retry"] for call in c_calls),
            "every_call": module_c_scores(c_calls),
            "upstream_correct": module_c_scores([call for call in c_calls if call["upstream_correct"]]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pipeline-artifact", type=Path, default=PIPELINE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if not args.pipeline_artifact.is_file():
        raise SystemExit(f"Missing test artifact {args.pipeline_artifact}. Run the Modular test first (see README).")
    payload = json.loads(args.pipeline_artifact.read_text(encoding="utf-8"))
    gold, polls = load(payload)
    videos = sorted(gold)
    erroneous = sorted(video_id for video_id, rows in gold.items() if any(row.fine_class != "CORRECT" for row in rows))
    report = {
        "all_test": breakdown(gold, polls, videos),
        "erroneous_videos": breakdown(gold, polls, erroneous),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for scope, row in report.items():
        a, b, c = row["module_a"], row["module_b"], row["module_c"]
        print(
            f"{scope} ({row['videos']} videos): "
            f"A macro-F1 {100 * a['step_id_macro_f1']:.2f} | "
            f"B F1@0.5 {100 * b['every_call']['f1_at_0.5']:.2f} / {100 * b['upstream_correct']['f1_at_0.5']:.2f} | "
            f"C fine macro-F1 {100 * c['every_call']['fine_macro_f1']:.2f} / {100 * c['upstream_correct']['fine_macro_f1']:.2f}"
        )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
