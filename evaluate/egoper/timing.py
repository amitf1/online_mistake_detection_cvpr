#!/usr/bin/env python3
"""Inference time of the Modular and Joint test runs, from the test artifacts.

    PYTHONPATH=src:. python evaluate/egoper/timing.py \
        --pipeline-artifact outputs/online_pipeline_eval/pipeline_egoper_27b_test/pipeline_artifacts.json \
        --monolith-artifact outputs/abc_monolith/eval/abc_monolith_egoper_27b_test/monolith_test_artifact.json \
        --output outputs/results/timing.json

Measured generation time of every model call, in seconds (mean, median, p95).
Every call counts, including calls repeated on a longer clip. Module C's
candidate-scoring pass and the Joint model's scoring pass are not included.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
PIPELINE = ROOT / "outputs/online_pipeline_eval/pipeline_egoper_27b_test/pipeline_artifacts.json"
MONOLITH = ROOT / "outputs/abc_monolith/eval/abc_monolith_egoper_27b_test/monolith_test_artifact.json"
OUTPUT = ROOT / "outputs/results/timing.json"
FINAL = frozenset({"C_CLASSIFIED", "C_SKIPPED_EXTRA"})


def stats(values: Iterable[Any]) -> dict[str, float]:
    durations = sorted(float(value) for value in values if isinstance(value, (int, float)) and value >= 0)
    if not durations:
        raise ValueError("no measured durations")

    def percentile(q: float) -> float:
        position = (len(durations) - 1) * q
        lower = int(position)
        upper = min(lower + 1, len(durations) - 1)
        return durations[lower] + (durations[upper] - durations[lower]) * (position - lower)

    return {
        "n": len(durations),
        "mean": sum(durations) / len(durations),
        "median": percentile(0.5),
        "p95": percentile(0.95),
        "total": sum(durations),
    }


def modular(payload: dict[str, Any]) -> dict[str, Any]:
    polls = payload["module_a_predictions"]
    events = payload["events"]
    per_video: dict[str, float] = defaultdict(float)
    for row in polls:
        per_video[str(row["window"]["video_id"])] += float(row.get("prediction_seconds") or 0.0)
    for event in events:
        for key in ("module_b_prediction_seconds", "module_c_prediction_seconds"):
            per_video[str(event["video_id"])] += float(event.get(key) or 0.0)
    return {
        "module_a_per_call": stats(row.get("prediction_seconds") for row in polls),
        "module_b_per_call": stats(event.get("module_b_prediction_seconds") for event in events),
        "module_c_per_call": stats(event.get("module_c_prediction_seconds") for event in events),
        "a_b_c_per_final_output": stats(
            event.get("pipeline_prediction_seconds") for event in events if event.get("stage_status") in FINAL
        ),
        "per_video": stats(per_video.values()),
    }


def joint(payload: dict[str, Any]) -> dict[str, Any]:
    polls = payload["polls"]
    per_video: dict[str, float] = defaultdict(float)
    for row in polls:
        per_video[str(row["video_id"])] += float(row.get("monolith_prediction_seconds") or 0.0)
    return {
        "per_call": stats(row.get("monolith_prediction_seconds") for row in polls),
        "per_final_output": stats(
            row.get("monolith_prediction_seconds") for row in polls if row.get("completed") is True
        ),
        "per_video": stats(per_video.values()),
    }


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"Missing test artifact {path}. Run the test step first (see README).")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pipeline-artifact", type=Path, default=PIPELINE)
    parser.add_argument("--monolith-artifact", type=Path, default=MONOLITH)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    report = {"modular": modular(_read(args.pipeline_artifact)), "joint": joint(_read(args.monolith_artifact))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for system, rows in report.items():
        for name, row in rows.items():
            print(f"{system:8s} {name:24s} n={row['n']:5d} mean={row['mean']:7.2f} median={row['median']:7.2f} p95={row['p95']:7.2f}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
