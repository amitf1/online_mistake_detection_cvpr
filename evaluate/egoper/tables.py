#!/usr/bin/env python3
"""Render the result JSON files as the paper's tables, and optionally compare two result folders.

    PYTHONPATH=src:. python evaluate/egoper/tables.py --results results
    PYTHONPATH=src:. python evaluate/egoper/tables.py --results outputs/results --compare results

Reads metrics.json, component_metrics.json, timing.json and explanations.json
(the last three are optional) and writes tables.md next to them.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterator

FILES = ("metrics.json", "component_metrics.json", "timing.json", "explanations.json")
SYSTEMS = (("pipeline", "Modular"), ("monolith", "Joint"))
SCOPES = (("all_test", "All test videos"), ("erroneous_videos", "Erroneous videos"))
ERROR_CLASSES = ("CORRECTION", "MODIFICATION", "SLIP", "ADDITION")
FINE_CLASSES = ("CORRECT", "CORRECTION", "MODIFICATION", "SLIP")


def _row(cells: list[Any]) -> str:
    return "| " + " | ".join(str(cell) for cell in cells) + " |"


def _table(header: list[str], rows: list[list[Any]]) -> list[str]:
    return [_row(header), "|" + "---|" * len(header), *(_row(row) for row in rows), ""]


def _f(value: float | None, scale: float = 1.0) -> str:
    return "—" if value is None else f"{scale * value:.2f}"


def system_tables(metrics: dict[str, Any]) -> list[str]:
    lines = ["## System level (IoU 0.5)", ""]
    header = ["System", "EDA", "Binary F1", "Fine P", "Fine R", "Fine F1", "TAS F1", "O-Acc", "O-IoU"]
    for scope, title in SCOPES:
        report = metrics[scope]
        lines += [f"### {title} ({report['videos']} videos)", ""]
        rows = []
        for name, label in SYSTEMS:
            row = report["methods"].get(name)
            if row is None:
                continue
            macro = row["fine_at_0.5"]["macro"]
            rows.append(
                [label]
                + [
                    _f(value)
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
            )
        lines += _table(header, rows)
    lines += ["### Fine-grained mistakes per class (precision / recall / F1)", ""]
    for scope, title in SCOPES:
        report = metrics[scope]
        rows = []
        for name, label in SYSTEMS:
            row = report["methods"].get(name)
            if row is None:
                continue
            per_class = row["fine_at_0.5"]["per_class"]
            rows.append(
                [label]
                + [
                    f"{per_class[c]['precision']:.2f} / {per_class[c]['recall']:.2f} / {per_class[c]['f1']:.2f}"
                    for c in ERROR_CLASSES
                ]
            )
        lines += [f"{title}:", ""] + _table(["System", *[c.title() for c in ERROR_CLASSES]], rows)
    lines += ["### Predicted segments", ""]
    header = ["Scope", "Source", "Segments", "Errors", "Correct", *[c.title() for c in ERROR_CLASSES]]
    rows = []
    for scope, title in SCOPES:
        report = metrics[scope]
        by_class = report["annotated_by_class"]
        rows.append(
            [title, "Annotation", report["annotated_segments"], report["annotated_error_segments"], by_class["CORRECT"]]
            + [by_class[c] for c in ERROR_CLASSES]
        )
        for name, label in SYSTEMS:
            row = report["methods"].get(name)
            if row is None:
                continue
            by_class = row["predicted_by_class"]
            rows.append(
                [title, label, row["predicted_segments"], row["predicted_error_segments"], by_class["CORRECT"]]
                + [by_class[c] for c in ERROR_CLASSES]
            )
    lines += _table(header, rows)
    return lines


def component_tables(component: dict[str, Any]) -> list[str]:
    lines = ["## Component level (Modular)", ""]
    rows = []
    for scope, title in SCOPES:
        row = component[scope]
        a, b, c = row["module_a"], row["module_b"], row["module_c"]
        rows += [
            [title, "A", a["polls"], a["not_completed"], a["retries"]],
            [title, "B", b["every_call"]["calls"], b["every_call"]["not_completed"], b["retries"]],
            [title, "C", c["every_call"]["calls"], c["every_call"]["not_completed"], c["retries"]],
        ]
    lines += ["Scored calls, \"not completed\" answers and repeated calls on a longer clip:", ""]
    lines += _table(["Scope", "Module", "Calls", "Not completed", "Repeated"], rows)
    rows = [
        [title, component[scope]["module_a"]["polls"], _f(component[scope]["module_a"]["step_id_macro_f1"], 100),
         _f(component[scope]["module_a"]["accuracy"], 100)]
        for scope, title in SCOPES
    ]
    lines += ["### Module A", ""] + _table(["Scope", "Polls", "Step-ID macro-F1", "Accuracy"], rows)
    rows = []
    for scope, title in SCOPES:
        for key, calls in (("every_call", "every call"), ("upstream_correct", "upstream correct")):
            b = component[scope]["module_b"][key]
            rows.append(
                [title, calls, b["calls"], _f(b["precision_at_0.5"], 100), _f(b["recall_at_0.5"], 100),
                 _f(b["f1_at_0.5"], 100), _f(b["mean_iou"], 100), _f(b["boundary_mae_seconds"])]
            )
    lines += ["### Module B", ""] + _table(
        ["Scope", "Calls", "N", "P@0.5", "R@0.5", "F1@0.5", "Mean IoU", "Boundary MAE (s)"], rows
    )
    rows = []
    for scope, title in SCOPES:
        for key, calls in (("every_call", "every call"), ("upstream_correct", "upstream correct")):
            c = component[scope]["module_c"][key]
            per_class = c["per_class"]
            rows.append(
                [title, calls, c["calls"], _f(c["accuracy"], 100), _f(c["fine_macro_f1"], 100),
                 _f(c["macro_f1_with_no_match"], 100)]
                + [_f(per_class[name]["f1"], 100) for name in FINE_CLASSES]
            )
    lines += ["### Module C", ""] + _table(
        ["Scope", "Calls", "N", "Accuracy", "Fine macro-F1", "Macro-F1 with NO_MATCH",
         *[f"{name.title()} F1" for name in FINE_CLASSES]],
        rows,
    )
    return lines


def timing_table(timing: dict[str, Any]) -> list[str]:
    labels = {
        ("modular", "module_a_per_call"): ("Modular", "Module A, per call"),
        ("modular", "module_b_per_call"): ("Modular", "Module B, per call"),
        ("modular", "module_c_per_call"): ("Modular", "Module C, per call"),
        ("modular", "a_b_c_per_final_output"): ("Modular", "A + B + C, per final output"),
        ("modular", "per_video"): ("Modular", "Summed per video"),
        ("joint", "per_call"): ("Joint", "Per call"),
        ("joint", "per_final_output"): ("Joint", "Per final output"),
        ("joint", "per_video"): ("Joint", "Summed per video"),
    }
    rows = [
        [*labels[(system, name)], row["n"], _f(row["mean"]), _f(row["median"]), _f(row["p95"])]
        for system, values in timing.items()
        for name, row in values.items()
    ]
    return ["## Inference time (seconds)", ""] + _table(["System", "Measurement", "N", "Mean", "Median", "P95"], rows)


def explanation_table(explanations: dict[str, Any]) -> list[str]:
    rows = []
    for name, label in SYSTEMS:
        row = explanations.get(name)
        if row is None:
            continue
        scores = row["bertscore_f1"]
        rows.append(
            [label, row["scored_pairs"], _f(scores["gold"]), _f(scores["instruction_restatement"]),
             _f(scores["mismatched_reference"])]
        )
    return ["## Explanations (raw BERTScore F1)", ""] + _table(
        ["System", "Pairs", "Gold", "Instruction restatement", "Mismatched reference"], rows
    )


def render(folder: Path) -> str:
    data = {name: json.loads((folder / name).read_text(encoding="utf-8")) for name in FILES if (folder / name).is_file()}
    if "metrics.json" not in data:
        raise SystemExit(f"Missing {folder / 'metrics.json'}")
    lines = system_tables(data["metrics.json"])
    if "component_metrics.json" in data:
        lines += component_tables(data["component_metrics.json"])
    if "explanations.json" in data:
        lines += explanation_table(data["explanations.json"])
    if "timing.json" in data:
        lines += timing_table(data["timing.json"])
    return "\n".join(lines)


def _leaves(value: Any, prefix: str = "") -> Iterator[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _leaves(item, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(value, list):
        yield prefix, value
    else:
        yield prefix, value


def compare(folder: Path, reference: Path, tolerance: float) -> int:
    differences = 0
    for name in FILES:
        if name == "timing.json" or not (folder / name).is_file() or not (reference / name).is_file():
            continue
        ours = dict(_leaves(json.loads((folder / name).read_text(encoding="utf-8"))))
        theirs = dict(_leaves(json.loads((reference / name).read_text(encoding="utf-8"))))
        for key in sorted(set(ours) | set(theirs)):
            left, right = ours.get(key), theirs.get(key)
            same = (
                math.isclose(float(left), float(right), abs_tol=tolerance)
                if isinstance(left, (int, float)) and isinstance(right, (int, float))
                else left == right
            )
            if not same:
                differences += 1
                print(f"{name}: {key}: {left} vs {right}")
    print(f"{differences} differences (timing.json is not compared)")
    return differences


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=Path("outputs/results"))
    parser.add_argument("--compare", type=Path, default=None, help="another result folder, e.g. results")
    parser.add_argument("--tolerance", type=float, default=0.005, help="absolute tolerance; scores are in percent")
    args = parser.parse_args()
    text = render(args.results)
    (args.results / "tables.md").write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"wrote {args.results / 'tables.md'}")
    if args.compare is not None:
        print()
        compare(args.results, args.compare, args.tolerance)


if __name__ == "__main__":
    main()
