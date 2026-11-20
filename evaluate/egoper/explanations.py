#!/usr/bin/env python3
"""BERTScore of the mistake explanations against the annotation and two controls.

    PYTHONPATH=src:. python evaluate/egoper/explanations.py \
        --predictions results/final_predictions --output results/explanations.json

Pairs: final segments matched one to one to an annotated segment of the same
step at IoU >= 0.5 (greedy by IoU), where the annotation is a mistake and the
prediction is any mistake class. Only pairs with a non-empty explanation are
scored.

Raw BERTScore F1 (microsoft/deberta-xlarge-mnli, layer 40, no IDF, no baseline
rescaling) of the explanation against:
  gold          the annotated description of the segment
  restatement   control: the step instruction ("off-script action" for an extra
                step) scored against the annotated description
  mismatched    control: the explanation against another annotated description of the
                same recipe (fixed derangement, seed 3407). Recipes where one
                description makes up more than half of the pairs are left out.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from online_mistake_detection.metrics.egoper import ERROR_CLASSES
from online_mistake_detection.metrics.temporal import temporal_iou

ROOT = Path(__file__).resolve().parents[2]
METHODS = (("pipeline", "Modular"), ("monolith", "Joint"))
SEED = 3407
MODEL = "microsoft/deberta-xlarge-mnli"
LAYERS = 40
EXTRA_INSTRUCTION = "off-script action"


def _step_index(segment: dict[str, Any]) -> int:
    return -1 if segment["step_id"] == "EXTRA" else int(segment["step_id"].removeprefix("STEP_"))


def mistake_pairs(predicted: dict[str, Any], gold: dict[str, Any], video_ids: list[str]) -> list[dict[str, str]]:
    candidates = [
        (temporal_iou((segment["start"], segment["end"]), (target["start"], target["end"])), video_id, i, j)
        for video_id in video_ids
        for i, segment in enumerate(predicted[video_id]["segments"])
        for j, target in enumerate(gold[video_id]["segments"])
        if _step_index(segment) == _step_index(target)
    ]
    used_segments: set[tuple[str, int]] = set()
    used_targets: set[tuple[str, int]] = set()
    rows = []
    for iou, video_id, i, j in sorted(candidates, reverse=True):
        if iou < 0.5 or (video_id, i) in used_segments or (video_id, j) in used_targets:
            continue
        used_segments.add((video_id, i))
        used_targets.add((video_id, j))
        segment, target = predicted[video_id]["segments"][i], gold[video_id]["segments"][j]
        if target["classification"] not in ERROR_CLASSES or segment["classification"] not in ERROR_CLASSES:
            continue
        rows.append(
            {
                "video_id": video_id,
                "task": gold[video_id]["task"],
                "reference": (target.get("reasoning") or "").strip(),
                "prediction": (segment.get("reasoning") or "").strip(),
                "instruction": (segment.get("step") or EXTRA_INSTRUCTION).strip(),
            }
        )
    return rows


def _derange(size: int, rng: np.random.Generator) -> list[int]:
    for _ in range(100):
        permutation = [int(index) for index in rng.permutation(size)]
        if all(source != target for source, target in enumerate(permutation)):
            return permutation
    return [(index + 1) % size for index in range(size)]


def mismatched_references(pairs: list[dict[str, str]], seed: int = SEED) -> list[str]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, pair in enumerate(pairs):
        grouped[pair["task"]].append(index)
    assigned = [""] * len(pairs)
    rng = np.random.default_rng(seed)
    for task in sorted(grouped):
        indexes = grouped[task]
        references = [pairs[index]["reference"] for index in indexes]
        if len(indexes) < 2 or len(set(references)) < 2:
            raise ValueError(f"{task}: the mismatched control needs two distinct references")
        permutation = _derange(len(indexes), rng)
        for _ in range(20):
            if all(references[permutation[index]] != references[index] for index in range(len(indexes))):
                break
            permutation = _derange(len(indexes), rng)
        else:
            raise ValueError(f"{task}: no derangement avoids a reference's own text")
        for local_index, source_index in enumerate(permutation):
            assigned[indexes[local_index]] = references[source_index]
    return assigned


def bertscore_f1(predictions: list[str], references: list[str], device: str) -> list[float]:
    import bert_score.utils as bert_utils
    from bert_score import score

    def sent_encode(tokenizer: Any, sentence: str) -> list[int]:
        limit = min(int(getattr(tokenizer, "model_max_length", 512) or 512), 512)
        return tokenizer.encode(sentence, add_special_tokens=True, max_length=limit, truncation=True)

    bert_utils.sent_encode = sent_encode
    (_precision, _recall, f1) = score(
        predictions,
        references,
        model_type=MODEL,
        num_layers=LAYERS,
        idf=False,
        rescale_with_baseline=False,
        batch_size=8,
        device=device,
        verbose=False,
    )
    return [float(value) for value in f1.detach().cpu().tolist()]


def _mean(values: list[float]) -> float:
    return 100.0 * sum(values) / len(values)


def controls(pairs: list[dict[str, str]], device: str) -> dict[str, Any]:
    scored = [pair for pair in pairs if pair["reference"] and pair["prediction"] and pair["instruction"]]
    by_task: dict[str, Counter[str]] = defaultdict(Counter)
    for pair in scored:
        by_task[pair["task"]][pair["reference"]] += 1
    excluded = sorted(task for task, refs in by_task.items() if 2 * max(refs.values()) > sum(refs.values()))
    mismatch_pairs = [pair for pair in scored if pair["task"] not in excluded]
    gold = bertscore_f1([p["prediction"] for p in scored], [p["reference"] for p in scored], device)
    restatement = bertscore_f1([p["instruction"] for p in scored], [p["reference"] for p in scored], device)
    mismatched = bertscore_f1(
        [p["prediction"] for p in mismatch_pairs], mismatched_references(mismatch_pairs), device
    )
    return {
        "mistake_pairs": len(pairs),
        "scored_pairs": len(scored),
        "mismatched_pairs": len(mismatch_pairs),
        "mismatched_excluded_recipes": excluded,
        "bertscore_f1": {
            "gold": _mean(gold),
            "instruction_restatement": _mean(restatement),
            "mismatched_reference": _mean(mismatched),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", type=Path, default=ROOT / "outputs/final_predictions")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/results/explanations.json")
    parser.add_argument("--device", default=None, help="cuda or cpu (default: cuda when available)")
    args = parser.parse_args()
    if args.device is None:
        import torch

        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    gold = json.loads((args.predictions / "ground_truth_test.json").read_text(encoding="utf-8"))["videos"]
    video_ids = sorted(gold)
    report: dict[str, Any] = {"model": MODEL, "num_layers": LAYERS, "idf": False, "rescale_with_baseline": False}
    for name, label in METHODS:
        path = args.predictions / f"{name}_test.json"
        if not path.is_file():
            continue
        predicted = json.loads(path.read_text(encoding="utf-8"))["videos"]
        report[name] = controls(mistake_pairs(predicted, gold, video_ids), args.device)
        scores = report[name]["bertscore_f1"]
        print(
            f"{label}: {report[name]['scored_pairs']} pairs | gold {scores['gold']:.2f} | "
            f"restatement {scores['instruction_restatement']:.2f} | mismatched {scores['mismatched_reference']:.2f}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
