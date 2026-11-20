#!/usr/bin/env python3
"""Evaluate previously generated online-pipeline events."""

import argparse
import json
from dataclasses import asdict, fields
from pathlib import Path

from online_mistake_detection.metrics.pipeline import evaluate_pipeline_events
from online_mistake_detection.pipeline.events import GroundTruthEvent, PipelineEvent
from online_mistake_detection.schemas import WindowSpec


def _construct(cls: type, payload: dict):
    allowed = {field.name for field in fields(cls)}
    return cls(**{key: value for key, value in payload.items() if key in allowed})


def _module_a_prediction(row: dict) -> tuple[WindowSpec, str, str, float]:
    window_payload = dict(row["window"])
    window_payload["video_path"] = Path(window_payload["video_path"])
    for key in (
        "completed_steps",
        "pending_steps",
        "all_instructions",
        "completed_step_ids",
        "fine_classes",
    ):
        if key in window_payload:
            window_payload[key] = tuple(window_payload[key])
    window = _construct(WindowSpec, window_payload)
    return (
        window,
        str(row["prediction"]),
        str(row.get("raw_output") or ""),
        float(row.get("prediction_seconds") or 0.0),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifacts-json",
        type=Path,
        help="Self-contained pipeline_artifacts.json produced with --defer-metrics.",
    )
    parser.add_argument("--events-json", type=Path)
    parser.add_argument("--ground-truth-json", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        choices=("egoper",),
        default="egoper",
    )
    parser.add_argument(
        "--compute-bertscore",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--bertscore-model-type",
        default="microsoft/deberta-xlarge-mnli",
    )
    parser.add_argument("--bertscore-batch-size", type=int, default=16)
    parser.add_argument("--bertscore-device", default="auto")
    args = parser.parse_args()
    if args.artifacts_json:
        if args.events_json or args.ground_truth_json:
            parser.error("--artifacts-json cannot be combined with separate event/GT inputs.")
        artifact_payload = json.loads(args.artifacts_json.read_text(encoding="utf-8"))
        events_payload = artifact_payload["events"]
        gt_payload = artifact_payload["ground_truth"]
        module_a_predictions = [
            _module_a_prediction(row)
            for row in artifact_payload.get("module_a_predictions", [])
        ]
        run_args = artifact_payload.get("args", {})
        label_mode = str(run_args.get("module_a_label_mode", "step_id"))
    else:
        if not args.events_json or not args.ground_truth_json:
            parser.error(
                "Provide --artifacts-json or both --events-json and --ground-truth-json."
            )
        artifact_payload = None
        events_payload = json.loads(args.events_json.read_text(encoding="utf-8"))
        gt_payload = json.loads(args.ground_truth_json.read_text(encoding="utf-8"))
        module_a_predictions = None
        label_mode = "step_id"
    events = [_construct(PipelineEvent, row) for row in events_payload]
    ground_truth = [_construct(GroundTruthEvent, row) for row in gt_payload]
    metrics = evaluate_pipeline_events(
        events,
        ground_truth,
        module_a_predictions=module_a_predictions,
        label_mode=label_mode,
        dataset=args.dataset,
    )
    bertscore_hash = None
    if args.compute_bertscore:
        from online_mistake_detection.training.module_c import (
            module_c_reasoning_metrics,
        )

        matched_events = [
            event
            for event in events
            if event.matched_gt_index is not None and not event.duplicate
        ]
        reasoning_metrics, bertscore_hash = module_c_reasoning_metrics(
            [str(event.gt_reasoning or "") for event in matched_events],
            [
                None
                if event.module_c_reasoning is None
                else str(event.module_c_reasoning)
                for event in matched_events
            ],
            total_examples=len(matched_events),
            compute_bertscore=True,
            bertscore_model_type=args.bertscore_model_type,
            bertscore_batch_size=args.bertscore_batch_size,
            bertscore_device=args.bertscore_device,
        )
        metrics.update(
            {
                key.replace("eval_reasoning/", "reasoning/"): value
                for key, value in reasoning_metrics.items()
            }
        )
        metrics["reasoning/bertscore_hash"] = bertscore_hash
    output_payload = (
        {
            **artifact_payload,
            "metrics": metrics,
            "events": [asdict(event) for event in events],
            "metrics_source_artifact": str(args.artifacts_json),
            "reasoning_metric_config": {
                "bertscore_enabled": args.compute_bertscore,
                "bertscore_model_type": args.bertscore_model_type,
                "bertscore_batch_size": args.bertscore_batch_size,
                "bertscore_device": args.bertscore_device,
                "bertscore_hash": bertscore_hash,
            },
        }
        if artifact_payload is not None
        else metrics
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(output_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
