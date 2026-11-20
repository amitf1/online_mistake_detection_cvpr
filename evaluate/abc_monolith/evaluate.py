#!/usr/bin/env python3
"""Offline evaluation for a generated EgoPER ABC-monolith artifact."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

from online_mistake_detection.datasets import EgoPERProvider
from online_mistake_detection.metrics.abc_monolith import (
    monolith_runtime_metrics,
    validate_metric_manifest,
)
from online_mistake_detection.metrics.pipeline import evaluate_pipeline_events
from online_mistake_detection.pipeline.events import GroundTruthEvent, PipelineEvent
from online_mistake_detection.pipeline.runner import apply_graph_stage
from online_mistake_detection.schemas import WindowSpec


def _construct(cls: type, payload: dict[str, Any]) -> Any:
    allowed = {field.name for field in fields(cls)}
    return cls(**{key: value for key, value in payload.items() if key in allowed})


def _module_a_prediction(
    row: dict[str, Any],
) -> tuple[WindowSpec, str, str, float]:
    payload = dict(row["window"])
    payload["video_path"] = Path(payload["video_path"])
    for key in (
        "completed_steps",
        "pending_steps",
        "all_instructions",
        "completed_step_ids",
        "fine_classes",
    ):
        if key in payload:
            payload[key] = tuple(payload[key])
    window = _construct(WindowSpec, payload)
    return (
        window,
        str(row["prediction"]),
        str(row.get("raw_output") or ""),
        float(row.get("prediction_seconds") or 0.0),
    )


def evaluate_artifact(
    artifact: dict[str, Any],
    *,
    egoper_root: str | Path | None = None,
    require_complete_manifest: bool = False,
    compute_bertscore: bool = False,
    bertscore_model_type: str = "microsoft/deberta-xlarge-mnli",
    bertscore_batch_size: int = 16,
    bertscore_device: str = "auto",
) -> dict[str, Any]:
    if artifact.get("schema_version") != "egoper_abc_monolith_v1":
        raise ValueError("Expected schema_version='egoper_abc_monolith_v1'.")
    events = [_construct(PipelineEvent, row) for row in artifact.get("events", [])]
    ground_truth = [
        _construct(GroundTruthEvent, row)
        for row in artifact.get("ground_truth", [])
    ]
    dataset = str((artifact.get("metadata") or {}).get("dataset") or "egoper")
    if dataset == "egoper" and egoper_root is not None:
        video_ids = {event.video_id for event in ground_truth}
        provider = EgoPERProvider(egoper_root, video_ids=video_ids)
        apply_graph_stage(events, list(provider.iter_video_records()))
    module_a_predictions = [
        _module_a_prediction(row)
        for row in artifact.get("module_a_predictions", [])
    ]
    metrics = evaluate_pipeline_events(
        events,
        ground_truth,
        module_a_predictions=module_a_predictions or None,
        label_mode="step_id",
        dataset=dataset,
    )
    polls = list(artifact.get("polls", []))
    streamed_seconds = artifact.get("metadata", {}).get("streamed_video_seconds")
    gpu_count = int(artifact.get("metadata", {}).get("gpu_count", 1))
    metrics.update(
        monolith_runtime_metrics(
            polls,
            streamed_video_seconds=(
                None if streamed_seconds is None else float(streamed_seconds)
            ),
            gpu_count=gpu_count,
        )
    )
    bertscore_hash = None
    if compute_bertscore:
        from online_mistake_detection.training.module_c import (
            module_c_reasoning_metrics,
        )

        reasoning_events = [
            event
            for event in events
            if event.matched_gt_index is not None and not event.duplicate
        ]
        reasoning_metrics, bertscore_hash = module_c_reasoning_metrics(
            [str(event.gt_reasoning or "") for event in reasoning_events],
            [event.module_c_reasoning for event in reasoning_events],
            total_examples=len(reasoning_events),
            compute_bertscore=True,
            bertscore_model_type=bertscore_model_type,
            bertscore_batch_size=bertscore_batch_size,
            bertscore_device=bertscore_device,
        )
        metrics.update(
            {
                key.replace("eval_reasoning/", "reasoning/"): value
                for key, value in reasoning_metrics.items()
            }
        )
    if require_complete_manifest:
        validate_metric_manifest(metrics)
    return {
        **artifact,
        "events": [asdict(event) for event in events],
        "ground_truth": [asdict(event) for event in ground_truth],
        "metrics": metrics,
        "metadata": {
            **dict(artifact.get("metadata") or {}),
            "bertscore": {
                "enabled": compute_bertscore,
                "model_type": bertscore_model_type,
                "hash": bertscore_hash,
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--egoper-root", type=Path)
    parser.add_argument(
        "--require-complete-manifest",
        action=argparse.BooleanOptionalAction,
        default=False,
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
    payload = json.loads(args.artifacts_json.read_text(encoding="utf-8"))
    result = evaluate_artifact(
        payload,
        egoper_root=args.egoper_root,
        require_complete_manifest=args.require_complete_manifest,
        compute_bertscore=args.compute_bertscore,
        bertscore_model_type=args.bertscore_model_type,
        bertscore_batch_size=args.bertscore_batch_size,
        bertscore_device=args.bertscore_device,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
