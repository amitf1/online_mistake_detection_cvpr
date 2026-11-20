#!/usr/bin/env python3
"""Final per-video outputs of the Modular and Joint test runs, plus the annotation.

    PYTHONPATH=src:. python evaluate/egoper/final_predictions.py

Writes pipeline_test.json (Modular), monolith_test.json (Joint) and
ground_truth_test.json. Each holds only the system's final decisions.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from online_mistake_detection.metrics.egoper import _annotated_addition_steps
from online_mistake_detection.pipeline.events import (
    GroundTruthEvent,
    PipelineEvent,
    events_from_rows,
    predicted_step_index,
)

ROOT = Path(__file__).resolve().parents[2]
PIPELINE = ROOT / "outputs/online_pipeline_eval/pipeline_egoper_27b_test/pipeline_artifacts.json"
MONOLITH = ROOT / "outputs/abc_monolith/eval/abc_monolith_egoper_27b_test/monolith_test_artifact.json"
OUTPUT_DIR = ROOT / "outputs/final_predictions"
PIPELINE_ENV = ROOT / "configs/env/pipeline_test.env"
MONOLITH_ENV = ROOT / "configs/env/monolith_test.env"
CLASSES = ("CORRECT", "CORRECTION", "MODIFICATION", "SLIP", "ADDITION")
FINAL_STATUSES = frozenset({"C_CLASSIFIED", "C_SKIPPED_EXTRA"})
IS_ERROR_FIELD = (
    "classification is not CORRECT; this is the error label EDA and binary F1 score, "
    "so CORRECTION counts as an error"
)


def require(path: Path) -> Path:
    if not path.is_file():
        raise SystemExit(f"Missing test artifact {path}. Run the test step first (see README).")
    return path


def display_path(value: str | Path | None) -> str | None:
    if value is None:
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = ROOT / path
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return path.name


def _seconds(value: float) -> float:
    return float(value)


def _step(event: PipelineEvent) -> tuple[str, str | None]:
    step = predicted_step_index(event)
    if step is None:
        raise ValueError(f"{event.video_id} at {event.trigger_time}: no predicted step")
    if step < 0:
        return "EXTRA", None
    return f"STEP_{step}", event.instruction


def is_final(event: PipelineEvent) -> bool:
    return (
        event.stage_status in FINAL_STATUSES
        and event.module_c_classification in CLASSES
        and event.pred_global_start is not None
        and event.pred_global_end is not None
        and float(event.pred_global_end) > float(event.pred_global_start)
    )


def _model_output(event: PipelineEvent) -> dict[str, Any]:
    text = str(event.module_c_raw or "").removesuffix("<|im_end|>").strip()
    return json.loads(text)


def segment(event: PipelineEvent, *, method: str) -> dict[str, Any]:
    step_id, step = _step(event)
    row: dict[str, Any] = {
        "start": _seconds(event.pred_global_start),
        "end": _seconds(event.pred_global_end),
        "step_id": step_id,
        "step": step,
        "classification": event.module_c_classification,
        "is_error": event.module_c_classification != "CORRECT",
        "decided_at": _seconds(event.trigger_time),
        "clip": [_seconds(event.accumulation_start), _seconds(event.accumulation_end)],
    }
    if method == "monolith":
        output = _model_output(event)
        if output.get("step_id") != step_id or output.get("classification") != row["classification"]:
            raise ValueError(f"{event.video_id} at {event.trigger_time}: final fields differ from the model output")
        row["reasoning"] = output.get("reasoning")
        row["model_output"] = output
        row["clamped_to_clip"] = bool(event.module_b_was_clamped)
    else:
        row["reasoning"] = event.module_c_reasoning
    return row


def build(artifact: dict[str, Any], *, method: str, source: Path) -> dict[str, Any]:
    events = events_from_rows(PipelineEvent, artifact["events"])
    video_ids = sorted({row["video_id"] for row in artifact["ground_truth"]})
    tasks = {row["video_id"]: row["task_id"] for row in artifact["ground_truth"]}
    final = [event for event in events if is_final(event)]
    videos = {
        video_id: {
            "task": tasks[video_id],
            "segments": [
                segment(event, method=method)
                for event in sorted(
                    (event for event in final if event.video_id == video_id),
                    key=lambda event: (float(event.pred_global_start), float(event.pred_global_end)),
                )
            ],
        }
        for video_id in video_ids
    }
    classes = Counter(event.module_c_classification for event in final)
    return {
        "source_artifact": display_path(source),
        "videos_count": len(videos),
        "segments_count": len(final),
        "classification_counts": {name: classes.get(name, 0) for name in CLASSES},
        "videos": videos,
    }


def env_values(path: Path) -> dict[str, str]:
    return dict(
        line.split("=", 1)
        for line in path.read_text(encoding="utf-8").splitlines()
        if "=" in line and not line.lstrip().startswith("#")
    )


def env_checkpoints(path: Path) -> dict[str, str | None]:
    values = env_values(path)
    return {f"module_{key}": display_path(values.get(f"MODULE_{key}_CHECKPOINT")) for key in ("A", "B", "C")}


def pipeline_report(path: Path, env_path: Path = PIPELINE_ENV) -> dict[str, Any]:
    artifact = json.loads(require(path).read_text(encoding="utf-8"))
    return {
        "method": "Modular (Modules A, B and C, causal stream)",
        "run": path.parent.name,
        "checkpoints": env_checkpoints(env_path),
        "split": "seed-3407 test (configs/splits/egoper_video_split_seed3407.json)",
        "fields": {
            "start, end": "predicted step span in seconds of the trimmed video (Module B)",
            "step_id, step": "predicted step (Module A); EXTRA is an off-script step and has no instruction",
            "classification": "execution class (Module C); EXTRA steps are ADDITION",
            "is_error": IS_ERROR_FIELD,
            "decided_at": "video time of the Module A poll that triggered this step",
            "clip": "clip Module A and Module B saw: the cursor (previous step's end) to decided_at",
            "reasoning": "Module C's explanation, null for EXTRA steps",
        },
        **build(artifact, method="pipeline", source=path),
    }


def monolith_report(path: Path, env_path: Path = MONOLITH_ENV) -> dict[str, Any]:
    artifact = json.loads(require(path).read_text(encoding="utf-8"))
    return {
        "method": "Joint (one model, one answer per clip)",
        "run": path.parent.name,
        "checkpoints": {"model": display_path(env_values(env_path).get("CHECKPOINT"))},
        "split": "seed-3407 test (configs/splits/egoper_video_split_seed3407.json)",
        "fields": {
            "start, end": "clip start plus the model's start and end, clamped to the clip, in seconds of the trimmed video",
            "step_id, step": "the model's step_id; EXTRA is an off-script step and has no instruction",
            "classification": "the model's classification",
            "is_error": IS_ERROR_FIELD,
            "decided_at": "video time of the poll whose answer placed this step",
            "clip": "clip the model saw: the cursor (previous step's end) to decided_at",
            "reasoning": "the model's reasoning",
            "model_output": "the model's answer as emitted, with start and end in seconds from the clip start",
            "clamped_to_clip": "true when the model's span ran past the clip and was clamped",
        },
        **build(artifact, method="monolith", source=path),
    }


def _ground_truth_key(row: dict[str, Any]) -> tuple:
    return (row["video_id"], row["gt_start"], row["gt_end"], row["step_index"], row["gt_fine_class"])


def ground_truth_report(pipeline_path: Path, monolith_path: Path) -> dict[str, Any]:
    pipeline_rows = json.loads(require(pipeline_path).read_text(encoding="utf-8"))["ground_truth"]
    monolith_rows = json.loads(require(monolith_path).read_text(encoding="utf-8"))["ground_truth"]
    if sorted(map(_ground_truth_key, pipeline_rows)) != sorted(map(_ground_truth_key, monolith_rows)):
        raise ValueError("the pipeline and monolith artifacts hold different ground truth")
    targets = [event for event in events_from_rows(GroundTruthEvent, pipeline_rows) if event.eligible]
    instructions = {
        (event.task_id, event.step_index): event.instruction for event in targets if event.step_index >= 0
    }
    addition_steps = _annotated_addition_steps()
    videos: dict[str, dict[str, Any]] = {}
    for event in sorted(targets, key=lambda event: (event.video_id, event.gt_start, event.gt_end)):
        row: dict[str, Any] = {
            "start": float(event.gt_start),
            "end": float(event.gt_end),
            "step_id": f"STEP_{event.step_index}" if event.step_index >= 0 else "EXTRA",
            "step": event.instruction if event.step_index >= 0 else None,
            "classification": event.gt_fine_class,
            "is_error": event.gt_fine_class != "CORRECT",
            "reasoning": event.gt_reasoning,
        }
        if event.step_index < 0:
            step = addition_steps.get(
                (event.video_id, round(event.gt_start, 1), round(event.gt_end, 1))
            )
            row["annotated_step_id"] = None if step is None else f"STEP_{step}"
            row["annotated_step"] = None if step is None else instructions.get((event.task_id, step))
        videos.setdefault(event.video_id, {"task": event.task_id, "segments": []})["segments"].append(row)
    classes = Counter(event.gt_fine_class for event in targets)
    return {
        "method": "EgoPER annotation",
        "source": "data/EgoPER/annotation.json, as loaded into the test artifacts' ground_truth",
        "source_artifact": display_path(pipeline_path),
        "split": "seed-3407 test (configs/splits/egoper_video_split_seed3407.json)",
        "fields": {
            "start, end": "annotated segment in seconds of the trimmed video",
            "step_id, step": "annotated step; ADDITION segments are EXTRA, as the models predict them",
            "classification": "annotated execution class",
            "is_error": IS_ERROR_FIELD,
            "reasoning": "annotated description of what was done",
            "annotated_step_id, annotated_step": (
                "ADDITION only: the step EgoPER's annotation gives the segment, when it gives one. "
                "TAS F1 accepts either EXTRA or this step"
            ),
        },
        "videos_count": len(videos),
        "segments_count": len(targets),
        "classification_counts": {name: classes.get(name, 0) for name in CLASSES},
        "videos": dict(sorted(videos.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pipeline-artifact", type=Path, default=PIPELINE)
    parser.add_argument("--pipeline-env", type=Path, default=PIPELINE_ENV)
    parser.add_argument("--monolith-artifact", type=Path, default=MONOLITH)
    parser.add_argument("--monolith-env", type=Path, default=MONOLITH_ENV)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, report in (
        ("pipeline", pipeline_report(args.pipeline_artifact, args.pipeline_env)),
        ("monolith", monolith_report(args.monolith_artifact, args.monolith_env)),
    ):
        path = args.output_dir / f"{name}_test.json"
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"{path}: {report['videos_count']} videos, {report['segments_count']} segments")
    path = args.output_dir / "ground_truth_test.json"
    report = ground_truth_report(args.pipeline_artifact, args.monolith_artifact)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{path}: {report['videos_count']} videos, {report['segments_count']} segments")


if __name__ == "__main__":
    main()
