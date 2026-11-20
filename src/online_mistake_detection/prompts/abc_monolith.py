from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..text import normalize_instruction_text
from .contracts import NOT_COMPLETED_TARGET


MONOLITH_CLASSES = ("CORRECT", "SLIP", "MODIFICATION", "CORRECTION", "ADDITION")
MONOLITH_SYSTEM_PROMPT = (
    "Review the accumulated causal video once. Decide whether a procedural action has "
    "finished. If none has finished, return exactly "
    f"{NOT_COMPLETED_TARGET}. Otherwise return exactly one JSON object with keys "
    '"step_id","start","end","classification","reasoning". step_id must be STEP_k '
    "for a listed instruction or EXTRA for an off-script action. Times are seconds "
    "relative to the beginning of this clip and must satisfy 0 <= start < end <= "
    "the stated clip duration. ADDITION means an off-script action and therefore "
    "requires step_id EXTRA; EXTRA must always be classified ADDITION. CORRECTION "
    "repairs a preceding error and is not itself a mistake. Return JSON only. Never "
    "return WAIT, COMPLETE, or a COMPLETED status."
)


NATIVE_PROMPT_VERSION = "abc_monolith_v1"


def normalize_monolith_dataset(dataset: str) -> str:
    normalized = str(dataset or "egoper").strip().lower().replace("-", "_")
    if normalized == "egoper":
        return "egoper"
    raise ValueError(f"Unsupported monolith dataset: {dataset}")


def resolve_monolith_contract(
    dataset: str,
    prompt_contract: str | None = None,
) -> str:
    data_name = normalize_monolith_dataset(dataset)
    if prompt_contract is not None and str(prompt_contract).strip():
        normalize_monolith_dataset(str(prompt_contract))
    return data_name


def prompt_version_for_contract(
    dataset: str,
    prompt_contract: str | None = None,
) -> str:
    resolve_monolith_contract(dataset, prompt_contract)
    return NATIVE_PROMPT_VERSION


def monolith_dataset_label(dataset: str) -> str:
    normalize_monolith_dataset(dataset)
    return "EgoPER"


def execution_classes_for_dataset(
    dataset: str,
    *,
    prompt_contract: str | None = None,
) -> tuple[str, ...]:
    resolve_monolith_contract(dataset, prompt_contract)
    return MONOLITH_CLASSES


def classification_matches_step(
    step_id: str,
    classification: str,
    dataset: str = "egoper",
    *,
    prompt_contract: str | None = None,
) -> bool:
    """EXTRA is paired only with ADDITION."""

    resolve_monolith_contract(dataset, prompt_contract)
    return classification in MONOLITH_CLASSES and (
        (step_id == "EXTRA") == (classification == "ADDITION")
    )


def system_prompt_for_dataset(
    dataset: str,
    *,
    prompt_contract: str | None = None,
) -> str:
    resolve_monolith_contract(dataset, prompt_contract)
    return MONOLITH_SYSTEM_PROMPT


@dataclass(frozen=True)
class MonolithPrediction:
    valid: bool
    completed: bool
    step_id: str | None = None
    start: float | None = None
    end: float | None = None
    classification: str | None = None
    reasoning: str | None = None
    error: str | None = None
    payload: Mapping[str, Any] | None = None


def format_monolith_target(
    *,
    step_id: str,
    start: float,
    end: float,
    classification: str,
    reasoning: str,
    dataset: str = "egoper",
    prompt_contract: str | None = None,
) -> str:
    """Format and validate a completed-action target."""
    prediction = _validate_payload(
        {
            "step_id": step_id,
            "start": round(float(start), 2),
            "end": round(float(end), 2),
            "classification": classification,
            "reasoning": reasoning,
        },
        clip_duration=None,
        dataset=dataset,
        prompt_contract=prompt_contract,
    )
    if not prediction.valid:
        raise ValueError(prediction.error)
    return json.dumps(prediction.payload, ensure_ascii=False, separators=(",", ":"))


def parse_monolith_prediction(
    text: str,
    *,
    clip_duration: float | None = None,
    dataset: str = "egoper",
    prompt_contract: str | None = None,
) -> MonolithPrediction:
    """Strictly parse the complete monolith response; prose and extra keys are invalid."""
    try:
        payload = json.loads(str(text).strip())
    except (TypeError, json.JSONDecodeError) as exc:
        return MonolithPrediction(False, False, error=f"invalid JSON: {exc}")
    if not isinstance(payload, dict):
        return MonolithPrediction(False, False, error="response must be a JSON object")
    return _validate_payload(
        payload,
        clip_duration=clip_duration,
        dataset=dataset,
        prompt_contract=prompt_contract,
    )


def _validate_payload(
    payload: Mapping[str, Any],
    *,
    clip_duration: float | None,
    dataset: str = "egoper",
    prompt_contract: str | None = None,
) -> MonolithPrediction:
    keys = set(payload)
    if keys == {"status"}:
        if payload.get("status") == "NOT_COMPLETED":
            return MonolithPrediction(True, False, payload=dict(payload))
        return MonolithPrediction(
            False,
            False,
            error="the only status payload is NOT_COMPLETED",
            payload=dict(payload),
        )
    required = {"step_id", "start", "end", "classification", "reasoning"}
    if keys != required:
        return MonolithPrediction(
            False,
            False,
            error=f"completed payload must contain exactly {sorted(required)}",
            payload=dict(payload),
        )
    step_id = payload.get("step_id")
    classification = payload.get("classification")
    reasoning = payload.get("reasoning")
    if not isinstance(step_id, str) or not (
        step_id == "EXTRA"
        or (step_id.startswith("STEP_") and step_id[5:].isdigit())
    ):
        return MonolithPrediction(False, False, error="invalid step_id", payload=dict(payload))
    allowed = execution_classes_for_dataset(
        dataset,
        prompt_contract=prompt_contract,
    )
    if not isinstance(classification, str) or classification not in allowed:
        return MonolithPrediction(
            False, False, error="invalid classification", payload=dict(payload)
        )
    if not classification_matches_step(
        step_id,
        classification,
        dataset,
        prompt_contract=prompt_contract,
    ):
        return MonolithPrediction(
            False,
            False,
            error="step_id does not match the classification",
            payload=dict(payload),
        )
    if not isinstance(reasoning, str) or not reasoning.strip():
        return MonolithPrediction(
            False, False, error="reasoning must be non-empty", payload=dict(payload)
        )
    start = payload.get("start")
    end = payload.get("end")
    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, (int, float))
        or not isinstance(end, (int, float))
        or not math.isfinite(float(start))
        or not math.isfinite(float(end))
        or float(start) < 0
        or float(start) >= float(end)
    ):
        return MonolithPrediction(False, False, error="invalid local span", payload=dict(payload))
    normalized_end = float(end)
    if clip_duration is not None:
        if (
            not math.isfinite(float(clip_duration))
            or float(clip_duration) <= 0
            or float(end) > round(float(clip_duration), 2) + 1e-9
            or float(start) >= float(clip_duration)
        ):
            return MonolithPrediction(
                False,
                False,
                error="local span exceeds clip duration",
                payload=dict(payload),
            )
        normalized_end = min(float(end), float(clip_duration))
    return MonolithPrediction(
        True,
        True,
        step_id=step_id,
        start=float(start),
        end=normalized_end,
        classification=str(classification),
        reasoning=reasoning.strip(),
        payload=dict(payload),
    )


def monolith_global_span(
    prediction: MonolithPrediction,
    *,
    window_start: float,
) -> tuple[float, float] | None:
    if not prediction.valid or not prediction.completed:
        return None
    assert prediction.start is not None and prediction.end is not None
    return window_start + prediction.start, window_start + prediction.end


def build_monolith_prompt(
    *,
    task_name: str,
    task_description: str,
    instructions: Sequence[str],
    clip_duration: float,
    history: Sequence[Mapping[str, object]] | None,
    max_history: int = 12,
    history_source: str = "gold",
    dataset: str = "egoper",
    prompt_contract: str | None = None,
    display_dataset: str | None = None,
) -> str:
    if not math.isfinite(clip_duration) or clip_duration <= 0:
        raise ValueError("clip_duration must be a finite positive number")
    procedure = "\n".join(
        f"  STEP_{index}: {normalize_instruction_text(instruction)}"
        for index, instruction in enumerate(instructions)
    )
    if history is None:
        history_text = "  (unavailable due to training history dropout)"
    else:
        selected = list(history)[-max(0, max_history) :]
        history_text = "\n".join(
            "  "
            f"{index}. {entry.get('step_id', 'UNKNOWN')} | "
            f"{normalize_instruction_text(str(entry.get('action') or entry.get('instruction') or 'off-script action'))} | "
            f"{str(entry.get('classification') or 'UNKNOWN').upper()} | "
            f"{' '.join(str(entry.get('reasoning') or entry.get('evidence') or 'no evidence').split())}"
            for index, entry in enumerate(selected, start=1)
        ) or "  (none)"
    if history_source not in {"gold", "predicted", "none"}:
        raise ValueError("history_source must be 'gold', 'predicted', or 'none'")
    if history_source == "predicted":
        history_heading = "Predicted prior eligible actions"
    elif history_source == "none":
        history_heading = "No prior actions supplied"
    else:
        history_heading = "Gold prior eligible actions"
    resolve_monolith_contract(dataset, prompt_contract)
    class_definitions = (
        "Class definitions: CORRECT follows the listed instruction; SLIP performs it "
        "but fails its goal; MODIFICATION performs it differently from the procedure; "
        "CORRECTION repairs an earlier deviation; ADDITION is a completed off-script "
        "action. Determine only what completed in this causal clip."
    )
    dataset_label = "EgoPER"
    if display_dataset is not None and str(display_dataset).strip():
        dataset_label = str(display_dataset).strip()
    return (
        f"Dataset: {dataset_label}\nTask: {task_name}\nTask description: {task_description}\n"
        f"Clip-local prediction region: [0.00, {clip_duration:.2f}] seconds.\n"
        f"Procedure:\n{procedure}\n"
        f"{history_heading} (up to {max_history}, oldest to newest):\n"
        f"{history_text}\n"
        f"{class_definitions}"
    )
