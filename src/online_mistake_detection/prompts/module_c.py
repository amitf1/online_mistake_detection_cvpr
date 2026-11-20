from __future__ import annotations

from collections.abc import Mapping, Sequence

from ..text import normalize_instruction_text
from .contracts import NOT_COMPLETED_STATUS, NOT_COMPLETED_TARGET


MODULE_C_SYSTEM_PROMPT = (
    "Classify the visible execution of the requested instruction. Choose exactly "
    "one outcome. CORRECTION is a repair of a preceding error and is not itself "
    "a mistake. If the clip does not contain a completed attempt, return "
    f"{NOT_COMPLETED_STATUS}. Return JSON only as "
    '{"classification":"<class>","reasoning":"<visual evidence>"} or '
    f"{NOT_COMPLETED_TARGET}."
)

EGOPER_MODULE_C_OUTCOMES = (
    "CORRECT",
    "CORRECTION",
    "MODIFICATION",
    "SLIP",
)


def _compact_history_text(value: object, *, fallback: str) -> str:
    text = " ".join(str(value or "").split())
    return text or fallback


def append_module_c_history(
    prompt: str,
    history: Sequence[Mapping[str, object]] | None,
    *,
    max_events: int = 12,
) -> str:
    """Append compact prior-action context."""
    if max_events <= 0:
        return prompt
    heading = (
        f"\nRecent prior action history (up to the {max_events} most recent eligible "
        "completed actions; shown oldest to newest; older actions may be omitted; "
        "use only when relevant):\n"
    )
    if history is None:
        return (
            f"{prompt}{heading}"
            "  (unavailable; classify from the current instruction and video)"
        )
    selected = list(history)[-max_events:]
    if not selected:
        return f"{prompt}{heading}  (none; no earlier action has completed)"
    lines = []
    for position, entry in enumerate(selected, start=1):
        action = _compact_history_text(
            entry.get("action") or entry.get("instruction"),
            fallback="off-script action",
        )
        classification = _compact_history_text(
            entry.get("classification"),
            fallback="UNKNOWN",
        ).upper()
        evidence = _compact_history_text(
            entry.get("evidence") or entry.get("reasoning"),
            fallback="no evidence recorded",
        )
        lines.append(
            f"  {position}. Action: {action} | Classification: "
            f"{classification} | Evidence: {evidence}"
        )
    return f"{prompt}{heading}" + "\n".join(lines)


def module_c_classes_for_dataset(dataset_name: str, allowed_classes: Sequence[str]) -> list[str]:
    normalized = dataset_name.strip().lower().replace("-", "_")
    allowed = [item for item in allowed_classes if item != "ADDITION"]
    allowed_set = set(allowed)
    if normalized == "egoper":
        ordered = [item for item in EGOPER_MODULE_C_OUTCOMES if item in allowed_set]
        return ordered or [item for item in allowed if item in EGOPER_MODULE_C_OUTCOMES]
    return list(allowed)


def build_module_c_prompt(
    instruction: str,
    allowed_classes: Sequence[str],
    *,
    dataset_name: str,
) -> str:
    classes = module_c_classes_for_dataset(dataset_name, allowed_classes)
    if not classes:
        raise ValueError("allowed_classes cannot be empty")
    outcomes = [*classes, NOT_COMPLETED_STATUS]
    dataset_guidance = ""
    correction_definition = (
        "CORRECTION means a visible action that repairs a preceding error. "
    )
    if dataset_name.strip().lower().replace("-", "_") == "egoper":
        definitions = []
        if "SLIP" in classes:
            definitions.append(
                "SLIP means executing the requested step in a way that fails to achieve "
                "its goal, such as adding water to a different bowl from the one "
                "containing oats or dropping a tortilla on the floor."
            )
        if "MODIFICATION" in classes:
            definitions.append(
                "MODIFICATION means performing the requested step differently from the "
                "recipe, such as scooping nut butter with a spoon or pouring water "
                "without the specified circular motion. This does not necessarily "
                "change the step outcome."
            )
        dataset_guidance = " ".join(definitions)
        if dataset_guidance:
            dataset_guidance += " "
        correction_definition = (
            "CORRECTION means performing an action that mitigates the effect of a "
            "preceding SLIP, such as transferring water from the second bowl to the "
            "one containing oats or discarding a tortilla dropped on the floor and "
            "picking up a new one. "
        )
    return (
        f"Dataset: {dataset_name}\n"
        f"Instruction: {normalize_instruction_text(instruction)}\n"
        "Classify only the execution visible in this bounded clip. "
        f"Allowed outcomes: {', '.join(outcomes)}. "
        f"{dataset_guidance}"
        f"{correction_definition}"
        "CORRECT means the visible execution follows the instruction. "
        f"{NOT_COMPLETED_STATUS} means this clip is not a completed attempt. "
        "Describe the observable evidence without repeating the class name as the explanation."
    )


def build_reasoning(
    *,
    observed_description: str,
    instruction: str,
    fine_class: str,
) -> str:
    observed = observed_description.strip()
    if observed and observed.upper() != "BG":
        return observed
    if fine_class == "CORRECT":
        return "The visible action follows the instructed step without a procedural deviation."
    if fine_class == "CORRECTION":
        return "The visible action repairs an earlier deviation and returns to the intended procedure."
    if fine_class == NOT_COMPLETED_STATUS:
        return "The clip does not contain a completed attempt at the requested instruction."
    return f"The visible execution deviates from the intended action: {normalize_instruction_text(instruction)}."
