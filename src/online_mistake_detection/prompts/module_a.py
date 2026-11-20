from __future__ import annotations

import json
from collections.abc import Sequence

from ..text import normalize_instruction_text


MODULE_A_SYSTEM_PROMPT = (
    "Review a person following procedural instructions. Answer WAIT or COMPLETE. "
    "Say COMPLETE when the attempt at the current step has ended, even if it "
    "included mistakes; otherwise say WAIT."
)
MODULE_A_STEP_ID_SYSTEM_PROMPT = (
    "Detect which procedural step, if any, completed in the clip. Reply with "
    "exactly one choice letter. A means no step completed. B means an extra "
    "off-instruction step completed. C and later letters identify a listed "
    "instruction that just finished. If the person did a listed action "
    "incorrectly, still pick that listed letter when it is the closest match "
    "(for example, measuring 13 ounces still maps to Measure 12 ounces of cold "
    "water using measuring cup). Use B only when the completed action is not any "
    "listed instruction. Do not assume recipe order."
)
MODULE_A_COMPLETION_SET_SYSTEM_PROMPT = (
    "Detect every recipe step that reached its physical end in this causal clip. "
    "Return JSON only as {\"completed_steps\":[\"STEP_k\", ...]}. Use an empty "
    "list if nothing finished."
)


def step_id_for_instruction_index(index: int) -> str:
    return "EXTRA" if index < 0 else f"STEP_{index}"


def build_module_a_user_prompt(
    *,
    completed_steps: Sequence[str],
    current_step: str,
    pending_steps: Sequence[str],
) -> str:
    return (
        "Completed Steps: "
        f"{json.dumps(list(completed_steps), ensure_ascii=False)}\n"
        f"Current Step: {normalize_instruction_text(current_step)}\n"
        "Pending Steps: "
        f"{json.dumps([normalize_instruction_text(step) for step in pending_steps], ensure_ascii=False)}\n"
        "Status?"
    )


def build_module_a_step_id_user_prompt(
    *,
    task_name: str,
    task_description: str,
    instructions: Sequence[str],
    completed_steps: Sequence[str],
    dataset_name: str = "",
) -> str:
    numbered = "\n".join(
        f"{chr(ord('C') + index)}: STEP_{index} — {normalize_instruction_text(instruction)}"
        for index, instruction in enumerate(instructions)
    )
    completed = (
        "\n".join(f"- {normalize_instruction_text(step)}" for step in completed_steps)
        if completed_steps
        else "- None"
    )
    lines = [
        f"Task: {task_name}",
        f"Task description: {task_description}",
    ]
    if dataset_name:
        lines.append(f"Dataset: {dataset_name}")
    lines.extend(
        [
            "Completed instructions before this clip:",
            completed,
            "Choices:",
            "A: NONE (no newly completed procedural event)",
            "B: EXTRA (an off-instruction procedural event completed)",
            numbered,
        ]
    )
    return "\n".join(lines)


def build_module_a_prompt(
    *,
    task_name: str,
    task_description: str,
    instructions: Sequence[str],
    completed_steps: Sequence[str],
) -> str:
    return build_module_a_step_id_user_prompt(
        task_name=task_name,
        task_description=task_description,
        instructions=instructions,
        completed_steps=completed_steps,
    )


def build_module_a_completion_set_prompt(
    *,
    task_name: str,
    task_description: str,
    instructions: Sequence[str],
    completed_steps: Sequence[str],
) -> str:
    numbered = "\n".join(
        f"STEP_{index}: {normalize_instruction_text(instruction)}"
        for index, instruction in enumerate(instructions)
    )
    completed = json.dumps(
        [normalize_instruction_text(step) for step in completed_steps],
        ensure_ascii=False,
    )
    return (
        f"Task: {task_name}\n"
        f"Task description: {task_description}\n"
        f"Steps already completed: {completed}\n"
        f"{numbered}"
    )


def format_completion_set_target(step_ids: Sequence[str]) -> str:
    unique = []
    for step_id in step_ids:
        if step_id not in unique:
            unique.append(step_id)
    return json.dumps({"completed_steps": unique}, ensure_ascii=False, separators=(",", ":"))


def parse_completion_set(text: str) -> list[str]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return []
        try:
            payload = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return []
    raw = payload.get("completed_steps") if isinstance(payload, dict) else None
    if not isinstance(raw, list):
        return []
    parsed: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            continue
        normalized = item.strip().upper().replace(" ", "")
        if normalized == "NONE":
            continue
        if normalized.startswith("STEP_"):
            parsed.append(normalized)
    return parsed
