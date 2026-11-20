from __future__ import annotations

from ..schemas import Segment

EXTRA_STEP_INSTRUCTION = "undefined / extra step"

MODULE_A_STEP_ID_NONE_LETTER = "A"
MODULE_A_STEP_ID_EXTRA_LETTER = "B"
MODULE_A_STEP_ID_FIRST_INSTRUCTION_LETTER = "C"


def module_a_step_id_letter_for_instruction_index(instruction_index: int) -> str:
    if instruction_index < 0:
        return MODULE_A_STEP_ID_EXTRA_LETTER
    return chr(ord(MODULE_A_STEP_ID_FIRST_INSTRUCTION_LETTER) + instruction_index)


def module_a_instruction_for_step_id_letter(
    letter: str,
    instructions: list[str] | tuple[str, ...],
) -> str | None:
    normalized = letter.strip().upper()[:1]
    if not normalized or not ("A" <= normalized <= "Z"):
        return None
    if normalized == MODULE_A_STEP_ID_NONE_LETTER:
        return None
    if normalized == MODULE_A_STEP_ID_EXTRA_LETTER:
        return EXTRA_STEP_INSTRUCTION
    index = ord(normalized) - ord(MODULE_A_STEP_ID_FIRST_INSTRUCTION_LETTER)
    if 0 <= index < len(instructions):
        return instructions[index]
    return None


def is_module_a_completion_label(label: str, *, label_mode: str) -> bool:
    if label_mode == "legacy":
        return label.strip().upper() == "COMPLETE"
    if label_mode == "completion_set":
        from ..prompts.module_a import parse_completion_set

        return bool(parse_completion_set(label))
    return label.strip().upper()[:1] not in ("", MODULE_A_STEP_ID_NONE_LETTER)


def reasoning_for_segment(segment: Segment) -> str:
    if segment.is_mistake:
        if segment.caption:
            return segment.caption
        if segment.error_label:
            return f"The execution deviates from the instruction and is labeled as {segment.error_label}."
        return "The segment is labeled as an undefined or extra step."
    return "The observed action follows the instructed step without a visible procedural deviation."
