from __future__ import annotations

import math

from ..text import normalize_instruction_text
from .contracts import NOT_COMPLETED_TARGET, format_module_b_span_target


MODULE_B_SYSTEM_PROMPT = (
    "Locate the requested instruction attempt in the clip. If it is completed, "
    "return JSON only as {\"start\":<seconds>,\"end\":<seconds>} using seconds "
    "relative to the start of the clip. If the attempt is not completed, return "
    f"JSON only as {NOT_COMPLETED_TARGET}."
)


def build_module_b_prompt(instruction: str, clip_duration: float) -> str:
    if not math.isfinite(clip_duration) or clip_duration <= 0:
        raise ValueError("clip_duration must be a finite positive number")
    bound = f"{clip_duration:.2f}"
    return (
        f"Instruction: {normalize_instruction_text(instruction)}\n"
        f"The clip-local physical prediction region is [0.00, {bound}] seconds. "
        f"A valid span must satisfy 0.00 <= start < end <= {bound}."
    )


def format_module_b_target(start: float, end: float) -> str:
    return format_module_b_span_target(start, end)
