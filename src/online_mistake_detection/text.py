from __future__ import annotations

import re


_PREFIX_PATTERN = re.compile(
    r"^(?P<prefix>[A-Za-z][\w]*(?:[\s/]+[A-Za-z][\w]*)*)\s*-\s*(?P<rest>.+)$"
)


def normalize_instruction_text(text: str) -> str:
    """Strip a duplicated verb prefix such as ``Clean-Clean the knife``."""

    cleaned = " ".join(str(text).replace("—", "-").replace("–", "-").split())
    if not cleaned:
        return ""
    match = _PREFIX_PATTERN.match(cleaned)
    if match is None:
        return cleaned
    prefix = match.group("prefix")
    rest = match.group("rest")
    rest_tokens = rest.split()
    prefix_tokens = prefix.split()
    if (
        rest_tokens
        and len(rest_tokens) >= len(prefix_tokens)
        and [token.casefold() for token in rest_tokens[: len(prefix_tokens)]]
        == [token.casefold() for token in prefix_tokens]
    ):
        return rest
    return cleaned


def normalize_instructions(instructions: list[str] | tuple[str, ...]) -> list[str]:
    return [normalize_instruction_text(item) for item in instructions]
