from __future__ import annotations

import json


NOT_COMPLETED_STATUS = "NOT_COMPLETED"
NOT_COMPLETED_TARGET = json.dumps(
    {"status": NOT_COMPLETED_STATUS},
    ensure_ascii=False,
    separators=(",", ":"),
)


def format_module_b_span_target(start: float, end: float) -> str:
    return json.dumps(
        {"start": round(float(start), 2), "end": round(float(end), 2)},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def is_not_completed_text(text: str) -> bool:
    cleaned = str(text).strip()
    if not cleaned:
        return False
    if cleaned.casefold() in {"not completed", NOT_COMPLETED_STATUS.casefold()}:
        return True
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            return "not completed" in cleaned.casefold()
        try:
            payload = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            return "not completed" in cleaned.casefold()
    if isinstance(payload, dict):
        status = str(payload.get("status") or "").strip().casefold()
        if status == NOT_COMPLETED_STATUS.casefold():
            return True
        classification = str(payload.get("classification") or "").strip().casefold()
        if classification == NOT_COMPLETED_STATUS.casefold():
            return True
    return "not completed" in cleaned.casefold()
