from __future__ import annotations

import json
import re
from typing import Any


_JSON_RE = re.compile(r"\{[\s\S]*\}")


def extract_json_object(text: str) -> dict[str, Any]:
    """Best-effort JSON object extraction from an LLM response."""

    if not text:
        raise ValueError("Empty response")

    text = text.strip()

    # Try direct parse first.
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Fallback: first {...} block.
    m = _JSON_RE.search(text)
    if not m:
        raise ValueError("No JSON object found")

    obj = json.loads(m.group(0))
    if not isinstance(obj, dict):
        raise ValueError("Parsed JSON is not an object")

    return obj
