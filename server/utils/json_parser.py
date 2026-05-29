"""JSON parsing utilities."""

from __future__ import annotations

import json
import re


def parse_json_safe(s: str) -> dict:
    """Parse JSON safely, return empty dict on error.

    Handles common LLM output quirks:
    - Empty/None input
    - JSON decode errors
    """
    if not s:
        return {}
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return {}


def extract_json_object(text: str) -> dict | None:
    """Extract first JSON object from text.

    Handles:
    - <tool_call>...</tool_call> wrappers
    - ```json ... ``` code fences
    - {name, arguments} tool call envelopes

    Returns:
        Parsed dict or None if extraction fails
    """
    stripped = text.strip()

    # Strip <tool_call>...</tool_call> or <tools>...</tools> wrappers
    tag = re.match(r"^<(tool_call|tools)>\s*(.*?)\s*</\1>\s*$", stripped, re.DOTALL)
    if tag:
        stripped = tag.group(2).strip()

    # Strip markdown code fences
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, re.DOTALL)
    if fence:
        stripped = fence.group(1).strip()

    try:
        obj = json.loads(stripped)
    except Exception:
        return None

    if not isinstance(obj, dict):
        return None

    # Unwrap tool_call envelope {name, arguments}
    if set(obj.keys()) == {"name", "arguments"} and isinstance(obj.get("arguments"), dict):
        return obj["arguments"]

    return obj
