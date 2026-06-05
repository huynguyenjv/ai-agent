"""Tool Result Validation — Phase 17.4.

Validates tool results (returned by the client) before they enter the model
context, so the model can recover from tool errors instead of treating a failed
call as authoritative content. Conservative: only flags genuine errors/empty/
malformed output — a legitimate "no matches" or "tests failed" is still valid.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

logger = logging.getLogger("server.agent.tool_validator")


@dataclass
class ToolValidation:
    valid: bool
    category: str = "ok"          # ok | error | empty | malformed
    reason: str = ""
    retry_suggestion: str = ""


_RETRY_HINTS = {
    "vtrip_read_file": "Check the file_path is correct and relative to the repo root.",
    "vtrip_search_symbol": "Try vtrip_grep for a text search if the symbol name is unknown.",
    "vtrip_grep": "Refine the regex or widen path_glob.",
    "vtrip_apply_edits": "Re-read the file; the search text may no longer match.",
    "vtrip_apply_edits_atomic": "Re-read the file; a search text may no longer match (edit was rolled back).",
    "vtrip_run_command": "Verify the command is whitelisted and the working_dir exists.",
}


def validate_tool_result(tool_name: str, content: str) -> ToolValidation:
    """Validate a single tool result string."""
    name = tool_name or ""

    if content is None or not str(content).strip():
        return ToolValidation(False, "empty", "tool returned empty output",
                              _RETRY_HINTS.get(name, "Retry with corrected arguments."))

    # Try to parse as JSON (most vtrip_* tools return JSON)
    data = None
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        # Non-JSON text result — accept if non-empty.
        return ToolValidation(True, "ok")

    if isinstance(data, dict):
        if data.get("error"):
            return ToolValidation(
                False, "error", str(data["error"])[:200],
                _RETRY_HINTS.get(name, "Adjust arguments and retry."),
            )
        # Per-tool structural checks
        if name == "vtrip_read_file" and "content" not in data:
            return ToolValidation(False, "malformed", "read_file result missing 'content'",
                                  _RETRY_HINTS[name])
        if name in ("vtrip_apply_edits", "vtrip_apply_edits_atomic") and data.get("success") is False:
            return ToolValidation(False, "error", str(data.get("error", "edit failed"))[:200],
                                  _RETRY_HINTS.get(name, "Re-read and retry."))

    return ToolValidation(True, "ok")


def annotate_invalid(content: str, v: ToolValidation) -> str:
    """Prepend a short, model-readable note to an invalid tool result."""
    if v.valid:
        return content
    note = f"[tool_validation: {v.category} — {v.reason}. Suggestion: {v.retry_suggestion}]"
    return f"{note}\n{content}"
