"""Node: post_process — Section 8.

Intent-specific validation rules. No automatic retry in V1.
"""

from __future__ import annotations

import logging
import re

from server.agent.state import AgentState

logger = logging.getLogger("server.agent.post_process")


def post_process(state: AgentState) -> dict:
    """Apply intent-specific validation to the draft.

    Section 8:
    - unit_test: verify output contains a test function declaration
    - code_gen: verify syntactic plausibility (opening and closing blocks)
    - All: strip accidentally leaked system prompt text
    """
    intent = state.get("intent", "code_gen")
    draft = state.get("draft", "")

    # Strip leaked system prompt fragments
    draft = _strip_leaked_prompt(draft)

    # Intent-specific validation
    if intent == "unit_test":
        if not _has_test_function(draft):
            logger.warning("unit_test output missing test function declaration")

    elif intent == "code_gen":
        if not _has_code_blocks(draft):
            logger.warning("code_gen output missing code blocks")

    return {"draft": draft}


def _strip_leaked_prompt(text: str) -> str:
    """Remove accidentally leaked system prompt text."""
    markers = [
        "You are an expert coding assistant",
        "## Codebase Context",
        "## Execution Plan",
        "BASE_SYSTEM_PROMPT",
    ]
    for marker in markers:
        if marker in text:
            idx = text.find(marker)
            # If it appears near the start, strip everything before actual content
            if idx < 100:
                # Find the first meaningful content after the marker
                after = text[idx + len(marker):]
                newline_idx = after.find("\n\n")
                if newline_idx != -1:
                    text = after[newline_idx:].strip()
    return text


def _has_test_function(text: str) -> bool:
    """Check if text contains a test function declaration."""
    patterns = [
        r"@Test", r"def\s+test_", r"function\s+test",
        r"it\s*\(", r"describe\s*\(", r"func\s+Test",
        r"\[Test\]", r"\[Fact\]",
    ]
    return any(re.search(p, text) for p in patterns)


def _has_code_blocks(text: str) -> bool:
    """Check if text has code-like structure."""
    return "```" in text or "{" in text or "def " in text or "function " in text
