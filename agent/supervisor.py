"""
Supervisor — intent classifier for the agent.

Uses regex/keyword matching (no LLM overhead) to classify user intent
and route to the appropriate subgraph.

Currently supports:
  - unit_test (default) — generate JUnit5+Mockito tests

Future (placeholders):
  - code_review — analyze code quality
  - refactor — suggest refactoring
  - doc_gen — generate documentation
"""

from __future__ import annotations

import re

import structlog

logger = structlog.get_logger()


# ═══════════════════════════════════════════════════════════════════════
# Intent patterns (regex, case-insensitive)
# ═══════════════════════════════════════════════════════════════════════

INTENT_PATTERNS: dict[str, list[str]] = {
    "unit_test": [
        r"test",
        r"junit",
        r"mock",
        r"unit\s*test",
        r"generate.*test",
        r"gen.*test",
        r"write.*test",
        r"create.*test",
    ],
    "code_review": [
        r"review",
        r"check.*code",
        r"analyze.*quality",
        r"code.*review",
    ],
    "refactor": [
        r"refactor",
        r"clean.*up",
        r"restructure",
        r"improve.*code",
    ],
    "doc_gen": [
        r"document",
        r"javadoc",
        r"readme",
        r"generate.*doc",
    ],
}


def classify_intent(user_input: str) -> str:
    """Classify user intent from input text.

    Uses regex pattern matching — no LLM call needed.
    Default: "unit_test" (backward compatible).

    Args:
        user_input: User's request text.

    Returns:
        Intent string: "unit_test", "code_review", "refactor", "doc_gen", or "unknown".
    """
    if not user_input:
        return "unit_test"  # default

    text = user_input.lower().strip()

    # Score each intent by number of matching patterns
    scores: dict[str, int] = {}
    for intent, patterns in INTENT_PATTERNS.items():
        score = 0
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                score += 1
        if score > 0:
            scores[intent] = score

    if not scores:
        # Default to unit_test for backward compatibility
        return "unit_test"

    # Return intent with highest score
    best = max(scores, key=scores.get)
    logger.debug(
        "supervisor: classified intent",
        intent=best,
        scores=scores,
        input_preview=text[:80],
    )
    return best


def supervisor_node(state: dict) -> dict:
    """Supervisor node — classifies intent and prepares routing.

    Args:
        state: AgentState dict.

    Returns:
        State updates: intent.
    """
    user_input = state.get("user_input", "")
    intent = classify_intent(user_input)

    logger.info("supervisor_node: intent classified", intent=intent)

    return {"intent": intent}
