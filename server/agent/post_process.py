"""Node: post_process — Section 8.

Intent-specific validation rules with lightweight checks.
"""

from __future__ import annotations

import logging
import re

from server.agent.state import AgentState

logger = logging.getLogger("server.agent.post_process")


def post_process(state: AgentState) -> dict:
    """Apply intent-specific validation to the draft.

    Returns validation_warnings for observability.
    """
    intent = state.get("intent", "code_gen")
    draft = state.get("draft", "")
    warnings: list[str] = []

    # Strip leaked system prompt fragments
    draft = _strip_leaked_prompt(draft)

    # Intent-specific validation
    if intent == "unit_test":
        warnings.extend(_validate_unit_test(draft))
    elif intent == "code_gen":
        warnings.extend(_validate_code_gen(draft))
    elif intent == "code_review":
        warnings.extend(_validate_code_review(draft))

    if warnings:
        logger.warning("post_process warnings: %s", warnings)

    return {"draft": draft, "validation_warnings": warnings}


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


def _validate_unit_test(draft: str) -> list[str]:
    """Validate unit test output."""
    warnings = []

    if not _has_test_function(draft):
        warnings.append("Missing test function declaration")

    # Extract code blocks
    code = _extract_code_blocks(draft)
    if not code:
        return warnings

    # Java-specific checks
    if "```java" in draft.lower():
        if "@SpringBootTest" in code and "@Mock" not in code and "@MockBean" not in code:
            warnings.append("@SpringBootTest without @Mock/@MockBean - may need mocking")
        if "@Autowired" in code and "@Mock" not in code:
            warnings.append("@Autowired in test without mocks - consider using @Mock")
        if "assert" not in code.lower() and "verify(" not in code:
            warnings.append("Test has no assertions or verifications")

    # Python-specific checks
    if "```python" in draft.lower() or "def test_" in code:
        if "assert" not in code and "pytest.raises" not in code:
            warnings.append("Python test missing assertions")

    return warnings


def _validate_code_gen(draft: str) -> list[str]:
    """Validate code generation output."""
    warnings = []

    if not _has_code_blocks(draft):
        warnings.append("Missing code blocks in response")

    code = _extract_code_blocks(draft)
    if code:
        # Check for common issues
        if "TODO" in code or "FIXME" in code:
            warnings.append("Code contains TODO/FIXME placeholders")
        if "..." in code and "```" not in code:
            warnings.append("Code may be incomplete (contains ...)")

    return warnings


def _validate_code_review(draft: str) -> list[str]:
    """Validate code review output."""
    warnings = []

    # Review should have some structure
    if len(draft) < 50:
        warnings.append("Review output seems too short")

    return warnings


def _extract_code_blocks(text: str) -> str:
    """Extract all code from markdown code blocks."""
    blocks = re.findall(r"```\w*\n(.*?)```", text, re.DOTALL)
    return "\n".join(blocks)
