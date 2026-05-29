"""Node: verify_result — validates LLM output and decides if retry needed.

Checks:
1. Tool calls executed successfully
2. Output contains expected artifacts (code blocks, etc.)
3. No obvious errors in generated code

Returns:
- verification_passed: bool
- retry_reason: str (if failed)
- retry_count: int
"""

from __future__ import annotations

import logging
import re
from server.agent.state import AgentState

logger = logging.getLogger("server.agent.verify_result")

MAX_RETRIES = 2


def verify_result(state: AgentState) -> dict:
    """Verify the generated result and decide if retry is needed."""
    draft = state.get("draft", "")
    intent = state.get("intent", "")
    tool_results = state.get("tool_results", [])
    retry_count = state.get("retry_count", 0)
    pending_tool_calls = state.get("pending_tool_calls", [])

    # If there are pending tool calls, don't verify yet - let tools execute
    if pending_tool_calls:
        logger.info("verify_result: pending tool calls, skipping verification")
        return {"verification_passed": True}

    # Check retry limit
    if retry_count >= MAX_RETRIES:
        logger.warning("verify_result: max retries reached (%d)", retry_count)
        return {"verification_passed": True, "retry_reason": "max_retries_reached"}

    issues: list[str] = []

    # Check 1: Tool execution errors
    for result in tool_results:
        if isinstance(result, dict) and result.get("error"):
            issues.append(f"Tool error: {result.get('error')}")

    # Check 2: Empty response for code-generating intents
    if intent in ("code_gen", "unit_test", "refine") and not draft.strip():
        issues.append("Empty response for code generation request")

    # Check 3: Code block expected but missing
    if intent in ("code_gen", "unit_test", "refine", "debug"):
        has_code_block = bool(re.search(r"```\w*\n", draft))
        if not has_code_block and len(draft) > 100:
            # Long response without code block might be explanation only
            pass  # Allow explanations
        elif not has_code_block and "error" not in draft.lower():
            issues.append("Expected code block in response")

    # Check 4: Syntax errors in code blocks (basic check)
    code_blocks = re.findall(r"```(\w+)?\n(.*?)```", draft, re.DOTALL)
    for lang, code in code_blocks:
        if lang in ("python", "py"):
            if _has_python_syntax_error(code):
                issues.append("Python syntax error detected in code block")
        elif lang in ("javascript", "js", "typescript", "ts"):
            if _has_js_syntax_error(code):
                issues.append("JavaScript/TypeScript syntax error detected")

    # Check 5: Incomplete response markers
    incomplete_markers = [
        "...",
        "// TODO",
        "# TODO",
        "/* TODO",
        "FIXME",
        "[continue]",
        "[...]",
    ]
    for marker in incomplete_markers:
        if marker in draft and draft.count(marker) > 2:
            issues.append(f"Response appears incomplete (multiple '{marker}')")
            break

    if issues:
        logger.warning("verify_result: issues found: %s", issues)
        return {
            "verification_passed": False,
            "retry_reason": "; ".join(issues[:3]),  # Limit to 3 reasons
            "retry_count": retry_count + 1,
        }

    logger.info("verify_result: passed")
    return {"verification_passed": True}


def _has_python_syntax_error(code: str) -> bool:
    """Basic Python syntax check."""
    try:
        compile(code, "<string>", "exec")
        return False
    except SyntaxError:
        return True
    except Exception:
        return False  # Other errors (e.g., encoding) - don't flag


def _has_js_syntax_error(code: str) -> bool:
    """Basic JS/TS syntax check (heuristic only)."""
    # Check for obviously broken syntax
    issues = [
        (r"function\s*\([^)]*$", "incomplete function"),
        (r"if\s*\([^)]*$", "incomplete if"),
        (r"\{\s*$", "unclosed brace at end"),
        (r"=>\s*$", "incomplete arrow function"),
    ]
    for pattern, _ in issues:
        if re.search(pattern, code):
            return True

    # Check bracket balance (simple)
    opens = code.count("{") + code.count("[") + code.count("(")
    closes = code.count("}") + code.count("]") + code.count(")")
    if abs(opens - closes) > 2:
        return True

    return False
