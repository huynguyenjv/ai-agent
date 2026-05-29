"""Node: critic — review generated output for quality issues.

Evaluates code/responses for correctness, completeness, quality, and safety.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from server.agent.state import AgentState

logger = logging.getLogger("server.agent.critic")

CRITIC_PROMPT = """You are a code reviewer and quality critic for a coding assistant.

Review the generated output for:
1. **Correctness** - Does it solve the stated problem?
2. **Completeness** - Are all requirements from the request addressed?
3. **Quality** - Is it well-structured, readable, and maintainable?
4. **Safety** - Any security vulnerabilities (injection, XSS, etc.)?

User Request:
{user_request}

Generated Output:
{generated_output}

Intent: {intent}

Evaluate and output JSON:
{{
    "passed": true | false,
    "score": 0-10,
    "issues": [
        {{
            "severity": "critical" | "high" | "medium" | "low",
            "category": "correctness" | "completeness" | "quality" | "safety",
            "description": "what's wrong",
            "location": "where in the output (line or section)",
            "suggestion": "how to fix"
        }}
    ],
    "strengths": ["what's good about this output"],
    "retry_needed": true | false,
    "retry_feedback": "specific instructions for improvement if retry needed"
}}

Scoring guide:
- 9-10: Excellent, production ready
- 7-8: Good, minor improvements possible
- 5-6: Acceptable, some issues to address
- 3-4: Poor, significant issues
- 0-2: Unacceptable, major rework needed

Be constructive but thorough. Output JSON only."""

# Minimum score to pass
PASS_THRESHOLD = 6


async def critique_output(
    state: AgentState,
    vllm_client=None,
    model: str = "",
) -> dict[str, Any]:
    """Review generated output for quality issues.

    This is a LangGraph node. Reviews the draft output and determines
    if it meets quality standards or needs revision.

    Args:
        state: Current agent state
        vllm_client: vLLM client for LLM calls
        model: Model name

    Returns:
        Dict with critic_passed, critic_issues, critic_feedback, etc.
    """
    draft = state.get("draft", "")
    intent = state.get("intent", "code_gen")

    # Extract user request
    messages = state.get("messages", [])
    user_request = ""
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type in ("human", "user"):
            user_request = getattr(msg, "content", "")
            break
        elif isinstance(msg, dict) and msg.get("role") in ("human", "user"):
            user_request = msg.get("content", "")
            break

    # Skip critic for certain intents
    skip_intents = {"explain", "question", "refine"}
    if intent in skip_intents:
        logger.info("critic: skipping for intent=%s", intent)
        return {
            "critic_passed": True,
            "critic_score": 8,
            "critic_issues": [],
            "critic_feedback": "",
            "critic_skipped": True,
        }

    # Skip if no draft
    if not draft or len(draft.strip()) < 10:
        logger.info("critic: no substantial draft to review")
        return {
            "critic_passed": True,
            "critic_score": 0,
            "critic_issues": [],
            "critic_feedback": "No output to review",
            "critic_skipped": True,
        }

    # Quick heuristic checks before LLM
    quick_issues = _quick_quality_check(draft, intent)
    if quick_issues:
        logger.info("critic: quick check found %d issues", len(quick_issues))

    # Use LLM for thorough review
    if vllm_client is None:
        logger.warning("critic: no vllm_client, using quick check only")
        passed = len([i for i in quick_issues if i["severity"] in ("critical", "high")]) == 0
        return {
            "critic_passed": passed,
            "critic_score": 7 if passed else 4,
            "critic_issues": quick_issues,
            "critic_feedback": "" if passed else "Issues found in quick check",
        }

    prompt = CRITIC_PROMPT.format(
        user_request=user_request[:1500],
        generated_output=draft[:3000],
        intent=intent,
    )

    try:
        response = await vllm_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a code quality reviewer. Output JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=1000,
        )

        content = response.choices[0].message.content or ""
        review = _parse_critic_response(content)

        # Merge quick check issues
        all_issues = quick_issues + review.get("issues", [])
        review["issues"] = all_issues

        # Recalculate passed based on all issues
        critical_count = sum(1 for i in all_issues if i.get("severity") == "critical")
        high_count = sum(1 for i in all_issues if i.get("severity") == "high")

        score = review.get("score", 7)
        passed = score >= PASS_THRESHOLD and critical_count == 0

        logger.info("critic: score=%d, passed=%s, issues=%d (critical=%d, high=%d)",
                    score, passed, len(all_issues), critical_count, high_count)

        return {
            "critic_passed": passed,
            "critic_score": score,
            "critic_issues": all_issues,
            "critic_feedback": review.get("retry_feedback", "") if not passed else "",
            "critic_strengths": review.get("strengths", []),
            "critic_retry_needed": review.get("retry_needed", not passed),
        }

    except Exception as e:
        logger.exception("critic: LLM call failed")
        # On error, use quick check result
        passed = len([i for i in quick_issues if i["severity"] in ("critical", "high")]) == 0
        return {
            "critic_passed": passed,
            "critic_score": 6,
            "critic_issues": quick_issues,
            "critic_feedback": "",
            "critic_error": str(e),
        }


def _quick_quality_check(draft: str, intent: str) -> list[dict[str, Any]]:
    """Quick heuristic quality checks.

    Args:
        draft: Generated output
        intent: Task intent

    Returns:
        List of issue dicts
    """
    issues = []

    # Check for common code issues
    if intent in ("code_gen", "unit_test", "refactor"):
        # Check for TODO/FIXME left in code
        if "TODO" in draft or "FIXME" in draft:
            issues.append({
                "severity": "medium",
                "category": "completeness",
                "description": "Contains TODO/FIXME markers",
                "suggestion": "Complete the implementation",
            })

        # Check for placeholder text
        placeholders = ["...", "# implement", "pass  # TODO", "raise NotImplementedError"]
        for placeholder in placeholders:
            if placeholder in draft:
                issues.append({
                    "severity": "high",
                    "category": "completeness",
                    "description": f"Contains placeholder: {placeholder}",
                    "suggestion": "Provide complete implementation",
                })
                break

        # Check for hardcoded credentials (security)
        cred_patterns = [
            r'password\s*=\s*["\'][^"\']+["\']',
            r'api_key\s*=\s*["\'][^"\']+["\']',
            r'secret\s*=\s*["\'][^"\']+["\']',
        ]
        for pattern in cred_patterns:
            if re.search(pattern, draft, re.IGNORECASE):
                issues.append({
                    "severity": "critical",
                    "category": "safety",
                    "description": "Hardcoded credentials detected",
                    "suggestion": "Use environment variables or config files",
                })
                break

        # Check for SQL injection risk
        if "execute(" in draft and ("format(" in draft or "%" in draft or "f'" in draft):
            if "execute(" in draft:
                issues.append({
                    "severity": "high",
                    "category": "safety",
                    "description": "Potential SQL injection - string formatting in execute()",
                    "suggestion": "Use parameterized queries",
                })

    # Check for very short output
    if len(draft.strip()) < 50 and intent in ("code_gen", "unit_test"):
        issues.append({
            "severity": "medium",
            "category": "completeness",
            "description": "Output is very short for code generation",
            "suggestion": "Provide more complete implementation",
        })

    return issues


def _parse_critic_response(content: str) -> dict[str, Any]:
    """Parse LLM critic response.

    Args:
        content: Raw LLM response

    Returns:
        Parsed review dict
    """
    content = content.strip()

    # Remove markdown code blocks
    if content.startswith("```"):
        lines = content.split("\n")
        json_lines = []
        in_block = False
        for line in lines:
            if line.startswith("```"):
                in_block = not in_block
                continue
            if in_block or not line.startswith("```"):
                json_lines.append(line)
        content = "\n".join(json_lines)

    try:
        review = json.loads(content)
        # Validate and set defaults
        if "passed" not in review:
            review["passed"] = review.get("score", 5) >= PASS_THRESHOLD
        if "score" not in review:
            review["score"] = 7 if review["passed"] else 4
        if "issues" not in review:
            review["issues"] = []
        return review
    except json.JSONDecodeError:
        logger.warning("critic: failed to parse JSON response")
        return {
            "passed": True,
            "score": 6,
            "issues": [],
            "retry_needed": False,
        }
