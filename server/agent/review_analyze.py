"""Node: review_analyze — Code Review Flow.

Analyze diff (PR mode) or file content (file mode) via LLM.
Returns structured findings: bugs, security, syntax.
"""

from __future__ import annotations

import json
import logging
import os
import re

from langchain_core.messages import ToolMessage
from openai import AsyncOpenAI

from server.agent.state import AgentState

logger = logging.getLogger("server.agent.review_analyze")

MAX_DIFF_TOKENS = int(os.environ.get("REVIEW_MAX_DIFF_TOKENS", "12000"))

SYSTEM_PROMPT = """You are a senior code reviewer. Review the provided content strictly for:
1. BUGS — logic errors, null deref, off-by-one, race conditions, incorrect error handling, wrong business logic
2. SECURITY — OWASP top 10, secrets, injection (SQL/command/path), SSRF, unsafe deserialization, missing auth
3. SYNTAX — syntax errors, missing imports, clear naming violations, magic numbers, dead code

Rules:
- Only report REAL issues. Do NOT suggest stylistic preferences unless they clearly violate a convention.
- Each finding MUST map to a specific line number.
- Severity:
  - blocker: security vulnerabilities, crashes, data loss
  - major: logic bugs, missing error handling, incorrect API usage
  - minor: style issues, minor inefficiencies
  - info: observations, not actionable
- If no issues found, return [].
- Output STRICTLY as JSON array. No markdown fences, no prose, no explanation.

Output schema:
[{"file": "path", "line": N, "severity": "blocker|major|minor|info", "category": "bug|security|syntax", "title": "short", "description": "detailed", "suggestion": "fix code or guidance"}]
"""


def _extract_last_user_text(messages: list) -> str:
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "human":
            return msg.content or ""
        if isinstance(msg, dict) and msg.get("role") == "user":
            return msg.get("content", "") or ""
    return ""


def _extract_tool_result_diff(messages: list) -> dict | None:
    """Find last ToolMessage that looks like get_pr_diff result."""
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage) or (hasattr(msg, "type") and msg.type == "tool"):
            try:
                payload = json.loads(msg.content)
            except Exception:
                continue
            if isinstance(payload, dict) and "diff" in payload:
                return payload
    return None


def _parse_json_array(text: str) -> list[dict] | None:
    """Parse LLM output as JSON array. Strip markdown fences if present."""
    stripped = text.strip()
    # Remove ```json ... ``` fences if LLM ignored instructions
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, re.DOTALL)
    if fence:
        stripped = fence.group(1).strip()
    try:
        parsed = json.loads(stripped)
    except Exception:
        return None
    if not isinstance(parsed, list):
        return None
    return parsed


async def _call_llm(vllm_client: AsyncOpenAI, model: str, system: str, user: str) -> str:
    resp = await vllm_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=4096,
        temperature=0.1,
        stream=False,
    )
    return resp.choices[0].message.content or ""


async def review_analyze(state: AgentState, vllm_client: AsyncOpenAI, model: str) -> dict:
    """Analyze diff/file → review_findings."""
    messages = state.get("messages", [])
    review_mode = state.get("review_mode") or "file"
    pr_ctx = dict(state.get("pr_context") or {})

    # Build user prompt content
    if review_mode == "pr":
        diff_payload = _extract_tool_result_diff(messages)
        if not diff_payload:
            logger.warning("review_analyze: no PR diff found in tool results")
            return {"review_findings": [], "draft": ""}

        pr_ctx.update({
            "commit_sha": diff_payload.get("commit_sha", ""),
            "base_sha": diff_payload.get("base_sha", ""),
            "files": diff_payload.get("files", []),
            "diff": diff_payload.get("diff", ""),
            "title": diff_payload.get("title", ""),
        })
        diff_text = diff_payload.get("diff", "")
        if len(diff_text) > MAX_DIFF_TOKENS * 4:  # rough char estimate
            diff_text = diff_text[: MAX_DIFF_TOKENS * 4]
            logger.warning("review_analyze: diff truncated to %d chars", len(diff_text))

        user_prompt = (
            f"PR title: {diff_payload.get('title', '')}\n"
            f"Source → target: {diff_payload.get('source_branch', '')} → {diff_payload.get('target_branch', '')}\n"
            f"Commit: {diff_payload.get('commit_sha', '')[:8]}\n\n"
            f"Unified diff:\n```diff\n{diff_text}\n```\n\n"
            "Review this diff and output findings as JSON array per the schema."
        )
    else:
        # file mode — Continue sent full file in code fence
        user_text = _extract_last_user_text(messages)
        user_prompt = (
            f"{user_text}\n\n"
            "Review the above code and output findings as JSON array per the schema."
        )

    # Call LLM with 1 retry on JSON parse failure
    findings: list[dict] = []
    for attempt in range(2):
        try:
            raw = await _call_llm(vllm_client, model, SYSTEM_PROMPT, user_prompt)
        except Exception as exc:
            logger.error("review_analyze LLM call failed: %s", exc)
            break
        parsed = _parse_json_array(raw)
        if parsed is not None:
            findings = parsed
            break
        logger.warning("review_analyze attempt %d: invalid JSON, retrying", attempt + 1)
        user_prompt = (
            user_prompt
            + "\n\nIMPORTANT: your previous response was not valid JSON. "
            "Return ONLY a JSON array, no prose, no markdown fences."
        )

    # Normalise records
    clean: list[dict] = []
    for f in findings:
        if not isinstance(f, dict):
            continue
        clean.append({
            "file": str(f.get("file", "") or ""),
            "line": int(f.get("line", 0) or 0),
            "severity": str(f.get("severity", "info") or "info").lower(),
            "category": str(f.get("category", "bug") or "bug").lower(),
            "title": str(f.get("title", "") or "")[:200],
            "description": str(f.get("description", "") or ""),
            "suggestion": str(f.get("suggestion", "") or ""),
        })

    return {"review_findings": clean, "pr_context": pr_ctx or None}
