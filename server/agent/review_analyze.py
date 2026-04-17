"""Node: review_analyze — Code Review Flow.

Analyze PR diff or file content via LLM using frameworks-based prompt
(OWASP Top 10 + CWE Top 25). Output structured JSON findings.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re

from langchain_core.messages import ToolMessage
from openai import AsyncOpenAI

from server.agent.prompts import load_prompt
from server.agent.state import AgentState

logger = logging.getLogger("server.agent.review_analyze")

MAX_DIFF_CHARS_PER_FILE = int(os.environ.get("REVIEW_MAX_DIFF_CHARS_PER_FILE", "40000"))
LLM_TIMEOUT_SECS = int(os.environ.get("REVIEW_LLM_TIMEOUT_SECS", "60"))
LLM_MAX_TOKENS = int(os.environ.get("REVIEW_LLM_MAX_TOKENS", "4096"))


def _extract_last_user_text(messages: list) -> str:
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "human":
            return msg.content or ""
        if isinstance(msg, dict) and msg.get("role") == "user":
            return msg.get("content", "") or ""
    return ""


def _extract_tool_result_diff(messages: list) -> dict | None:
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage) or (hasattr(msg, "type") and msg.type == "tool"):
            try:
                payload = json.loads(msg.content)
            except Exception:
                continue
            if isinstance(payload, dict) and "diff" in payload:
                return payload
    return None


_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")


def _annotate_diff_lines(diff: str) -> str:
    """Prefix each +/context line with [L<new_line>] so the LLM can only cite real
    line numbers. Unchanged/removed lines keep their `-`/` ` marker but also get the
    annotation so context is unambiguous. Lines without a valid hunk are passed through.
    """
    out: list[str] = []
    new_line = 0
    in_hunk = False
    for raw in diff.splitlines():
        m = _HUNK_RE.match(raw)
        if m:
            new_line = int(m.group(1))
            in_hunk = True
            out.append(raw)
            continue
        if not in_hunk or raw.startswith(("--- ", "+++ ", "diff ", "index ")):
            out.append(raw)
            continue
        if raw.startswith("+") and not raw.startswith("+++"):
            out.append(f"[L{new_line}] {raw}")
            new_line += 1
        elif raw.startswith("-") and not raw.startswith("---"):
            out.append(f"[L---] {raw}")
        elif raw.startswith("\\"):  # "\ No newline at end of file"
            out.append(raw)
        else:
            out.append(f"[L{new_line}] {raw}")
            new_line += 1
    return "\n".join(out)


def _extract_added_lines(raw_diff: str) -> list[int]:
    """Return list of new-file line numbers for every `+` line in the diff."""
    added: list[int] = []
    new_line = 0
    in_hunk = False
    for raw in raw_diff.splitlines():
        m = _HUNK_RE.match(raw)
        if m:
            new_line = int(m.group(1))
            in_hunk = True
            continue
        if not in_hunk or raw.startswith(("--- ", "+++ ", "diff ", "index ")):
            continue
        if raw.startswith("+") and not raw.startswith("+++"):
            added.append(new_line)
            new_line += 1
        elif raw.startswith("-") and not raw.startswith("---"):
            pass
        elif raw.startswith("\\"):
            pass
        else:
            new_line += 1
    return added


def _split_diff_by_file(diff: str, max_chars: int = MAX_DIFF_CHARS_PER_FILE) -> list[dict]:
    """Split unified diff by file boundary. Each chunk's diff is annotated with
    [L<new_line>] prefixes so the LLM cites only real post-fix line numbers.
    Also records added_lines (set of `+` line numbers) for downstream inline comments.
    Oversize files truncated with notice in diff body."""
    if not diff.strip():
        return []
    parts = re.split(r"(?m)^(?=--- a/)", diff)
    chunks: list[dict] = []
    for part in parts:
        if not part.strip():
            continue
        m = re.search(r"\+\+\+ b/(.+)", part)
        path = m.group(1).strip() if m else "<unknown>"
        added = _extract_added_lines(part)
        annotated = _annotate_diff_lines(part)
        if len(annotated) > max_chars:
            annotated = annotated[:max_chars] + f"\n... [diff truncated, original {len(annotated)} chars]\n"
        chunks.append({
            "path": path, "diff": annotated, "added_lines": added,
            "skipped": False, "reason": None,
        })
    return chunks


def _parse_json_object(text: str) -> dict | None:
    stripped = text.strip()
    # Strip <tool_call>...</tool_call> or <tools>...</tools> wrappers Qwen models love
    tag = re.match(r"^<(tool_call|tools)>\s*(.*?)\s*</\1>\s*$", stripped, re.DOTALL)
    if tag:
        stripped = tag.group(2).strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, re.DOTALL)
    if fence:
        stripped = fence.group(1).strip()
    try:
        obj = json.loads(stripped)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    # If model wrapped output in tool_call envelope {name, arguments}, unwrap
    if set(obj.keys()) == {"name", "arguments"} and isinstance(obj.get("arguments"), dict):
        return obj["arguments"]
    return obj


async def _call_llm(client: AsyncOpenAI, model: str, system: str, user: str) -> str:
    resp = await asyncio.wait_for(
        client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=LLM_MAX_TOKENS,
            temperature=0.1,
            stream=False,  # batch response — V1 per spec (non-streaming review)
        ),
        timeout=LLM_TIMEOUT_SECS,
    )
    return resp.choices[0].message.content or ""


def _normalise_findings(raw: list) -> list[dict]:
    clean: list[dict] = []
    for f in raw or []:
        if not isinstance(f, dict):
            continue
        clean.append({
            "file": str(f.get("file", "") or ""),
            "line": int(f.get("line", 0) or 0),
            "severity": str(f.get("severity", "low") or "low").lower(),
            "category": str(f.get("category", "bug") or "bug").lower(),
            "framework": str(f.get("framework", "") or ""),
            "title": str(f.get("title", "") or "")[:200],
            "message": str(f.get("message", f.get("description", "")) or ""),
            "suggestion": str(f.get("suggestion", "") or ""),
        })
    return clean


async def _analyze_once(client: AsyncOpenAI, model: str, system: str, user: str) -> dict | None:
    """One LLM call + JSON parse. Retry once on parse failure with stricter prompt."""
    for attempt in range(2):
        try:
            raw = await _call_llm(client, model, system, user)
        except asyncio.TimeoutError:
            logger.warning("review_analyze LLM timeout (attempt %d)", attempt + 1)
            return None
        except Exception as exc:
            logger.error("review_analyze LLM error: %s", exc)
            return None
        parsed = _parse_json_object(raw)
        if parsed is not None:
            return parsed
        logger.warning("review_analyze invalid JSON (attempt %d), raw=%s", attempt + 1, raw[:200])
        user = user + "\n\nIMPORTANT: previous response was invalid JSON. Return ONLY a JSON object matching the schema, no prose, no markdown fences."
    return None


async def review_analyze(state: AgentState, vllm_client: AsyncOpenAI, model: str) -> dict:
    messages = state.get("messages", [])
    review_mode = state.get("review_mode") or "file"
    pr_ctx = dict(state.get("pr_context") or {})

    system = load_prompt("review_system")

    if review_mode == "pr":
        diff_payload = _extract_tool_result_diff(messages)
        if not diff_payload:
            logger.warning("review_analyze: no PR diff found")
            return {"review_findings": [], "draft": ""}

        pr_ctx.update({
            "commit_sha": diff_payload.get("commit_sha", ""),
            "base_sha": diff_payload.get("base_sha", ""),
            "head_sha": diff_payload.get("head_sha", ""),
            "start_sha": diff_payload.get("start_sha", ""),
            "files": diff_payload.get("files", []),
            "title": diff_payload.get("title", ""),
        })

        diff_text = diff_payload.get("diff", "")
        chunks = _split_diff_by_file(diff_text)
        pr_ctx["added_lines"] = {c["path"]: c["added_lines"] for c in chunks}
        logger.info("review_analyze: %d file chunks", len(chunks))

        user_tpl = load_prompt("review_user_pr")
        all_findings: list[dict] = []
        summaries: list[str] = []

        for chunk in chunks:
            user = user_tpl.format(
                title=diff_payload.get("title", ""),
                source_branch=diff_payload.get("source_branch", ""),
                target_branch=diff_payload.get("target_branch", ""),
                commit_short=(diff_payload.get("commit_sha", "") or "")[:8],
                file_list=chunk["path"],
                diff=chunk["diff"],
            )
            result = await _analyze_once(vllm_client, model, system, user)
            if result is None:
                continue
            summaries.append(result.get("summary", ""))
            all_findings.extend(_normalise_findings(result.get("findings", [])))

        # Keep only first meaningful summary to avoid concatenated rambling
        first_summary = next((s for s in summaries if s and s.strip()), "")
        pr_ctx["summary"] = first_summary[:240]
        return {"review_findings": all_findings, "pr_context": pr_ctx or None}

    # File mode (Continue)
    user_tpl = load_prompt("review_user_file")
    user_text = _extract_last_user_text(messages)
    user = user_tpl.format(user_content=user_text)
    result = await _analyze_once(vllm_client, model, system, user)
    if result is None:
        return {"review_findings": [], "draft": ""}
    return {
        "review_findings": _normalise_findings(result.get("findings", [])),
        "pr_context": {**pr_ctx, "summary": result.get("summary", "")[:500]} if pr_ctx or result.get("summary") else None,
    }
