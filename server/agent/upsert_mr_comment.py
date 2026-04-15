"""Node: upsert_mr_comment — Code Review Flow.

GET existing note with marker → PUT (update) or POST (create).
Parses previous_reviews from old body and re-renders markdown with merged history.
Calls GitLab API directly via import (not MCP stdio), since this runs inside the agent.
"""

from __future__ import annotations

import logging
import os

from server.agent.state import AgentState
from server.agent.review_format import MARKER, review_format

logger = logging.getLogger("server.agent.upsert_mr_comment")

INLINE_MARKER = os.environ.get("AI_REVIEWER_INLINE_MARKER", "AI_REVIEW_INLINE:v1")
INLINE_SEVERITIES = {"critical", "high", "medium"}
_SEV_EMOJI = {"critical": "🔴", "high": "🟠", "medium": "🟡"}


def _render_inline_body(f: dict) -> str:
    sev = f.get("severity", "low")
    emoji = _SEV_EMOJI.get(sev, "🔵")
    fw = f.get("framework") or "—"
    title = (f.get("title") or "").strip() or "Issue"
    msg = (f.get("message") or "").strip()
    sugg = (f.get("suggestion") or "").strip()
    parts = [f"<!-- {INLINE_MARKER} -->", f"{emoji} **[{fw}] {title}**"]
    if msg and msg != title:
        parts.append("")
        parts.append(msg)
    if sugg and sugg.lower() not in {"null", "none", ""}:
        parts.append("")
        parts.append(f"💡 **Suggestion:**\n\n{sugg}")
    return "\n".join(parts)


async def _resolve_old_inline(provider: str, repo: str, pr_id: int) -> int:
    from mcp_server.tools_review import list_mr_discussions, resolve_mr_discussion
    try:
        discussions = await list_mr_discussions(provider=provider, repo=repo, pr_id=pr_id)
    except Exception as exc:
        logger.warning("list_mr_discussions failed: %s", exc)
        return 0

    resolved = 0
    for d in discussions:
        notes = d.get("notes") or []
        if not notes:
            continue
        first = notes[0]
        if first.get("resolved"):
            continue
        body = first.get("body", "") or ""
        if INLINE_MARKER not in body:
            continue
        did = d.get("id")
        if not did:
            continue
        try:
            await resolve_mr_discussion(provider=provider, repo=repo, pr_id=pr_id, discussion_id=did)
            resolved += 1
        except Exception as exc:
            logger.warning("resolve_mr_discussion %s failed: %s", did, exc)
    return resolved


async def _post_inline_findings(
    provider: str, repo: str, pr_id: int,
    findings: list[dict], pr_ctx: dict,
) -> tuple[int, int]:
    from mcp_server.tools_review import create_mr_discussion

    base_sha = pr_ctx.get("base_sha", "")
    head_sha = pr_ctx.get("head_sha", "")
    start_sha = pr_ctx.get("start_sha", "") or base_sha
    if not (base_sha and head_sha):
        logger.warning("inline skipped: missing diff_refs (base=%s head=%s)", base_sha, head_sha)
        return 0, 0

    added_lines_map: dict[str, list[int]] = pr_ctx.get("added_lines") or {}
    posted = 0
    skipped = 0
    for f in findings:
        sev = f.get("severity", "low")
        if sev not in INLINE_SEVERITIES:
            continue
        path = (f.get("file") or "").strip()
        try:
            line = int(f.get("line", 0) or 0)
        except (TypeError, ValueError):
            line = 0
        allowed = set(added_lines_map.get(path) or [])
        if line <= 0 or (allowed and line not in allowed):
            skipped += 1
            continue
        body = _render_inline_body(f)
        try:
            await create_mr_discussion(
                provider=provider, repo=repo, pr_id=pr_id, body=body,
                base_sha=base_sha, head_sha=head_sha, start_sha=start_sha,
                new_path=path, new_line=line,
            )
            posted += 1
        except Exception as exc:
            logger.warning("create_mr_discussion %s:%d failed: %s", path, line, exc)
            skipped += 1
    return posted, skipped


async def upsert_mr_comment(state: AgentState) -> dict:
    if not state.get("auto_post"):
        return {}

    pr_ctx = dict(state.get("pr_context") or {})
    provider = pr_ctx.get("provider", "gitlab")
    if provider != "gitlab":
        logger.warning("upsert_mr_comment: provider %s not supported in V1", provider)
        return {}

    repo = pr_ctx.get("repo")
    pr_id = pr_ctx.get("pr_id")
    if not repo or not pr_id:
        logger.warning("upsert_mr_comment: missing repo or pr_id")
        return {}

    # Import review tools directly (decision §16.1)
    from mcp_server.tools_review import get_mr_note, upsert_mr_comment as api_upsert

    # Step 1: fetch existing note
    existing = None
    try:
        existing = await get_mr_note(provider="gitlab", repo=repo, pr_id=pr_id, marker=MARKER)
    except Exception as exc:
        logger.warning("get_mr_note failed: %s", exc)

    # Step 2: parse previous_reviews from old body
    if existing and existing.get("body"):
        from server.agent.review_format import _parse_previous_reviews
        prev = _parse_previous_reviews(existing["body"])
        pr_ctx["previous_reviews"] = prev

    # Step 3: re-render markdown with merged history
    rerender_state = dict(state)
    rerender_state["pr_context"] = pr_ctx
    formatted = review_format(rerender_state)
    body = formatted["draft"]
    pr_ctx = formatted.get("pr_context") or pr_ctx

    # Step 4: upsert (with 1 retry)
    note_id = existing.get("note_id") if existing else None
    last_exc: Exception | None = None
    for attempt in range(2):
        try:
            result = await api_upsert(
                provider="gitlab", repo=repo, pr_id=pr_id, body=body, note_id=note_id
            )
            pr_ctx["note_id"] = result.get("note_id")
            logger.info(
                "MR comment %s: repo=%s mr=!%s note_id=%s",
                result.get("action"), repo, pr_id, result.get("note_id"),
            )
            # Inline discussions: resolve old AI threads, then post new for critical/high/medium.
            try:
                resolved = await _resolve_old_inline(provider="gitlab", repo=repo, pr_id=pr_id)
                findings = list(state.get("review_findings") or [])
                posted, skipped = await _post_inline_findings(
                    provider="gitlab", repo=repo, pr_id=pr_id,
                    findings=findings, pr_ctx=pr_ctx,
                )
                logger.info(
                    "inline discussions: resolved=%d posted=%d skipped=%d",
                    resolved, posted, skipped,
                )
                pr_ctx["inline_posted"] = posted
                pr_ctx["inline_resolved"] = resolved
                pr_ctx["inline_skipped"] = skipped
            except Exception as exc:
                logger.warning("inline discussion flow failed: %s", exc)
            return {"pr_context": pr_ctx, "draft": body}
        except Exception as exc:
            last_exc = exc
            logger.warning("upsert_mr_comment attempt %d failed: %s", attempt + 1, exc)

    logger.error("upsert_mr_comment failed after retries: %s", last_exc)
    return {"draft": body, "pr_context": pr_ctx}
