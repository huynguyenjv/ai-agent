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
            return {"pr_context": pr_ctx, "draft": body}
        except Exception as exc:
            last_exc = exc
            logger.warning("upsert_mr_comment attempt %d failed: %s", attempt + 1, exc)

    logger.error("upsert_mr_comment failed after retries: %s", last_exc)
    return {"draft": body, "pr_context": pr_ctx}
