"""POST /review/pr — CI-triggered code review for GitLab MRs.

Runs the agent graph, executes MCP tool calls in-process (import direct),
then re-invokes the graph with tool results until completion.

V1: GitLab only. Agent posts/updates a single MR note identified by marker.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Literal

from fastapi import APIRouter, Header, HTTPException, Request
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import BaseModel, Field

from server.auth import verify_api_key

logger = logging.getLogger("server.review")

REVIEW_TIMEOUT_SECS = int(os.environ.get("REVIEW_TIMEOUT_SECS", "180"))

router = APIRouter()


class ReviewPRRequest(BaseModel):
    provider: Literal["gitlab"] = "gitlab"
    repo: str = Field(..., description="Project path, e.g. group/project")
    pr_id: int = Field(..., ge=1)
    mode: Literal["pr"] = "pr"


def _count_findings(findings: list[dict]) -> dict:
    counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    for f in findings or []:
        sev = f.get("severity", "low")
        if sev in counts:
            counts[sev] += 1
    return counts


async def _execute_tool_calls(tool_calls: list[dict]) -> list[ToolMessage]:
    """Execute pending MCP tool calls in-process. V1 supports review tools only here."""
    from mcp_server import tools_review

    results: list[ToolMessage] = []
    for tc in tool_calls:
        fn = (tc.get("function") or {})
        name = fn.get("name", "")
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except Exception:
            args = {}

        try:
            if name == "get_pr_diff":
                payload = await tools_review.get_pr_diff(**args)
            elif name == "get_mr_note":
                payload = await tools_review.get_mr_note(**args)
            elif name == "upsert_mr_comment":
                payload = await tools_review.upsert_mr_comment(**args)
            else:
                payload = {"error": f"Tool {name} not supported in /review/pr"}
        except Exception as exc:
            logger.exception("Tool %s failed", name)
            payload = {"error": str(exc)}

        results.append(ToolMessage(
            content=json.dumps(payload, ensure_ascii=False),
            tool_call_id=tc.get("id", ""),
        ))
    return results


@router.post("/review/pr")
async def review_pr(
    request: ReviewPRRequest,
    req: Request,
    x_api_key: str | None = Header(None),
    authorization: str | None = Header(None),
) -> dict[str, Any]:
    verify_api_key(req, x_api_key, authorization)

    try:
        return await asyncio.wait_for(_run_review(request, req), timeout=REVIEW_TIMEOUT_SECS)
    except asyncio.TimeoutError:
        logger.warning("review_pr timeout after %ds for %s!%d", REVIEW_TIMEOUT_SECS, request.repo, request.pr_id)
        raise HTTPException(status_code=504, detail=f"Review timeout after {REVIEW_TIMEOUT_SECS}s")


async def _run_review(request: ReviewPRRequest, req: Request) -> dict[str, Any]:
    from server.agent.graph import build_agent_graph

    vllm_client = req.app.state.vllm_client
    model = req.app.state.vllm_model
    qdrant = req.app.state.qdrant
    embedder = req.app.state.embedder

    agent = build_agent_graph(
        vllm_client=vllm_client,
        model=model,
        qdrant=qdrant,
        embedder=embedder,
        sse_callback=None,
    )

    user_msg = HumanMessage(
        content=(
            f"Review merge request !{request.pr_id} in repo {request.repo}. "
            f"Fetch the diff and produce a code review."
        )
    )

    state: dict[str, Any] = {
        "messages": [user_msg],
        "intent": "code_review",
        "active_file": None,
        "mentioned_files": [],
        "freshness_signal": False,
        "force_reindex": False,
        "rag_chunks": [],
        "rag_hit": False,
        "hash_verified": False,
        "tool_results": [],
        "context_assembled": "",
        "draft": "",
        "emitted_steps": [],
        "volatile_rejected": False,
        "pending_tool_calls": [],
        "is_tool_result_turn": False,
        "tool_turns_used": 0,
        "review_mode": "pr",
        "pr_context": {
            "provider": request.provider,
            "repo": request.repo,
            "pr_id": request.pr_id,
        },
        "review_findings": [],
        "output_format": "markdown",
        "auto_post": True,
    }

    # Turn 1 — expect pending_tool_calls for get_pr_diff
    try:
        result = await agent.ainvoke(state)
    except Exception as exc:
        logger.exception("agent turn 1 failed")
        raise HTTPException(status_code=500, detail=f"Agent failure: {exc}")

    if result.get("pending_tool_calls"):
        tool_calls = result["pending_tool_calls"]
        tool_messages = await _execute_tool_calls(tool_calls)

        turn2: dict[str, Any] = {
            **result,
            "messages": list(result.get("messages") or []) + [
                AIMessage(content="", additional_kwargs={"tool_calls": tool_calls}),
                *tool_messages,
            ],
            "is_tool_result_turn": True,
            "pending_tool_calls": [],
        }
        try:
            result = await agent.ainvoke(turn2)
        except Exception as exc:
            logger.exception("agent turn 2 failed")
            raise HTTPException(status_code=500, detail=f"Agent failure: {exc}")

    pr_ctx = result.get("pr_context") or {}
    findings = result.get("review_findings") or []

    return {
        "commit_sha": pr_ctx.get("commit_sha", ""),
        "note_id": pr_ctx.get("note_id"),
        "markdown": result.get("draft", ""),
        "findings_count": _count_findings(findings),
    }
