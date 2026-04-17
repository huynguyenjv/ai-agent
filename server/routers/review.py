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


class FileChange(BaseModel):
    path: str
    status: str = "modified"


class ReviewAnalyzeRequest(BaseModel):
    repo: str
    pr_id: int = Field(..., ge=1)
    title: str = ""
    source_branch: str = ""
    target_branch: str = ""
    author: str = ""
    commit_sha: str = ""
    base_sha: str = ""
    head_sha: str = ""
    start_sha: str = ""
    diff: str
    files: list[FileChange] = Field(default_factory=list)

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
    from server import gitlab

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
                payload = await gitlab.get_pr_diff(**args)
            elif name == "get_mr_note":
                payload = await gitlab.get_mr_note(**args)
            elif name == "upsert_mr_comment":
                payload = await gitlab.upsert_mr_comment(**args)
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


@router.post("/review/analyze")
async def review_analyze_endpoint(
    request: ReviewAnalyzeRequest,
    req: Request,
    x_api_key: str | None = Header(None),
    authorization: str | None = Header(None),
) -> dict[str, Any]:
    """Stateless review: receive diff, return markdown + findings + inline_comments.

    Caller (e.g. gitlab-review-runner) is responsible for fetching the diff and
    posting comments. Server never touches GitLab.
    """
    verify_api_key(req, x_api_key, authorization)

    try:
        return await asyncio.wait_for(
            _run_analyze(request, req), timeout=REVIEW_TIMEOUT_SECS
        )
    except asyncio.TimeoutError:
        logger.warning(
            "review_analyze timeout after %ds for %s!%d",
            REVIEW_TIMEOUT_SECS, request.repo, request.pr_id,
        )
        raise HTTPException(
            status_code=504, detail=f"Review timeout after {REVIEW_TIMEOUT_SECS}s"
        )


async def _run_analyze(request: ReviewAnalyzeRequest, req: Request) -> dict[str, Any]:
    """Skip the agent graph and call review_analyze + review_format directly.

    review_analyze reads diff from a ToolMessage in state.messages, so we
    fabricate one from the request payload — no actual GitLab call happens.
    """
    from langchain_core.messages import ToolMessage

    from server.agent.review_analyze import review_analyze
    from server.agent.review_format import review_format
    from server.agent.upsert_mr_comment import (
        INLINE_MARKER,
        INLINE_SEVERITIES,
        _render_inline_body,
    )

    vllm_client = req.app.state.vllm_client
    model = req.app.state.vllm_model

    diff_payload = {
        "diff": request.diff,
        "commit_sha": request.commit_sha,
        "base_sha": request.base_sha,
        "head_sha": request.head_sha or request.commit_sha,
        "start_sha": request.start_sha or request.base_sha,
        "source_branch": request.source_branch,
        "target_branch": request.target_branch,
        "author": request.author,
        "title": request.title,
        "files": [f.model_dump() for f in request.files],
    }

    tool_msg = ToolMessage(
        content=json.dumps(diff_payload, ensure_ascii=False),
        tool_call_id="synthetic-analyze",
    )

    state: dict[str, Any] = {
        "messages": [tool_msg],
        "intent": "code_review",
        "review_mode": "pr",
        "pr_context": {
            "provider": "gitlab",
            "repo": request.repo,
            "pr_id": request.pr_id,
        },
        "review_findings": [],
        "auto_post": False,
    }

    try:
        analyze_result = await review_analyze(state, vllm_client=vllm_client, model=model)
    except Exception as exc:
        logger.exception("review_analyze failed")
        raise HTTPException(status_code=500, detail=f"Analyze failure: {exc}")

    state.update(analyze_result)
    pr_ctx = state.get("pr_context") or {}
    findings = list(state.get("review_findings") or [])

    try:
        formatted = review_format(state)
    except Exception as exc:
        logger.exception("review_format failed")
        raise HTTPException(status_code=500, detail=f"Format failure: {exc}")

    markdown = formatted.get("draft", "")

    # Build ready-to-post inline_comments: filter by severity + validate path/line
    # against the added_lines map review_analyze computed.
    added_lines_map: dict[str, list[int]] = (
        (formatted.get("pr_context") or pr_ctx).get("added_lines") or {}
    )
    norm_map = {k.strip().lstrip("/"): set(v or []) for k, v in added_lines_map.items()}

    inline_comments: list[dict] = []
    for f in findings:
        sev = (f.get("severity") or "low").lower()
        if sev not in INLINE_SEVERITIES:
            continue
        path_raw = (f.get("file") or "").strip()
        path = path_raw.lstrip("/")
        try:
            line = int(f.get("line", 0) or 0)
        except (TypeError, ValueError):
            line = 0
        if line <= 0 or not path:
            continue
        allowed = norm_map.get(path)
        if allowed is None:
            matches = [k for k in norm_map if k.endswith("/" + path) or k == path]
            if len(matches) == 1:
                path = matches[0]
                allowed = norm_map[path]
            else:
                continue
        if line not in allowed:
            continue
        inline_comments.append({
            "new_path": path,
            "old_path": path,
            "new_line": line,
            "body": _render_inline_body(f),
            "severity": sev,
        })

    return {
        "markdown": markdown,
        "findings": findings,
        "findings_count": _count_findings(findings),
        "inline_comments": inline_comments,
        "inline_marker": INLINE_MARKER,
        "commit_sha": request.commit_sha,
    }
