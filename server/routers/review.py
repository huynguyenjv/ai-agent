"""POST /review/analyze — stateless code review for an external runner.

The caller (e.g. gitlab-review-runner) fetches the diff from GitLab and POSTs it
here; the server only runs the LLM analysis and returns markdown + findings +
inline_comments. The server never touches GitLab itself.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Request
from langchain_core.messages import ToolMessage
from pydantic import BaseModel, Field

from server.auth import verify_api_key

logger = logging.getLogger("server.review")

REVIEW_TIMEOUT_SECS = int(os.environ.get("REVIEW_TIMEOUT_SECS", "180"))

router = APIRouter()


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


def _count_findings(findings: list[dict]) -> dict:
    counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    for f in findings or []:
        sev = f.get("severity", "low")
        if sev in counts:
            counts[sev] += 1
    return counts


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
    from server.agent.review_analyze import review_analyze
    from server.agent.review_format import (
        INLINE_MARKER,
        INLINE_SEVERITIES,
        _render_inline_body,
        review_format,
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
