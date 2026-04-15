"""Code review GitLab tools — V1 GitLab only.

Tools:
- get_pr_diff(provider, repo, pr_id) → diff + metadata
- get_mr_note(provider, repo, pr_id, marker) → existing AI-review note or None
- upsert_mr_comment(provider, repo, pr_id, body, note_id?) → {note_id, action}

SSL verify controlled by env GITLAB_CA_BUNDLE:
- unset → verify=True (system CA)
- path → verify=<path to CA bundle>
- "0"/"false" → verify=False (dev only, warns)
"""

from __future__ import annotations

import asyncio
import logging
import os
from functools import wraps
from urllib.parse import quote

import httpx

logger = logging.getLogger("mcp_server.tools_review")

_GITLAB_URL = os.environ.get("GITLAB_URL", "https://gitlab.com").rstrip("/")
_GITLAB_TOKEN = os.environ.get("GITLAB_TOKEN", "")
_HTTP_TIMEOUT = float(os.environ.get("GITLAB_HTTP_TIMEOUT", "30"))
_CA_BUNDLE_RAW = os.environ.get("GITLAB_CA_BUNDLE", "").strip()

_RETRY_STATUSES = {429, 500, 502, 503, 504}
_RETRY_BACKOFF = (1.0, 2.0, 4.0)


def _resolve_verify() -> bool | str:
    if not _CA_BUNDLE_RAW:
        return True
    if _CA_BUNDLE_RAW.lower() in {"0", "false", "no"}:
        logger.warning("GITLAB_CA_BUNDLE disables SSL verification — dev only")
        return False
    return _CA_BUNDLE_RAW


_VERIFY = _resolve_verify()


def _headers() -> dict:
    if not _GITLAB_TOKEN:
        raise RuntimeError("GITLAB_TOKEN not configured")
    return {"PRIVATE-TOKEN": _GITLAB_TOKEN}


def _project_id(repo: str) -> str:
    return quote(repo, safe="")


def _validate_gitlab(provider: str) -> None:
    if provider != "gitlab":
        raise ValueError(f"Provider '{provider}' not supported in V1 (GitLab only)")


def _retry_http(fn):
    """Retry on 429/5xx + transient network errors. Max 3 attempts, exponential backoff."""
    @wraps(fn)
    async def wrapper(*args, **kwargs):
        last_exc: Exception | None = None
        for attempt, delay in enumerate((*_RETRY_BACKOFF, None)):
            try:
                return await fn(*args, **kwargs)
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if status not in _RETRY_STATUSES or delay is None:
                    raise
                last_exc = exc
                logger.warning("%s: HTTP %d, retry in %.1fs (attempt %d)", fn.__name__, status, delay, attempt + 1)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                if delay is None:
                    raise
                last_exc = exc
                logger.warning("%s: %s, retry in %.1fs (attempt %d)", fn.__name__, exc, delay, attempt + 1)
            await asyncio.sleep(delay)
        if last_exc:
            raise last_exc
    return wrapper


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=_HTTP_TIMEOUT, verify=_VERIFY)


@_retry_http
async def get_pr_diff(provider: str, repo: str, pr_id: int) -> dict:
    _validate_gitlab(provider)
    pid = _project_id(repo)
    base = f"{_GITLAB_URL}/api/v4/projects/{pid}/merge_requests/{pr_id}"

    async with _client() as client:
        meta_resp = await client.get(base, headers=_headers())
        meta_resp.raise_for_status()
        meta = meta_resp.json()

        # access_raw_diffs=true bypasses GitLab's cached diff (which can be stale
        # right after a new commit is pushed) and reads straight from git.
        changes_resp = await client.get(
            f"{base}/changes",
            headers=_headers(),
            params={"access_raw_diffs": "true"},
        )
        changes_resp.raise_for_status()
        changes = changes_resp.json()

    files = []
    diff_parts = []
    for ch in changes.get("changes", []):
        path = ch.get("new_path") or ch.get("old_path") or ""
        diff_text = ch.get("diff", "") or ""
        files.append({
            "path": path,
            "status": ("added" if ch.get("new_file") else "deleted" if ch.get("deleted_file")
                       else "renamed" if ch.get("renamed_file") else "modified"),
        })
        diff_parts.append(f"--- a/{ch.get('old_path', path)}\n+++ b/{path}\n{diff_text}")

    diff_refs = meta.get("diff_refs") or {}
    return {
        "commit_sha": meta.get("sha", ""),
        "base_sha": diff_refs.get("base_sha", ""),
        "head_sha": diff_refs.get("head_sha", "") or meta.get("sha", ""),
        "start_sha": diff_refs.get("start_sha", "") or diff_refs.get("base_sha", ""),
        "source_branch": meta.get("source_branch", ""),
        "target_branch": meta.get("target_branch", ""),
        "author": (meta.get("author") or {}).get("username", ""),
        "title": meta.get("title", ""),
        "diff": "\n".join(diff_parts),
        "files": files,
    }


@_retry_http
async def get_mr_note(provider: str, repo: str, pr_id: int, marker: str) -> dict | None:
    _validate_gitlab(provider)
    pid = _project_id(repo)
    url = f"{_GITLAB_URL}/api/v4/projects/{pid}/merge_requests/{pr_id}/notes"

    async with _client() as client:
        resp = await client.get(
            url, headers=_headers(),
            params={"per_page": 100, "order_by": "updated_at", "sort": "desc"},
        )
        resp.raise_for_status()
        notes = resp.json()

    for note in notes:
        body = note.get("body", "") or ""
        if marker in body:
            return {"note_id": note.get("id"), "body": body}
    return None


@_retry_http
async def list_mr_discussions(provider: str, repo: str, pr_id: int) -> list[dict]:
    _validate_gitlab(provider)
    pid = _project_id(repo)
    url = f"{_GITLAB_URL}/api/v4/projects/{pid}/merge_requests/{pr_id}/discussions"
    results: list[dict] = []
    page = 1
    async with _client() as client:
        while True:
            resp = await client.get(
                url, headers=_headers(),
                params={"per_page": 100, "page": page},
            )
            resp.raise_for_status()
            batch = resp.json()
            if not batch:
                break
            results.extend(batch)
            if len(batch) < 100:
                break
            page += 1
    return results


@_retry_http
async def resolve_mr_discussion(provider: str, repo: str, pr_id: int, discussion_id: str) -> dict:
    _validate_gitlab(provider)
    pid = _project_id(repo)
    url = (
        f"{_GITLAB_URL}/api/v4/projects/{pid}/merge_requests/{pr_id}"
        f"/discussions/{discussion_id}?resolved=true"
    )
    async with _client() as client:
        resp = await client.put(url, headers=_headers())
        resp.raise_for_status()
    return {"discussion_id": discussion_id, "resolved": True}


@_retry_http
async def create_mr_discussion(
    provider: str, repo: str, pr_id: int, body: str,
    base_sha: str, head_sha: str, start_sha: str,
    new_path: str, new_line: int, old_path: str | None = None,
) -> dict:
    """Create an inline discussion pinned to new_path:new_line on the MR diff."""
    _validate_gitlab(provider)
    pid = _project_id(repo)
    url = f"{_GITLAB_URL}/api/v4/projects/{pid}/merge_requests/{pr_id}/discussions"
    payload = {
        "body": body,
        "position[position_type]": "text",
        "position[base_sha]": base_sha,
        "position[head_sha]": head_sha,
        "position[start_sha]": start_sha,
        "position[new_path]": new_path,
        "position[old_path]": old_path or new_path,
        "position[new_line]": str(new_line),
    }
    async with _client() as client:
        resp = await client.post(url, headers=_headers(), data=payload)
        resp.raise_for_status()
        data = resp.json()
    return {"discussion_id": data.get("id"), "action": "created"}


@_retry_http
async def upsert_mr_comment(
    provider: str, repo: str, pr_id: int, body: str, note_id: int | None = None,
) -> dict:
    _validate_gitlab(provider)
    pid = _project_id(repo)
    base = f"{_GITLAB_URL}/api/v4/projects/{pid}/merge_requests/{pr_id}/notes"

    async with _client() as client:
        if note_id:
            resp = await client.put(f"{base}/{note_id}", headers=_headers(), json={"body": body})
            resp.raise_for_status()
            return {"note_id": note_id, "action": "updated"}
        resp = await client.post(base, headers=_headers(), json={"body": body})
        resp.raise_for_status()
        data = resp.json()
        return {"note_id": data.get("id"), "action": "created"}
