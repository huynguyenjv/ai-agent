"""Code review MCP tools — V1 GitLab only.

Tools:
- get_pr_diff(provider, repo, pr_id) → {commit_sha, base_sha, diff, files, ...}
- get_mr_note(provider, repo, pr_id, marker) → {note_id, body} | None
- upsert_mr_comment(provider, repo, pr_id, body, note_id?) → {note_id, action}

Uses GITLAB_URL + GITLAB_TOKEN (personal access token of maintainer, decision §16.3).
"""

from __future__ import annotations

import logging
import os
from urllib.parse import quote

import httpx

logger = logging.getLogger("mcp_server.tools_review")

_GITLAB_URL = os.environ.get("GITLAB_URL", "https://gitlab.com").rstrip("/")
_GITLAB_TOKEN = os.environ.get("GITLAB_TOKEN", "")
_HTTP_TIMEOUT = float(os.environ.get("GITLAB_HTTP_TIMEOUT", "30"))


def _headers() -> dict:
    if not _GITLAB_TOKEN:
        raise RuntimeError("GITLAB_TOKEN not configured")
    return {"PRIVATE-TOKEN": _GITLAB_TOKEN}


def _project_id(repo: str) -> str:
    """URL-encode project path (e.g. 'group/sub/project') for GitLab API."""
    return quote(repo, safe="")


def _validate_gitlab(provider: str) -> None:
    if provider != "gitlab":
        raise ValueError(f"Provider '{provider}' not supported in V1 (GitLab only)")


async def get_pr_diff(provider: str, repo: str, pr_id: int) -> dict:
    """Fetch MR diff + metadata.

    GitLab API:
      GET /projects/:id/merge_requests/:iid        → metadata
      GET /projects/:id/merge_requests/:iid/changes → changes with per-file diff
    """
    _validate_gitlab(provider)
    pid = _project_id(repo)
    base = f"{_GITLAB_URL}/api/v4/projects/{pid}/merge_requests/{pr_id}"

    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT, verify=False) as client:
        meta_resp = await client.get(base, headers=_headers())
        meta_resp.raise_for_status()
        meta = meta_resp.json()

        changes_resp = await client.get(f"{base}/changes", headers=_headers())
        changes_resp.raise_for_status()
        changes = changes_resp.json()

    files = []
    diff_parts = []
    for ch in changes.get("changes", []):
        path = ch.get("new_path") or ch.get("old_path") or ""
        diff_text = ch.get("diff", "") or ""
        files.append({
            "path": path,
            "status": ("added" if ch.get("new_file") else "deleted" if ch.get("deleted_file") else
                       "renamed" if ch.get("renamed_file") else "modified"),
        })
        diff_parts.append(f"--- a/{ch.get('old_path', path)}\n+++ b/{path}\n{diff_text}")

    return {
        "commit_sha": meta.get("sha", ""),
        "base_sha": (meta.get("diff_refs") or {}).get("base_sha", ""),
        "source_branch": meta.get("source_branch", ""),
        "target_branch": meta.get("target_branch", ""),
        "author": (meta.get("author") or {}).get("username", ""),
        "title": meta.get("title", ""),
        "diff": "\n".join(diff_parts),
        "files": files,
    }


async def get_mr_note(provider: str, repo: str, pr_id: int, marker: str) -> dict | None:
    """Find existing AI review note by marker substring in body.

    Returns the most recent matching note (decision §16.3: filter by marker only, not by author).
    """
    _validate_gitlab(provider)
    pid = _project_id(repo)
    url = f"{_GITLAB_URL}/api/v4/projects/{pid}/merge_requests/{pr_id}/notes"

    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT, verify=False) as client:
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


async def upsert_mr_comment(
    provider: str, repo: str, pr_id: int, body: str, note_id: int | None = None
) -> dict:
    """Create or update an MR note. Returns {note_id, action}."""
    _validate_gitlab(provider)
    pid = _project_id(repo)
    base = f"{_GITLAB_URL}/api/v4/projects/{pid}/merge_requests/{pr_id}/notes"

    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT, verify=False) as client:
        if note_id:
            resp = await client.put(
                f"{base}/{note_id}", headers=_headers(), json={"body": body},
            )
            resp.raise_for_status()
            return {"note_id": note_id, "action": "updated"}
        resp = await client.post(base, headers=_headers(), json={"body": body})
        resp.raise_for_status()
        data = resp.json()
        return {"note_id": data.get("id"), "action": "created"}
