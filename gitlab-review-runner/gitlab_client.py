"""Synchronous GitLab API client for the review runner.

Ported from mcp_server/tools_review.py but sync (runner is a short-lived
one-shot script, async adds no benefit). Simple retry for 429/5xx.
"""

from __future__ import annotations

import logging
import time
from functools import wraps
from urllib.parse import quote

import httpx

log = logging.getLogger("runner.gitlab")

_RETRY_STATUSES = {429, 500, 502, 503, 504}
_RETRY_BACKOFF = (1.0, 2.0, 4.0)


def _retry(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        last_exc: Exception | None = None
        for attempt, delay in enumerate((*_RETRY_BACKOFF, None)):
            try:
                return fn(*args, **kwargs)
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if status not in _RETRY_STATUSES or delay is None:
                    raise
                last_exc = exc
                log.warning("%s: HTTP %d, retry in %.1fs", fn.__name__, status, delay)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                if delay is None:
                    raise
                last_exc = exc
                log.warning("%s: %s, retry in %.1fs", fn.__name__, exc, delay)
            time.sleep(delay)
        if last_exc:
            raise last_exc
    return wrapper


class GitLabClient:
    def __init__(self, base_url: str, token: str, verify: bool | str, timeout: float):
        self.base_url = base_url.rstrip("/")
        self._headers = {"PRIVATE-TOKEN": token}
        self._client = httpx.Client(timeout=timeout, verify=verify)

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    @staticmethod
    def _pid(repo: str) -> str:
        return quote(repo, safe="")

    def _mr_url(self, repo: str, pr_id: int) -> str:
        return f"{self.base_url}/api/v4/projects/{self._pid(repo)}/merge_requests/{pr_id}"

    @_retry
    def fetch_diff(self, repo: str, pr_id: int) -> dict:
        base = self._mr_url(repo, pr_id)

        meta_resp = self._client.get(base, headers=self._headers)
        meta_resp.raise_for_status()
        meta = meta_resp.json()

        changes_resp = self._client.get(
            f"{base}/changes",
            headers=self._headers,
            params={"access_raw_diffs": "true"},
        )
        changes_resp.raise_for_status()
        changes = changes_resp.json()

        files = []
        diff_parts = []
        for ch in changes.get("changes", []):
            path = ch.get("new_path") or ch.get("old_path") or ""
            diff_text = ch.get("diff", "") or ""
            if ch.get("new_file"):
                status = "added"
            elif ch.get("deleted_file"):
                status = "deleted"
            elif ch.get("renamed_file"):
                status = "renamed"
            else:
                status = "modified"
            files.append({"path": path, "status": status})
            diff_parts.append(
                f"--- a/{ch.get('old_path', path)}\n+++ b/{path}\n{diff_text}"
            )

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

    @_retry
    def fetch_existing_note(self, repo: str, pr_id: int, marker: str) -> dict | None:
        url = f"{self._mr_url(repo, pr_id)}/notes"
        resp = self._client.get(
            url, headers=self._headers,
            params={"per_page": 100, "order_by": "updated_at", "sort": "desc"},
        )
        resp.raise_for_status()
        for note in resp.json():
            body = note.get("body", "") or ""
            if marker in body:
                return {"note_id": note.get("id"), "body": body}
        return None

    @_retry
    def upsert_comment(
        self, repo: str, pr_id: int, body: str, note_id: int | None = None,
    ) -> dict:
        base = f"{self._mr_url(repo, pr_id)}/notes"
        if note_id:
            resp = self._client.put(
                f"{base}/{note_id}", headers=self._headers, json={"body": body},
            )
            if resp.status_code == 403:
                log.warning("PUT note %s forbidden — creating new note", note_id)
            else:
                resp.raise_for_status()
                return {"note_id": note_id, "action": "updated"}
        resp = self._client.post(base, headers=self._headers, json={"body": body})
        resp.raise_for_status()
        data = resp.json()
        return {"note_id": data.get("id"), "action": "created"}

    @_retry
    def list_discussions(self, repo: str, pr_id: int) -> list[dict]:
        url = f"{self._mr_url(repo, pr_id)}/discussions"
        results: list[dict] = []
        page = 1
        while True:
            resp = self._client.get(
                url, headers=self._headers,
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

    @_retry
    def resolve_discussion(self, repo: str, pr_id: int, discussion_id: str) -> None:
        url = (
            f"{self._mr_url(repo, pr_id)}/discussions/{discussion_id}"
            f"?resolved=true"
        )
        resp = self._client.put(url, headers=self._headers)
        resp.raise_for_status()

    @_retry
    def create_inline_discussion(
        self, repo: str, pr_id: int, body: str,
        base_sha: str, head_sha: str, start_sha: str,
        new_path: str, new_line: int, old_path: str | None = None,
    ) -> dict:
        url = f"{self._mr_url(repo, pr_id)}/discussions"
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
        resp = self._client.post(url, headers=self._headers, data=payload)
        resp.raise_for_status()
        return resp.json()
