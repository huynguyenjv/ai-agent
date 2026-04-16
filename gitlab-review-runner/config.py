"""Load runtime config from env vars (GitLab CI auto-injected + user secrets)."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _require(name: str) -> str:
    v = os.environ.get(name, "").strip()
    if not v:
        raise RuntimeError(f"Env var {name} is required")
    return v


def _resolve_verify(raw: str) -> bool | str:
    raw = (raw or "").strip()
    if not raw:
        return True
    if raw.lower() in {"0", "false", "no"}:
        return False
    return raw


@dataclass
class Config:
    repo: str
    pr_id: int
    gitlab_url: str
    gitlab_token: str
    gitlab_verify: bool | str
    ai_server_url: str
    ai_api_key: str
    http_timeout: float
    summary_marker: str
    inline_marker: str


def load_config() -> Config:
    repo = _require("CI_PROJECT_PATH")
    try:
        pr_id = int(_require("CI_MERGE_REQUEST_IID"))
    except ValueError as exc:
        raise RuntimeError(f"CI_MERGE_REQUEST_IID invalid: {exc}")

    gitlab_url = (
        os.environ.get("GITLAB_URL")
        or os.environ.get("CI_SERVER_URL")
        or "https://gitlab.com"
    ).rstrip("/")

    return Config(
        repo=repo,
        pr_id=pr_id,
        gitlab_url=gitlab_url,
        gitlab_token=_require("GITLAB_TOKEN"),
        gitlab_verify=_resolve_verify(os.environ.get("GITLAB_CA_BUNDLE", "")),
        ai_server_url=_require("AI_SERVER_URL").rstrip("/"),
        ai_api_key=_require("AI_SERVER_API_KEY"),
        http_timeout=float(os.environ.get("HTTP_TIMEOUT", "60")),
        summary_marker=os.environ.get("AI_REVIEWER_MARKER", "AI_REVIEW_MARKER:v1"),
        inline_marker=os.environ.get("AI_REVIEWER_INLINE_MARKER", "AI_REVIEW_INLINE:v1"),
    )
