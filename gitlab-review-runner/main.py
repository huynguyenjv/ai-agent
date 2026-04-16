"""Entrypoint: orchestrate GitLab ↔ AI server review flow for one MR."""

from __future__ import annotations

import logging
import sys

import httpx

from ai_client import AIClient
from config import load_config
from gitlab_client import GitLabClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("runner.main")


def _embed_marker(marker: str, body: str) -> str:
    tag = f"<!-- {marker} -->"
    if tag in body:
        return body
    return f"{tag}\n{body}"


def _resolve_old_inline_threads(gl: GitLabClient, cfg, marker: str) -> int:
    try:
        discussions = gl.list_discussions(cfg.repo, cfg.pr_id)
    except Exception as exc:
        log.warning("list_discussions failed: %s", exc)
        return 0

    resolved = 0
    for d in discussions:
        notes = d.get("notes") or []
        if not notes:
            continue
        first = notes[0]
        if first.get("resolved"):
            continue
        if marker not in (first.get("body") or ""):
            continue
        did = d.get("id")
        if not did:
            continue
        try:
            gl.resolve_discussion(cfg.repo, cfg.pr_id, did)
            resolved += 1
        except Exception as exc:
            log.warning("resolve discussion=%s failed: %s", did, exc)
    return resolved


def _post_inline_comments(gl: GitLabClient, cfg, diff_payload: dict, inline_comments: list[dict]) -> tuple[int, int]:
    base_sha = diff_payload.get("base_sha", "")
    head_sha = diff_payload.get("head_sha", "")
    start_sha = diff_payload.get("start_sha", "") or base_sha
    if not (base_sha and head_sha):
        log.warning("inline skipped: missing diff_refs")
        return 0, len(inline_comments)

    posted = 0
    failed = 0
    for ic in inline_comments:
        try:
            gl.create_inline_discussion(
                cfg.repo, cfg.pr_id,
                body=ic["body"],
                base_sha=base_sha, head_sha=head_sha, start_sha=start_sha,
                new_path=ic["new_path"],
                new_line=int(ic["new_line"]),
                old_path=ic.get("old_path"),
            )
            posted += 1
        except httpx.HTTPStatusError as exc:
            failed += 1
            log.warning(
                "inline FAIL %s:%s — HTTP %d body=%s",
                ic.get("new_path"), ic.get("new_line"),
                exc.response.status_code, exc.response.text[:300],
            )
        except Exception as exc:
            failed += 1
            log.warning("inline FAIL %s:%s — %s", ic.get("new_path"), ic.get("new_line"), exc)
    return posted, failed


def run() -> int:
    cfg = load_config()
    log.info("runner start: repo=%s mr=!%d ai=%s", cfg.repo, cfg.pr_id, cfg.ai_server_url)

    with GitLabClient(cfg.gitlab_url, cfg.gitlab_token, cfg.gitlab_verify, cfg.http_timeout) as gl:
        diff_payload = gl.fetch_diff(cfg.repo, cfg.pr_id)
        if not (diff_payload.get("diff") or "").strip():
            log.info("empty diff, nothing to review")
            return 0

        log.info(
            "fetched diff: %d files, commit=%s",
            len(diff_payload.get("files") or []),
            (diff_payload.get("commit_sha") or "")[:8],
        )

        ai = AIClient(cfg.ai_server_url, cfg.ai_api_key, timeout=cfg.http_timeout * 6)
        ai_payload = {
            "repo": cfg.repo,
            "pr_id": cfg.pr_id,
            **diff_payload,
        }
        result = ai.analyze(ai_payload)

        markdown = result.get("markdown") or ""
        inline_comments = result.get("inline_comments") or []
        log.info(
            "AI result: markdown=%d chars, inline=%d, counts=%s",
            len(markdown), len(inline_comments), result.get("findings_count"),
        )

        if not markdown.strip():
            log.warning("AI returned empty markdown — skipping summary post")
        else:
            existing = gl.fetch_existing_note(cfg.repo, cfg.pr_id, cfg.summary_marker)
            body = _embed_marker(cfg.summary_marker, markdown)
            res = gl.upsert_comment(
                cfg.repo, cfg.pr_id, body,
                note_id=existing["note_id"] if existing else None,
            )
            log.info("summary note %s (id=%s)", res.get("action"), res.get("note_id"))

        resolved = _resolve_old_inline_threads(gl, cfg, cfg.inline_marker)
        log.info("resolved %d old inline threads", resolved)

        posted, failed = _post_inline_comments(gl, cfg, diff_payload, inline_comments)
        log.info("inline discussions: posted=%d failed=%d", posted, failed)

    log.info("runner done")
    return 0


def main() -> int:
    try:
        return run()
    except Exception as exc:
        log.exception("runner failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
