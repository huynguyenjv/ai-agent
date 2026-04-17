"""Node: upsert_mr_comment — Code Review Flow.

GET existing note with marker → PUT (update) or POST (create).
Parses previous_reviews from old body and re-renders markdown with merged history.
Calls GitLab API directly via import (not MCP stdio), since this runs inside the agent.
"""

from __future__ import annotations

import logging
import os

import httpx

from server.agent.state import AgentState
from server.agent.review_format import MARKER, review_format

logger = logging.getLogger("server.agent.upsert_mr_comment")

INLINE_MARKER = os.environ.get("AI_REVIEWER_INLINE_MARKER", "AI_REVIEW_INLINE:v1")
INLINE_SEVERITIES = {"critical", "high", "medium"}

_INLINE_EXT_LANG = {
    ".java": "java", ".py": "python", ".js": "javascript", ".ts": "typescript",
    ".go": "go", ".cs": "csharp", ".kt": "kotlin", ".rs": "rust",
}


def _guess_inline_lang(file_path: str) -> str:
    for ext, lang in _INLINE_EXT_LANG.items():
        if file_path.endswith(ext):
            return lang
    return ""
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
        file_path = f.get("file", "")
        lang = _guess_inline_lang(file_path)
        parts.append("")
        if sugg.startswith("```"):
            parts.append(f"💡 **Suggested fix:**\n\n{sugg}")
        else:
            parts.append(f"💡 **Suggested fix:**\n\n```{lang}\n{sugg}\n```")
    return "\n".join(parts)


async def _resolve_old_inline(provider: str, repo: str, pr_id: int) -> tuple[int, int]:
    """Return (resolved_count, failed_count)."""
    from server.gitlab import list_mr_discussions, resolve_mr_discussion
    try:
        discussions = await list_mr_discussions(provider=provider, repo=repo, pr_id=pr_id)
    except Exception as exc:
        logger.warning("list_mr_discussions failed: %s", exc)
        return 0, 0

    candidates = []
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
        if did:
            candidates.append(did)

    logger.info("inline resolve: %d AI threads need resolving", len(candidates))

    resolved = 0
    failed = 0
    for did in candidates:
        try:
            await resolve_mr_discussion(provider=provider, repo=repo, pr_id=pr_id, discussion_id=did)
            resolved += 1
        except httpx.HTTPStatusError as exc:
            failed += 1
            logger.warning(
                "resolve FAIL discussion=%s HTTP %d body=%s",
                did, exc.response.status_code, exc.response.text[:300],
            )
        except Exception as exc:
            failed += 1
            logger.warning("resolve FAIL discussion=%s %s", did, exc)
    return resolved, failed


async def _post_inline_findings(
    provider: str, repo: str, pr_id: int,
    findings: list[dict], pr_ctx: dict,
) -> tuple[int, int]:
    from server.gitlab import create_mr_discussion

    base_sha = pr_ctx.get("base_sha", "")
    head_sha = pr_ctx.get("head_sha", "")
    start_sha = pr_ctx.get("start_sha", "") or base_sha
    if not (base_sha and head_sha):
        logger.warning("inline skipped: missing diff_refs (base=%s head=%s)", base_sha, head_sha)
        return 0, 0

    added_lines_map: dict[str, list[int]] = pr_ctx.get("added_lines") or {}
    # Normalise map keys for fuzzy match (leading slash, trimming)
    norm_map = {k.strip().lstrip("/"): set(v or []) for k, v in added_lines_map.items()}

    posted = 0
    skipped = 0
    for f in findings:
        sev = f.get("severity", "low")
        if sev not in INLINE_SEVERITIES:
            continue
        path_raw = (f.get("file") or "").strip()
        path = path_raw.lstrip("/")
        try:
            line = int(f.get("line", 0) or 0)
        except (TypeError, ValueError):
            line = 0

        if line <= 0 or not path:
            logger.info("inline SKIP [%s] %s:%s — invalid line/path", sev, path_raw, line)
            skipped += 1
            continue

        allowed = norm_map.get(path)
        if allowed is None:
            # Try basename match as last resort (LLM sometimes strips dirs)
            matches = [k for k in norm_map if k.endswith("/" + path) or k == path]
            if len(matches) == 1:
                path = matches[0]
                allowed = norm_map[path]
            else:
                logger.info(
                    "inline SKIP [%s] %s:%d — path not in diff (known: %s)",
                    sev, path_raw, line, list(norm_map.keys())[:5],
                )
                skipped += 1
                continue

        if line not in allowed:
            logger.info(
                "inline SKIP [%s] %s:%d — line not a '+' line (allowed sample: %s)",
                sev, path, line, sorted(allowed)[:10],
            )
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
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "inline FAIL [%s] %s:%d — HTTP %d body=%s",
                sev, path, line, exc.response.status_code, exc.response.text[:400],
            )
            skipped += 1
        except Exception as exc:
            logger.warning("inline FAIL [%s] %s:%d — %s", sev, path, line, exc)
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
    from server.gitlab import get_mr_note, upsert_mr_comment as api_upsert

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
                resolved, resolve_failed = await _resolve_old_inline(
                    provider="gitlab", repo=repo, pr_id=pr_id,
                )
                findings = list(state.get("review_findings") or [])
                posted, skipped = await _post_inline_findings(
                    provider="gitlab", repo=repo, pr_id=pr_id,
                    findings=findings, pr_ctx=pr_ctx,
                )
                logger.info(
                    "inline discussions: resolved=%d resolve_failed=%d posted=%d skipped=%d",
                    resolved, resolve_failed, posted, skipped,
                )
                pr_ctx["inline_posted"] = posted
                pr_ctx["inline_resolved"] = resolved
                pr_ctx["inline_resolve_failed"] = resolve_failed
                pr_ctx["inline_skipped"] = skipped
            except Exception as exc:
                logger.warning("inline discussion flow failed: %s", exc)
            return {"pr_context": pr_ctx, "draft": body}
        except Exception as exc:
            last_exc = exc
            logger.warning("upsert_mr_comment attempt %d failed: %s", attempt + 1, exc)

    logger.error("upsert_mr_comment failed after retries: %s", last_exc)
    return {"draft": body, "pr_context": pr_ctx}
