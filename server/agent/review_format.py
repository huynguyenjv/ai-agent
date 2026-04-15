"""Node: review_format — render findings into markdown using prompt template."""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timezone

from server.agent.prompts import load_prompt
from server.agent.state import AgentState

logger = logging.getLogger("server.agent.review_format")

MARKER = os.environ.get("AI_REVIEWER_MARKER", "AI_REVIEW_MARKER:v1")
KEEP_HISTORY = int(os.environ.get("AI_REVIEWER_KEEP_HISTORY", "3"))

_SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}
_SEVERITY_EMOJI = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🔵"}
_SEVERITY_TITLE = {"critical": "Critical", "high": "High", "medium": "Medium", "low": "Low"}


def _count(findings: list[dict]) -> dict[str, int]:
    counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    for f in findings:
        sev = f.get("severity", "low")
        if sev in counts:
            counts[sev] += 1
    return counts


def _parse_previous_reviews(body: str) -> list[dict]:
    if not body:
        return []
    m = re.search(
        r"<details><summary>Previous reviews[^<]*</summary>\s*\n(.*?)</details>",
        body, re.DOTALL,
    )
    if not m:
        return []
    reviews = []
    for line in m.group(1).splitlines():
        entry = re.match(r"\s*-\s*`([^`]+)`\s*\(([^)]+)\)\s*—\s*(.+?)\s*$", line)
        if entry:
            reviews.append({"sha": entry.group(1), "ts": entry.group(2), "summary": entry.group(3)})
    return reviews


def _summary_stats(counts: dict[str, int]) -> str:
    return (
        f"{counts['critical']} critical · {counts['high']} high · "
        f"{counts['medium']} medium · {counts['low']} low"
    )


def _render_finding(f: dict) -> str:
    fw = f.get("framework", "") or "—"
    file = f.get("file", "?")
    line = f.get("line", 0)
    title = (f.get("title") or "").strip()
    msg = (f.get("message") or "").strip()
    sugg = (f.get("suggestion") or "").strip()

    # Pick the shorter non-empty one as the headline, skip duplication
    headline = title or msg[:120]
    body = "" if not msg or msg == title or msg.startswith(title) else msg

    out = [f"- **[{fw}]** `{file}:{line}` — {headline}"]
    if body:
        out.append(f"  {body}")
    if sugg and sugg.lower() not in {"null", "none", ""}:
        out.append(f"  💡 {sugg}")
    return "\n".join(out)


def _render_findings_block(findings: list[dict]) -> str:
    if not findings:
        return "✅ **No issues found.**"
    by_sev: dict[str, list[dict]] = {"critical": [], "high": [], "medium": [], "low": []}
    for f in findings:
        by_sev.setdefault(f.get("severity", "low"), []).append(f)

    parts = []
    for sev in ("critical", "high", "medium", "low"):
        items = by_sev.get(sev, [])
        if not items:
            continue
        parts.append(f"\n### {_SEVERITY_EMOJI[sev]} {_SEVERITY_TITLE[sev]}\n")
        parts.extend(_render_finding(f) for f in items)
    return "\n".join(parts)


def _render_previous_reviews_block(prev: list[dict]) -> str:
    if not prev:
        return ""
    lines = [f"<details><summary>Previous reviews ({len(prev)})</summary>\n"]
    for r in prev[:KEEP_HISTORY]:
        lines.append(f"- `{r['sha']}` ({r['ts']}) — {r['summary']}")
    lines.append("\n</details>\n")
    return "\n".join(lines)


def review_format(state: AgentState) -> dict:
    findings = list(state.get("review_findings") or [])
    pr_ctx = dict(state.get("pr_context") or {})

    findings.sort(key=lambda f: (_SEVERITY_ORDER.get(f.get("severity", "low"), 99), f.get("file", "")))
    counts = _count(findings)

    commit_sha = pr_ctx.get("commit_sha", "") or ""
    commit_short = commit_sha[:8] if commit_sha else "n/a"
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    summary = pr_ctx.get("summary") or _summary_stats(counts)

    prev_reviews: list[dict] = list(pr_ctx.get("previous_reviews") or [])
    next_history = [{"sha": commit_short, "ts": ts, "summary": _summary_stats(counts)}] + prev_reviews
    next_history = next_history[: KEEP_HISTORY + 1]

    template = load_prompt("review_output_template")
    markdown = template.format(
        marker=MARKER,
        commit_short=commit_short,
        timestamp=ts,
        summary=summary,
        critical_count=counts["critical"],
        high_count=counts["high"],
        medium_count=counts["medium"],
        low_count=counts["low"],
        previous_reviews_block=_render_previous_reviews_block(prev_reviews),
        findings_block=_render_findings_block(findings),
    )
    # Collapse multiple blank lines
    markdown = re.sub(r"\n{3,}", "\n\n", markdown).rstrip() + "\n"

    pr_ctx["previous_reviews"] = next_history
    return {"draft": markdown, "pr_context": pr_ctx or None}
