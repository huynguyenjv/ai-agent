"""Node: review_format — Code Review Flow.

Render review_findings into markdown. Handles previous reviews history.
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timezone

from server.agent.state import AgentState

logger = logging.getLogger("server.agent.review_format")

MARKER = os.environ.get("AI_REVIEWER_MARKER", "AI_REVIEW_MARKER:v1")
KEEP_HISTORY = int(os.environ.get("AI_REVIEWER_KEEP_HISTORY", "3"))

_SEVERITY_ORDER = {"blocker": 0, "major": 1, "minor": 2, "info": 3}
_SEVERITY_EMOJI = {"blocker": "🔴", "major": "🟠", "minor": "🟡", "info": "ℹ️"}


def _count(findings: list[dict]) -> dict[str, int]:
    counts = {"blocker": 0, "major": 0, "minor": 0, "info": 0}
    for f in findings:
        sev = f.get("severity", "info")
        if sev in counts:
            counts[sev] += 1
    return counts


def _parse_previous_reviews(body: str) -> list[dict]:
    """Extract previous_reviews from existing note <details> block."""
    if not body:
        return []
    m = re.search(
        r"<details><summary>Previous reviews[^<]*</summary>\s*\n(.*?)</details>",
        body,
        re.DOTALL,
    )
    if not m:
        return []
    reviews = []
    for line in m.group(1).splitlines():
        entry = re.match(r"\s*-\s*`([^`]+)`\s*\(([^)]+)\)\s*—\s*(.+?)\s*$", line)
        if entry:
            reviews.append({
                "sha": entry.group(1),
                "ts": entry.group(2),
                "summary": entry.group(3),
            })
    return reviews


def _summary_line(counts: dict[str, int]) -> str:
    return (
        f"{counts['blocker']} blockers · {counts['major']} majors · "
        f"{counts['minor']} minors · {counts['info']} info"
    )


def _render_section(title: str, emoji: str, findings: list[dict]) -> str:
    if not findings:
        return ""
    lines = [f"### {emoji} {title}", ""]
    for f in findings:
        file_line = f"`{f.get('file', '?')}:{f.get('line', 0)}`" if f.get("file") else ""
        lines.append(f"- **{file_line}** — {f.get('title', '')}")
        desc = f.get("description", "").strip()
        if desc:
            lines.append(f"  {desc}")
        sugg = f.get("suggestion", "").strip()
        if sugg:
            lines.append("  <details><summary>Suggestion</summary>\n")
            lines.append(f"  ```\n  {sugg}\n  ```\n")
            lines.append("  </details>")
        lines.append("")
    return "\n".join(lines)


def review_format(state: AgentState) -> dict:
    findings = list(state.get("review_findings") or [])
    pr_ctx = dict(state.get("pr_context") or {})
    commit_sha = pr_ctx.get("commit_sha", "")
    commit_short = commit_sha[:8] if commit_sha else "n/a"

    findings.sort(key=lambda f: (_SEVERITY_ORDER.get(f.get("severity", "info"), 99), f.get("file", "")))
    counts = _count(findings)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Merge previous history from existing note body (if any)
    prev_reviews: list[dict] = list(pr_ctx.get("previous_reviews") or [])
    # Add current to head for next round (but not into current render's history)
    next_history = [{"sha": commit_short, "ts": ts, "summary": _summary_line(counts)}] + prev_reviews
    next_history = next_history[:KEEP_HISTORY + 1]  # +1 because current counts as 1

    parts = [
        f"<!-- {MARKER} -->",
        "## 🤖 AI Code Review",
        "",
        f"**Commit**: `{commit_short}` · **Updated**: {ts}",
        f"**Summary**: {_summary_line(counts)}",
        "",
    ]

    if prev_reviews:
        parts.append(f"<details><summary>Previous reviews ({len(prev_reviews)})</summary>\n")
        for r in prev_reviews[:KEEP_HISTORY]:
            parts.append(f"- `{r['sha']}` ({r['ts']}) — {r['summary']}")
        parts.append("\n</details>")
        parts.append("")

    if not findings:
        parts.append("✅ **No issues found.**")
    else:
        by_sev = {"blocker": [], "major": [], "minor": [], "info": []}
        for f in findings:
            by_sev.setdefault(f.get("severity", "info"), []).append(f)
        for sev, title in [("blocker", "Blockers"), ("major", "Majors"), ("minor", "Minors"), ("info", "Info")]:
            section = _render_section(title, _SEVERITY_EMOJI[sev], by_sev.get(sev, []))
            if section:
                parts.append(section)

    markdown = "\n".join(parts).rstrip() + "\n"
    pr_ctx["previous_reviews"] = next_history
    return {"draft": markdown, "pr_context": pr_ctx or None}
