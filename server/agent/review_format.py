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

_EXT_LANG = {
    ".java": "java", ".py": "python", ".js": "javascript", ".ts": "typescript",
    ".go": "go", ".cs": "csharp", ".rb": "ruby", ".kt": "kotlin", ".rs": "rust",
    ".tf": "hcl", ".sql": "sql", ".xml": "xml", ".yml": "yaml", ".yaml": "yaml",
    ".json": "json", ".sh": "bash", ".jsx": "jsx", ".tsx": "tsx",
}


def _guess_lang(file_path: str) -> str:
    for ext, lang in _EXT_LANG.items():
        if file_path.endswith(ext):
            return lang
    return ""


_SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}
_SEVERITY_EMOJI = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🔵"}
_SEVERITY_TITLE = {"critical": "CRITICAL", "high": "HIGH", "medium": "MEDIUM", "low": "LOW"}


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
    """Collapsible <details> block per finding."""
    sev = f.get("severity", "low")
    emoji = _SEVERITY_EMOJI.get(sev, "🔵")
    sev_label = _SEVERITY_TITLE.get(sev, sev.upper())
    fw = (f.get("framework") or "—").strip()
    file = f.get("file", "?")
    line = f.get("line", 0)
    title = (f.get("title") or "").strip() or "Issue"
    msg = (f.get("message") or "").strip()
    sugg = (f.get("suggestion") or "").strip()

    summary = (
        f"{emoji} <b>{sev_label}</b> &nbsp;<code>{fw}</code>&nbsp; {title}"
    )

    body_lines = [f"**File:** `{file}` · line {line}", ""]
    if msg and msg != title:
        body_lines.append(msg)
        body_lines.append("")
    if sugg and sugg.lower() not in {"null", "none", ""}:
        body_lines.append("**💡 Suggested fix:**")
        body_lines.append("")
        lang = _guess_lang(file)
        if not sugg.startswith("```"):
            body_lines.append(f"```{lang}")
            body_lines.append(sugg)
            body_lines.append("```")
        else:
            body_lines.append(sugg)
        body_lines.append("")

    body = "\n".join(body_lines).rstrip()
    return (
        f"<details open>\n<summary>{summary}</summary>\n\n"
        f"{body}\n\n"
        f"</details>"
    )


def _render_findings_block(findings: list[dict]) -> str:
    if not findings:
        return "✅ **No issues found.**"
    by_sev: dict[str, list[dict]] = {"critical": [], "high": [], "medium": [], "low": []}
    for f in findings:
        by_sev.setdefault(f.get("severity", "low"), []).append(f)

    parts: list[str] = []
    for sev in ("critical", "high", "medium", "low"):
        items = by_sev.get(sev, [])
        if not items:
            continue
        parts.append(f"\n#### {_SEVERITY_EMOJI[sev]} {_SEVERITY_TITLE[sev]} ({len(items)})\n")
        parts.extend(_render_finding(f) for f in items)
    return "\n".join(parts)


_STATUS_ICON = {"fixed": "✅ fixed", "open": "🔴 open", "new": "🟠 new"}


def _render_previous_reviews_block(prev: list[dict]) -> str:
    """Render Previous reviews as a collapsible markdown table."""
    if not prev:
        return ""
    rows = prev[:KEEP_HISTORY]
    lines = [
        f"<details open>",
        f"<summary>📋 Previous reviews ({len(rows)})</summary>",
        "",
        "| Commit | When | Summary |",
        "|--------|------|---------|",
    ]
    for r in rows:
        sha = r.get("sha", "")
        ts = r.get("ts", "")
        summary = (r.get("summary", "") or "").replace("|", "\\|")
        lines.append(f"| `{sha}` | {ts} | {summary} |")
    lines.append("")
    lines.append("</details>")
    lines.append("")
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
