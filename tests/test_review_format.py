"""Tests for review_format: marker, previous reviews parse, findings grouping."""

import os

os.environ.setdefault("AI_REVIEWER_MARKER", "AI_REVIEW_MARKER:v1")
os.environ.setdefault("AI_REVIEWER_KEEP_HISTORY", "3")

from server.agent.review_format import (
    _count, _parse_previous_reviews, _render_finding,
    _render_findings_block, _render_previous_reviews_block, review_format, MARKER,
)


def _f(**kw):
    base = {"severity": "high", "category": "bug", "framework": "CWE-476",
            "file": "a.py", "line": 10, "title": "null deref", "message": "x",
            "suggestion": ""}
    base.update(kw)
    return base


def test_count_buckets():
    fs = [_f(severity="critical"), _f(severity="critical"), _f(severity="low")]
    c = _count(fs)
    assert c == {"critical": 2, "high": 0, "medium": 0, "low": 1}


def test_render_finding_has_framework_tag():
    out = _render_finding(_f(framework="OWASP:A03", file="app.py", line=5, title="sql inj"))
    assert "[OWASP:A03]" in out
    assert "`app.py:5`" in out


def test_render_findings_block_groups_by_severity():
    fs = [_f(severity="critical"), _f(severity="medium")]
    out = _render_findings_block(fs)
    assert "Critical" in out and "Medium" in out
    assert out.index("Critical") < out.index("Medium")


def test_render_findings_block_empty():
    assert "No issues" in _render_findings_block([])


def test_parse_previous_reviews_roundtrip():
    body = (
        "<details><summary>Previous reviews (2)</summary>\n"
        "- `aaaaaaaa` (2026-04-14 10:00 UTC) — 1 critical · 0 high · 0 medium · 0 low\n"
        "- `bbbbbbbb` (2026-04-14 09:00 UTC) — 0 critical · 2 high · 0 medium · 0 low\n"
        "</details>"
    )
    prev = _parse_previous_reviews(body)
    assert len(prev) == 2
    assert prev[0]["sha"] == "aaaaaaaa"
    assert "critical" in prev[0]["summary"]


def test_parse_previous_reviews_no_block():
    assert _parse_previous_reviews("nothing here") == []


def test_render_previous_reviews_block_truncates():
    prev = [{"sha": f"sha{i}", "ts": "t", "summary": "s"} for i in range(5)]
    out = _render_previous_reviews_block(prev)
    # KEEP_HISTORY=3 → only 3 entries rendered
    assert out.count("- `sha") == 3


def test_review_format_output_contains_marker_and_template_bits():
    state = {
        "review_findings": [_f(severity="critical", framework="OWASP:A03")],
        "pr_context": {"commit_sha": "abcdef1234567890", "summary": "Test review"},
    }
    out = review_format(state)
    md = out["draft"]
    assert MARKER in md
    assert "OWASP:A03" in md
    assert "abcdef12" in md
    assert "Test review" in md
    # next history pushed
    assert out["pr_context"]["previous_reviews"][0]["sha"] == "abcdef12"
