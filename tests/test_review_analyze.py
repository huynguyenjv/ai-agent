"""Tests for review_analyze: JSON parse, retry, diff split."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock

from server.agent.review_analyze import (
    _parse_json_object, _split_diff_by_file, _normalise_findings,
    _analyze_once, MAX_DIFF_CHARS_PER_FILE,
)


def test_parse_json_object_plain():
    assert _parse_json_object('{"a":1}') == {"a": 1}


def test_parse_json_object_fenced():
    assert _parse_json_object('```json\n{"a":1}\n```') == {"a": 1}


def test_parse_json_object_invalid():
    assert _parse_json_object("not json") is None


def test_parse_json_object_not_object():
    assert _parse_json_object("[1,2,3]") is None


def test_split_diff_by_file_basic():
    diff = (
        "--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-a\n+b\n"
        "--- a/bar.py\n+++ b/bar.py\n@@ -1 +1 @@\n-c\n+d\n"
    )
    chunks = _split_diff_by_file(diff)
    assert len(chunks) == 2
    paths = {c["path"] for c in chunks}
    assert paths == {"foo.py", "bar.py"}
    assert all(not c["skipped"] for c in chunks)


def test_split_diff_by_file_oversize_skipped():
    huge = "x" * (MAX_DIFF_CHARS_PER_FILE + 100)
    diff = f"--- a/big.py\n+++ b/big.py\n{huge}"
    chunks = _split_diff_by_file(diff)
    assert len(chunks) == 1
    assert chunks[0]["skipped"] is True
    assert chunks[0]["reason"] == "too_large"


def test_split_diff_empty():
    assert _split_diff_by_file("") == []


def test_normalise_findings_defaults_and_coercion():
    raw = [{"file": "a.py", "severity": "CRITICAL", "line": "5", "message": "x"}]
    out = _normalise_findings(raw)
    assert out[0]["severity"] == "critical"
    assert out[0]["line"] == 5
    assert out[0]["framework"] == ""


def test_normalise_findings_skips_non_dict():
    out = _normalise_findings([{"file": "a"}, "string", None, 42])
    assert len(out) == 1


@pytest.mark.asyncio
async def test_analyze_once_retry_on_bad_json(monkeypatch):
    calls = []

    async def fake_create(**kw):
        calls.append(kw["messages"][1]["content"])
        content = "garbage" if len(calls) == 1 else '{"summary":"ok","findings":[]}'
        msg = MagicMock()
        msg.content = content
        choice = MagicMock()
        choice.message = msg
        resp = MagicMock()
        resp.choices = [choice]
        return resp

    client = MagicMock()
    client.chat.completions.create = fake_create

    result = await _analyze_once(client, "m", "sys", "user")
    assert result == {"summary": "ok", "findings": []}
    assert len(calls) == 2
    assert "previous response was invalid JSON" in calls[1]


@pytest.mark.asyncio
async def test_analyze_once_returns_none_if_both_attempts_fail():
    async def fake_create(**kw):
        msg = MagicMock(); msg.content = "not json"
        choice = MagicMock(); choice.message = msg
        resp = MagicMock(); resp.choices = [choice]
        return resp

    client = MagicMock()
    client.chat.completions.create = fake_create

    result = await _analyze_once(client, "m", "sys", "user")
    assert result is None
