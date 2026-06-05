"""Phase 17 — tool result validation (17.4), analytics (17.5), atomic multi-file edits (17.1)."""

from __future__ import annotations

import inspect
import json

import mcp_server.server as mcp_srv
from server.agent.tool_validator import validate_tool_result, annotate_invalid
from server.metrics.tools import ToolAnalytics
from server.agent.generate import MCP_TOOL_NAMES
from mcp_server.tools_multifile import apply_multi_file_edits


# --------------------------------------------------------------------------- #
# 17.4 Tool result validation
# --------------------------------------------------------------------------- #
class TestToolValidator:
    def test_error_result_invalid(self):
        v = validate_tool_result("vtrip_read_file", json.dumps({"error": "File not found"}))
        assert not v.valid and v.category == "error"
        assert v.retry_suggestion

    def test_empty_result_invalid(self):
        v = validate_tool_result("vtrip_grep", "   ")
        assert not v.valid and v.category == "empty"

    def test_read_file_missing_content_malformed(self):
        v = validate_tool_result("vtrip_read_file", json.dumps({"file_path": "a.py"}))
        assert not v.valid and v.category == "malformed"

    def test_valid_result_ok(self):
        v = validate_tool_result("vtrip_read_file", json.dumps({"content": "x", "file_path": "a.py"}))
        assert v.valid

    def test_no_matches_is_valid(self):
        # empty search results are a legitimate answer, not an error
        v = validate_tool_result("vtrip_search_symbol", json.dumps({"results": []}))
        assert v.valid

    def test_annotate_invalid_prepends_note(self):
        v = validate_tool_result("vtrip_grep", "")
        out = annotate_invalid("payload", v)
        assert "tool_validation" in out


# --------------------------------------------------------------------------- #
# 17.5 Tool analytics
# --------------------------------------------------------------------------- #
class TestToolAnalytics:
    def test_success_failure_rates(self):
        a = ToolAnalytics()
        a.record("vtrip_read_file", success=True)
        a.record("vtrip_read_file", success=False)
        a.record("vtrip_grep", success=True, latency_ms=12.0)
        snap = a.snapshot()
        assert snap["total_calls"] == 3
        assert snap["tools"]["vtrip_read_file"]["success_rate"] == 0.5
        assert snap["tools"]["vtrip_grep"]["avg_latency_ms"] == 12.0
        assert snap["most_used"] == "vtrip_read_file"


# --------------------------------------------------------------------------- #
# 17.1 Atomic multi-file edits
# --------------------------------------------------------------------------- #
class TestAtomicMultiEdit:
    def test_apply_new_content(self, tmp_path):
        (tmp_path / "a.py").write_text("old", encoding="utf-8")
        res = apply_multi_file_edits(str(tmp_path), [
            {"file_path": "a.py", "new_content": "new"},
            {"file_path": "b.py", "new_content": "created"},
        ])
        assert res["success"] and res["applied"] == 2
        assert (tmp_path / "a.py").read_text(encoding="utf-8") == "new"
        assert (tmp_path / "b.py").read_text(encoding="utf-8") == "created"

    def test_search_replace(self, tmp_path):
        (tmp_path / "a.py").write_text("hello world", encoding="utf-8")
        res = apply_multi_file_edits(str(tmp_path), [
            {"file_path": "a.py", "search": "world", "replace": "there"},
        ])
        assert res["success"]
        assert (tmp_path / "a.py").read_text(encoding="utf-8") == "hello there"

    def test_conflict_aborts_without_writing(self, tmp_path):
        (tmp_path / "a.py").write_text("AAA", encoding="utf-8")
        (tmp_path / "b.py").write_text("BBB", encoding="utf-8")
        res = apply_multi_file_edits(str(tmp_path), [
            {"file_path": "a.py", "new_content": "changed"},
            {"file_path": "b.py", "search": "NOT_THERE", "replace": "x"},  # conflict
        ])
        assert not res["success"] and res.get("conflict")
        # first file must NOT have been written (pre-flight conflict detection)
        assert (tmp_path / "a.py").read_text(encoding="utf-8") == "AAA"

    def test_dry_run_produces_diff_no_write(self, tmp_path):
        (tmp_path / "a.py").write_text("old", encoding="utf-8")
        res = apply_multi_file_edits(str(tmp_path), [
            {"file_path": "a.py", "new_content": "new"},
        ], dry_run=True)
        assert res["success"] and res["dry_run"]
        assert res["diffs"][0]["diff"]
        assert (tmp_path / "a.py").read_text(encoding="utf-8") == "old"  # unchanged

    def test_path_outside_repo_rejected(self, tmp_path):
        res = apply_multi_file_edits(str(tmp_path), [
            {"file_path": "../escape.py", "new_content": "x"},
        ])
        assert not res["success"]


# --------------------------------------------------------------------------- #
# Tool parity for the new atomic edit tool
# --------------------------------------------------------------------------- #
class TestParity:
    def test_atomic_edit_advertised_and_routed(self):
        assert "vtrip_apply_edits_atomic" in MCP_TOOL_NAMES
        src = inspect.getsource(mcp_srv.create_server)
        assert '== "vtrip_apply_edits_atomic"' in src
