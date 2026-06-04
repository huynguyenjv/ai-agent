"""Tests for agentic-search additions and RAG opt-in.

Covers:
1. grep_content — full-text/regex content search (the new agentic primitive).
2. Tool parity — vtrip_grep is advertised and routed.
3. RAG is opt-in (ENABLE_RAG defaults to off).
"""

from __future__ import annotations

import inspect
import os

import mcp_server.server as mcp_srv
from mcp_server.tools import grep_content
from server.agent.generate import MCP_TOOL_NAMES, TOOL_NAME_MAP


class TestGrepContent:
    def _make_repo(self, tmp_path):
        (tmp_path / "a.py").write_text(
            "def handle_payment():\n    retry = True\n    return retry\n",
            encoding="utf-8",
        )
        (tmp_path / "b.go").write_text(
            "package main\nfunc Retry() {}\n",
            encoding="utf-8",
        )
        sub = tmp_path / "node_modules"
        sub.mkdir()
        (sub / "skip.py").write_text("def retry(): pass\n", encoding="utf-8")
        return tmp_path

    def test_basic_match(self, tmp_path):
        repo = self._make_repo(tmp_path)
        res = grep_content(str(repo), r"retry", ignore_case=True)
        files = {m["file_path"] for m in res["matches"]}
        assert "a.py" in files
        assert "b.go" in files

    def test_skips_skip_dirs(self, tmp_path):
        repo = self._make_repo(tmp_path)
        res = grep_content(str(repo), r"retry", ignore_case=True)
        files = {m["file_path"] for m in res["matches"]}
        assert not any("node_modules" in f for f in files)

    def test_path_glob_filter(self, tmp_path):
        repo = self._make_repo(tmp_path)
        res = grep_content(str(repo), r"Retry|retry", path_glob="*.go", ignore_case=False)
        files = {m["file_path"] for m in res["matches"]}
        assert files == {"b.go"}

    def test_case_sensitive(self, tmp_path):
        repo = self._make_repo(tmp_path)
        res = grep_content(str(repo), r"RETRY", ignore_case=False)
        assert res["total"] == 0

    def test_invalid_regex_returns_error(self, tmp_path):
        res = grep_content(str(tmp_path), r"(unclosed")
        assert "error" in res

    def test_line_numbers(self, tmp_path):
        repo = self._make_repo(tmp_path)
        res = grep_content(str(repo), r"return retry")
        match = next(m for m in res["matches"] if m["file_path"] == "a.py")
        assert match["line_number"] == 3


class TestGrepToolParity:
    def test_vtrip_grep_advertised(self):
        assert "vtrip_grep" in MCP_TOOL_NAMES

    def test_vtrip_grep_routed_in_mcp_server(self):
        src = inspect.getsource(mcp_srv.create_server)
        assert '== "vtrip_grep"' in src

    def test_grep_alias_maps_to_grep_tool(self):
        assert TOOL_NAME_MAP["grep"] == "vtrip_grep"
        assert TOOL_NAME_MAP["ripgrep"] == "vtrip_grep"


class TestRagOptIn:
    def test_rag_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv("ENABLE_RAG", raising=False)
        from server.routers.chat import _enable_rag

        assert _enable_rag() is False

    def test_rag_enabled_when_set(self, monkeypatch):
        monkeypatch.setenv("ENABLE_RAG", "true")
        from server.routers.chat import _enable_rag

        assert _enable_rag() is True


class TestRagInitGating:
    """When RAG is off, the app must not load Qdrant/embedder at startup."""

    def test_rag_off_skips_qdrant_and_embedder(self, monkeypatch):
        monkeypatch.delenv("ENABLE_RAG", raising=False)
        from fastapi.testclient import TestClient
        from server.app import create_app

        app = create_app()
        with TestClient(app):
            assert app.state.qdrant is None
            assert app.state.embedder is None
            # vLLM client is always initialized
            assert app.state.vllm_client is not None
