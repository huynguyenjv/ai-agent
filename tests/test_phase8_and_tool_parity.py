"""Regression tests for Phase 1-9 completion work.

Covers two gaps that were fixed:
1. Tool parity — every vtrip_* tool advertised to the model in
   `generate.MCP_TOOLS` must be routed by the client-side MCP server.
2. Phase 8 caching/context wiring — rag_search must serve repeat queries
   from the RAG result cache, and build_optimal_context must assemble
   retrieved chunks into the generation prompt.
"""

from __future__ import annotations

import inspect

import mcp_server.server as mcp_srv
from server.agent.generate import MCP_TOOLS, MCP_TOOL_NAMES, _to_openai_messages
from server.agent import rag_search as rs
from server.cache import get_rag_cache


class TestToolParity:
    """generate.MCP_TOOLS (advertised) must match MCP server routing."""

    def test_every_vtrip_tool_is_routed_in_mcp_server(self):
        src = inspect.getsource(mcp_srv.create_server)
        advertised = [
            t["function"]["name"]
            for t in MCP_TOOLS
            if t["function"]["name"].startswith("vtrip_")
        ]
        missing = [name for name in advertised if f'== "{name}"' not in src]
        assert not missing, f"MCP server has no routing branch for: {missing}"

    def test_phase7_tools_are_advertised(self):
        for name in (
            "vtrip_run_tests",
            "vtrip_lint_code",
            "vtrip_rename_symbol",
            "vtrip_extract_function",
            "vtrip_inline_variable",
        ):
            assert name in MCP_TOOL_NAMES

    def test_phase1_2_tools_are_routed(self):
        src = inspect.getsource(mcp_srv.create_server)
        for name in (
            "vtrip_run_command",
            "vtrip_diff_preview",
            "vtrip_apply_edits",
            "vtrip_git_status",
            "vtrip_git_commit",
        ):
            assert f'== "{name}"' in src


class _FakeEmbedder:
    def embed_both(self, query):
        return ([0.1, 0.2, 0.3], {0: 1.0})


class _CountingQdrant:
    def __init__(self):
        self.calls = 0

    async def hybrid_search(self, **kwargs):
        self.calls += 1
        return [
            {
                "body": "def foo(): pass",
                "file_path": "a.py",
                "start_line": 1,
                "end_line": 1,
                "score": 0.9,
            }
        ]


class TestRagSearchCache:
    """Phase 8.5 — rag_search serves repeat queries from cache."""

    async def test_repeat_query_served_from_cache(self):
        get_rag_cache().invalidate()
        qdrant = _CountingQdrant()
        embedder = _FakeEmbedder()
        state = {"messages": [{"role": "user", "content": "very-unique-query-abc987"}]}

        first = await rs.rag_search(state, qdrant, embedder)
        second = await rs.rag_search(state, qdrant, embedder)

        assert first["rag_hit"] is True
        assert second["rag_hit"] is True
        assert qdrant.calls == 1, "second identical query must hit the cache"

    async def test_freshness_bypasses_cache(self):
        get_rag_cache().invalidate()
        qdrant = _CountingQdrant()
        embedder = _FakeEmbedder()
        state = {
            "messages": [{"role": "user", "content": "fresh-query-xyz"}],
            "freshness_signal": True,
        }

        await rs.rag_search(state, qdrant, embedder)
        await rs.rag_search(state, qdrant, embedder)

        assert qdrant.calls == 2, "freshness must bypass the cache"


class TestRagContextInjection:
    """Phase 8.1 — retrieved chunks reach the generation prompt."""

    def test_rag_chunks_injected_into_system_prompt(self):
        state = {
            "intent": "code_gen",
            "messages": [],
            "rag_chunks": [
                {
                    "body": "class PaymentService: ...",
                    "file_path": "payment.py",
                    "start_line": 1,
                    "end_line": 5,
                    "score": 0.95,
                }
            ],
        }
        messages = _to_openai_messages(state)
        system = messages[0]["content"]
        assert "Relevant Code Context" in system
        assert "payment.py" in system

    def test_no_context_section_without_chunks(self):
        state = {"intent": "code_gen", "messages": []}
        messages = _to_openai_messages(state)
        assert "Relevant Code Context" not in messages[0]["content"]
