"""Tests for context building and caching modules."""

import pytest

from server.agent.context_builder import (
    estimate_tokens,
    build_optimal_context,
    get_context_summary,
)
from server.agent.summarize import (
    should_summarize,
    truncate_with_summary,
    _format_messages_for_summary,
    _fallback_summary,
)
from server.cache import (
    LRUCache,
    LLMResponseCache,
    RAGResultCache,
)


class TestEstimateTokens:
    """Tests for token estimation."""

    def test_empty_string(self):
        assert estimate_tokens("") == 0

    def test_short_string(self):
        # 12 chars / 4 = 3 tokens
        assert estimate_tokens("Hello World!") == 3

    def test_longer_string(self):
        text = "a" * 100
        assert estimate_tokens(text) == 25


class TestBuildOptimalContext:
    """Tests for context building."""

    def test_empty_state(self, tmp_path):
        state = {}
        result = build_optimal_context(state, str(tmp_path), token_budget=1000)
        assert result["context"] == ""
        assert result["tokens_used"] == 0

    def test_with_rag_chunks(self, tmp_path):
        state = {
            "rag_chunks": [
                {"body": "def foo(): pass", "file_path": "a.py", "start_line": 1, "end_line": 1, "score": 0.9},
                {"body": "def bar(): pass", "file_path": "b.py", "start_line": 1, "end_line": 1, "score": 0.5},
            ]
        }
        result = build_optimal_context(state, str(tmp_path), token_budget=1000)
        assert result["parts_included"] == 2
        assert "a.py" in result["context"]

    def test_budget_limit(self, tmp_path):
        # Create a large chunk
        state = {
            "rag_chunks": [
                {"body": "x" * 1000, "file_path": "large.py", "start_line": 1, "end_line": 1, "score": 0.9},
            ]
        }
        result = build_optimal_context(state, str(tmp_path), token_budget=100)
        # Should truncate
        assert result["tokens_used"] <= 100


class TestGetContextSummary:
    """Tests for context summary."""

    def test_empty_state(self):
        assert get_context_summary({}) == "No context"

    def test_with_active_file(self):
        state = {"active_file": "main.py"}
        assert "Active: main.py" in get_context_summary(state)

    def test_with_rag_chunks(self):
        state = {"rag_chunks": [{}, {}, {}]}
        assert "RAG: 3 chunks" in get_context_summary(state)


class TestShouldSummarize:
    """Tests for summarization trigger."""

    def test_few_messages(self):
        messages = [{"role": "user", "content": "hi"}] * 5
        assert should_summarize(messages) is False

    def test_many_messages(self):
        messages = [{"role": "user", "content": "hi"}] * 15
        assert should_summarize(messages) is True

    def test_high_tokens(self):
        messages = [{"role": "user", "content": "hi"}] * 3
        assert should_summarize(messages, token_estimate=7000) is True


class TestTruncateWithSummary:
    """Tests for message truncation."""

    def test_short_conversation(self):
        messages = [{"role": "user", "content": "hi"}]
        result = truncate_with_summary(messages, "summary", keep_recent=4)
        assert len(result) == 1  # No change

    def test_long_conversation(self):
        messages = [{"role": "user", "content": f"msg{i}"} for i in range(10)]
        result = truncate_with_summary(messages, "summary", keep_recent=3)
        assert len(result) == 4  # summary + 3 recent
        assert "summary" in result[0]["content"]


class TestFormatMessagesForSummary:
    """Tests for message formatting."""

    def test_format_messages(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        result = _format_messages_for_summary(messages)
        assert "USER: Hello" in result
        assert "ASSISTANT: Hi there" in result


class TestFallbackSummary:
    """Tests for fallback summary."""

    def test_basic_summary(self):
        messages = [
            {"role": "user", "content": "Write a function"},
            {"role": "assistant", "content": "Here it is"},
        ]
        result = _fallback_summary(messages)
        assert "1 user messages" in result
        assert "Write a function" in result


class TestLLMResponseCache:
    """Tests for LLM response caching."""

    def test_cache_miss(self):
        cache = LLMResponseCache(max_size=10)
        result = cache.get([{"role": "user", "content": "test"}], None, "code_gen")
        assert result is None

    def test_cache_hit(self):
        cache = LLMResponseCache(max_size=10)
        messages = [{"role": "user", "content": "test"}]
        response = {"draft": "hello", "tool_calls": []}

        cache.set(messages, None, "code_gen", response)
        result = cache.get(messages, None, "code_gen")

        assert result == response

    def test_different_intent_different_key(self):
        cache = LLMResponseCache(max_size=10)
        messages = [{"role": "user", "content": "test"}]

        cache.set(messages, None, "code_gen", {"result": "gen"})
        cache.set(messages, None, "explain", {"result": "explain"})

        assert cache.get(messages, None, "code_gen")["result"] == "gen"
        assert cache.get(messages, None, "explain")["result"] == "explain"


class TestRAGResultCache:
    """Tests for RAG result caching."""

    def test_cache_miss(self):
        cache = RAGResultCache(max_size=10)
        result = cache.get("hash123", lang_filter=None, top_k=5)
        assert result is None

    def test_cache_hit(self):
        cache = RAGResultCache(max_size=10)
        results = [{"file": "a.py", "score": 0.9}]

        cache.set("hash123", results, lang_filter="python", top_k=5)
        cached = cache.get("hash123", lang_filter="python", top_k=5)

        assert cached == results

    def test_invalidate(self):
        cache = RAGResultCache(max_size=10)
        cache.set("hash1", [{"file": "a.py"}])
        cache.set("hash2", [{"file": "b.py"}])

        cache.invalidate()

        assert cache.get("hash1") is None
        assert cache.get("hash2") is None

    def test_stats(self):
        cache = RAGResultCache(max_size=10)
        cache.set("hash1", [])
        cache.get("hash1")
        cache.get("hash2")  # miss

        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
