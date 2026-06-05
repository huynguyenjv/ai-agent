"""Phase 19 — batching (19.1), connection pool (19.2), speculative (19.3), streaming (19.4)."""

from __future__ import annotations

import asyncio

import pytest

from server.batch import AsyncBatcher
from server.connections import pool_limits, VLLM_MAX_CONNECTIONS
from server.speculative import prewarm_vllm, prewarm_enabled
from server.streaming.optimized import make_event_queue, TokenCoalescer


# --------------------------------------------------------------------------- #
# 19.1 Request batching
# --------------------------------------------------------------------------- #
class TestAsyncBatcher:
    async def test_concurrent_submits_share_batch(self):
        batches = []

        async def process(items):
            batches.append(list(items))
            return [x * 2 for x in items]

        b = AsyncBatcher(process, max_batch=3, max_wait=0.02)
        results = await asyncio.gather(*[b.submit(i) for i in range(3)])
        assert results == [0, 2, 4]
        assert len(batches) == 1  # all three coalesced

    async def test_max_wait_flushes_partial(self):
        async def process(items):
            return [x + 1 for x in items]

        b = AsyncBatcher(process, max_batch=10, max_wait=0.01)
        assert await b.submit(5) == 6  # flushed by timer, not full batch

    async def test_error_propagates(self):
        async def process(items):
            raise RuntimeError("boom")

        b = AsyncBatcher(process, max_batch=1, max_wait=0.01)
        with pytest.raises(RuntimeError):
            await b.submit(1)


# --------------------------------------------------------------------------- #
# 19.2 Connection pooling
# --------------------------------------------------------------------------- #
class TestConnectionPool:
    def test_pool_limits_from_config(self):
        limits = pool_limits()
        assert limits.max_connections == VLLM_MAX_CONNECTIONS


# --------------------------------------------------------------------------- #
# 19.3 Speculative pre-warm
# --------------------------------------------------------------------------- #
class TestSpeculative:
    async def test_prewarm_success(self):
        class FakeCompletions:
            async def create(self, **kw):
                return object()

        class FakeChat:
            completions = FakeCompletions()

        class FakeClient:
            chat = FakeChat()

        assert await prewarm_vllm(FakeClient(), "m") is True

    async def test_prewarm_failure_is_nonfatal(self):
        class FakeClient:
            class chat:
                class completions:
                    @staticmethod
                    async def create(**kw):
                        raise ConnectionError("down")

        assert await prewarm_vllm(FakeClient(), "m") is False

    def test_prewarm_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv("SPECULATIVE_PREWARM", raising=False)
        assert prewarm_enabled() is False


# --------------------------------------------------------------------------- #
# 19.4 Streaming optimization
# --------------------------------------------------------------------------- #
class TestStreamingOpt:
    def test_event_queue_is_bounded(self):
        q = make_event_queue(5)
        assert q.maxsize == 5

    def test_token_coalescer_flushes_on_threshold(self):
        c = TokenCoalescer(min_chars=5)
        assert c.add("ab") is None          # buffered
        out = c.add("cde")                  # reaches 5 chars → flush
        assert out == "abcde"

    def test_token_coalescer_manual_flush(self):
        c = TokenCoalescer(min_chars=100)
        c.add("hi")
        assert c.flush() == "hi"
        assert c.flush() is None            # empty after flush
