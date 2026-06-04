"""Phase 11 — retry, circuit breaker, graceful degradation, deep health."""

from __future__ import annotations

import pytest

from server.retry import retry_async, compute_delay, is_retryable
from server.circuit_breaker import (
    CircuitBreaker,
    CircuitState,
    CircuitOpenError,
    get_circuit_breaker,
)
from server.agent.fallback import llm_unavailable_draft


# --------------------------------------------------------------------------- #
# 11.6 Retry
# --------------------------------------------------------------------------- #
class TestRetry:
    async def test_success_first_try(self):
        calls = []

        async def fn():
            calls.append(1)
            return "ok"

        assert await retry_async(fn, max_retries=3, base_delay=0) == "ok"
        assert len(calls) == 1

    async def test_retries_then_succeeds(self):
        calls = []

        async def fn():
            calls.append(1)
            if len(calls) < 2:
                raise ConnectionError("transient")
            return "ok"

        assert await retry_async(fn, max_retries=3, base_delay=0, jitter=False) == "ok"
        assert len(calls) == 2

    async def test_exhausts_and_raises(self):
        async def fn():
            raise TimeoutError("down")

        with pytest.raises(TimeoutError):
            await retry_async(fn, max_retries=2, base_delay=0)

    async def test_non_retryable_raises_immediately(self):
        calls = []

        async def fn():
            calls.append(1)
            raise ValueError("nope")

        with pytest.raises(ValueError):
            await retry_async(fn, max_retries=3, base_delay=0)
        assert len(calls) == 1

    def test_compute_delay_caps(self):
        assert compute_delay(10, base_delay=1.0, max_delay=5.0, jitter=False) == 5.0

    def test_is_retryable(self):
        assert is_retryable(ConnectionError())
        assert not is_retryable(ValueError())


# --------------------------------------------------------------------------- #
# 11.3 Circuit Breaker
# --------------------------------------------------------------------------- #
class TestCircuitBreaker:
    def test_opens_after_threshold(self):
        cb = CircuitBreaker("t", failure_threshold=3, recovery_timeout=60)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert cb.allow() is False

    def test_success_resets_failures(self):
        cb = CircuitBreaker("t", failure_threshold=3, recovery_timeout=60)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED  # never reached threshold

    def test_half_open_then_close(self):
        cb = CircuitBreaker("t", failure_threshold=1, recovery_timeout=0.0, half_open_max=1)
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        # recovery_timeout 0 → allow() probes (half-open)
        assert cb.allow() is True
        assert cb.state == CircuitState.HALF_OPEN
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_half_open_failure_reopens(self):
        cb = CircuitBreaker("t", failure_threshold=1, recovery_timeout=0.0)
        cb.record_failure()
        cb.allow()  # → half-open
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    async def test_call_raises_when_open(self):
        cb = CircuitBreaker("t", failure_threshold=1, recovery_timeout=60)

        async def boom():
            raise ConnectionError()

        with pytest.raises(ConnectionError):
            await cb.call(boom)
        assert cb.state == CircuitState.OPEN

        async def ok():
            return 1

        with pytest.raises(CircuitOpenError):
            await cb.call(ok)

    def test_registry_singleton(self):
        a = get_circuit_breaker("shared-name-xyz")
        b = get_circuit_breaker("shared-name-xyz")
        assert a is b


# --------------------------------------------------------------------------- #
# 11.4 Graceful degradation
# --------------------------------------------------------------------------- #
class TestFallback:
    def test_fallback_shape(self):
        d = llm_unavailable_draft("circuit open")
        assert d["pending_tool_calls"] == []
        assert d["degraded"] is True
        assert d["draft"]


# --------------------------------------------------------------------------- #
# 11.5 Deep health
# --------------------------------------------------------------------------- #
class TestHealthEndpoints:
    def _client(self, monkeypatch):
        monkeypatch.delenv("ENABLE_RAG", raising=False)
        monkeypatch.delenv("DATABASE_URL", raising=False)
        from fastapi.testclient import TestClient
        from server.app import create_app

        return TestClient(create_app())

    def test_liveness_ok(self, monkeypatch):
        with self._client(monkeypatch) as c:
            r = c.get("/health/live")
            assert r.status_code == 200
            assert r.json()["status"] == "ok"

    def test_deep_health_structure(self, monkeypatch):
        with self._client(monkeypatch) as c:
            body = c.get("/health/deep").json()
            assert set(body["checks"].keys()) == {"vllm", "qdrant", "postgres"}
            # RAG off / no DB → those checks are intentionally 'disabled'
            assert body["checks"]["qdrant"]["status"] == "disabled"
            assert body["checks"]["postgres"]["status"] == "disabled"
            assert "circuits" in body
