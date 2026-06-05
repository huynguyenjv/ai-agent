"""Phase 11.1 — Redis-backed rate limiter + session store (with in-memory fallback)."""

from __future__ import annotations

from server.rate_limit import RedisRateLimiter, RateLimiter, get_rate_limiter, reset_rate_limiter
from server.session import RedisSessionStore
from server.redis_client import get_redis, reset_redis


class FakeRedis:
    """Minimal in-process stand-in for the redis client (decode_responses=True)."""

    def __init__(self):
        self.store: dict = {}
        self.expiries: dict = {}

    def incrby(self, key, amount=1):
        self.store[key] = int(self.store.get(key, 0)) + amount
        return self.store[key]

    def expire(self, key, seconds):
        self.expiries[key] = seconds
        return True

    def ttl(self, key):
        return self.expiries.get(key, -1)

    def get(self, key):
        return self.store.get(key)

    def setex(self, key, ttl, value):
        self.store[key] = value
        self.expiries[key] = ttl

    def delete(self, key):
        existed = key in self.store
        self.store.pop(key, None)
        return 1 if existed else 0

    def ping(self):
        return True


class BrokenRedis:
    """Every operation raises — exercises the in-memory fallback path."""

    def __getattr__(self, name):
        def boom(*a, **k):
            raise ConnectionError("redis down")
        return boom


class TestRedisRateLimiter:
    def test_allows_then_blocks(self):
        rl = RedisRateLimiter(FakeRedis(), max_requests=3, window_seconds=60)
        assert [rl.allow("client") for _ in range(3)] == [True, True, True]
        assert rl.allow("client") is False

    def test_separate_clients_independent(self):
        fake = FakeRedis()
        rl = RedisRateLimiter(fake, max_requests=1, window_seconds=60)
        assert rl.allow("a") is True
        assert rl.allow("b") is True   # different key
        assert rl.allow("a") is False

    def test_retry_after_reads_ttl(self):
        rl = RedisRateLimiter(FakeRedis(), max_requests=1, window_seconds=42)
        rl.allow("c")
        assert rl.retry_after("c") == 42.0

    def test_fallback_on_redis_error(self):
        rl = RedisRateLimiter(BrokenRedis(), max_requests=2, window_seconds=60)
        # Redis ops raise → in-memory fallback still allows within limit
        assert rl.allow("c") is True


class TestRedisSessionStore:
    def test_roundtrip(self):
        s = RedisSessionStore(FakeRedis())
        assert s.get("x") is None
        s.set("x", {"last_intent": "code_gen"})
        assert s.get("x") == {"last_intent": "code_gen"}
        assert s.delete("x") is True

    def test_fallback_on_redis_error(self):
        s = RedisSessionStore(BrokenRedis())
        s.set("x", {"a": 1})           # → in-memory fallback
        assert s.get("x") == {"a": 1}  # served from fallback


class TestFactories:
    def test_get_redis_none_without_url(self, monkeypatch):
        monkeypatch.delenv("REDIS_URL", raising=False)
        reset_redis()
        assert get_redis() is None

    def test_rate_limiter_in_memory_without_redis(self, monkeypatch):
        monkeypatch.delenv("REDIS_URL", raising=False)
        reset_redis()
        reset_rate_limiter()
        assert isinstance(get_rate_limiter(), RateLimiter)
        reset_rate_limiter()
