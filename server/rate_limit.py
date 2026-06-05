"""Simple token-bucket rate limiter.

Provides per-client rate limiting for API endpoints.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field

logger = logging.getLogger("server.rate_limit")

# Configuration from env
RATE_LIMIT_REQUESTS = int(os.environ.get("RATE_LIMIT_REQUESTS", "60"))
RATE_LIMIT_WINDOW = int(os.environ.get("RATE_LIMIT_WINDOW", "60"))  # seconds


@dataclass
class TokenBucket:
    """Token bucket for rate limiting."""
    tokens: float
    last_update: float
    max_tokens: float
    refill_rate: float  # tokens per second

    def consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens. Returns True if allowed."""
        now = time.time()
        elapsed = now - self.last_update

        # Refill tokens
        self.tokens = min(self.max_tokens, self.tokens + elapsed * self.refill_rate)
        self.last_update = now

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    def time_until_available(self, tokens: int = 1) -> float:
        """Seconds until tokens are available."""
        if self.tokens >= tokens:
            return 0
        needed = tokens - self.tokens
        return needed / self.refill_rate


class RateLimiter:
    """Per-client rate limiter using token bucket algorithm.

    Usage:
        limiter = get_rate_limiter()
        if not limiter.allow("client-123"):
            raise HTTPException(429, "Rate limit exceeded")
    """

    def __init__(
        self,
        max_requests: int = RATE_LIMIT_REQUESTS,
        window_seconds: int = RATE_LIMIT_WINDOW,
    ):
        self._buckets: dict[str, TokenBucket] = {}
        self._max_tokens = float(max_requests)
        self._refill_rate = max_requests / window_seconds
        self._lock = threading.Lock()
        self._max_clients = 10000

    def allow(self, client_id: str, tokens: int = 1) -> bool:
        """Check if request is allowed for client."""
        with self._lock:
            bucket = self._buckets.get(client_id)

            if bucket is None:
                # Cleanup old buckets if too many
                if len(self._buckets) >= self._max_clients:
                    self._cleanup_stale()

                bucket = TokenBucket(
                    tokens=self._max_tokens,
                    last_update=time.time(),
                    max_tokens=self._max_tokens,
                    refill_rate=self._refill_rate,
                )
                self._buckets[client_id] = bucket

            return bucket.consume(tokens)

    def retry_after(self, client_id: str, tokens: int = 1) -> float:
        """Get seconds until request would be allowed."""
        with self._lock:
            bucket = self._buckets.get(client_id)
            if bucket is None:
                return 0
            return bucket.time_until_available(tokens)

    def _cleanup_stale(self) -> None:
        """Remove stale buckets (not used in last hour)."""
        now = time.time()
        stale_threshold = 3600  # 1 hour

        stale = [
            k for k, v in self._buckets.items()
            if now - v.last_update > stale_threshold
        ]

        for k in stale:
            del self._buckets[k]

        if stale:
            logger.info("Cleaned up %d stale rate limit buckets", len(stale))

    def stats(self) -> dict:
        """Get rate limiter statistics."""
        with self._lock:
            return {
                "active_clients": len(self._buckets),
                "max_requests": self._max_tokens,
                "window_seconds": self._max_tokens / self._refill_rate,
            }


# =============================================================================
# Redis-backed limiter (Phase 11.1) — shared across instances
# =============================================================================

class RedisRateLimiter:
    """Fixed-window rate limiter in Redis, coordinated across instances.

    Falls back to a local in-memory RateLimiter on any Redis error so a Redis
    outage degrades (per-instance limiting) instead of failing requests.
    """

    def __init__(
        self,
        redis_client,
        max_requests: int = RATE_LIMIT_REQUESTS,
        window_seconds: int = RATE_LIMIT_WINDOW,
        prefix: str = "rl:",
    ):
        self._redis = redis_client
        self._max = max_requests
        self._window = window_seconds
        self._prefix = prefix
        self._fallback = RateLimiter(max_requests, window_seconds)

    def allow(self, client_id: str, tokens: int = 1) -> bool:
        key = f"{self._prefix}{client_id}"
        try:
            count = self._redis.incrby(key, tokens)
            if count == tokens:  # first hit in this window → set expiry
                self._redis.expire(key, self._window)
            return count <= self._max
        except Exception as e:
            logger.warning("Redis rate-limit error, using in-memory fallback: %s", e)
            return self._fallback.allow(client_id, tokens)

    def retry_after(self, client_id: str, tokens: int = 1) -> float:
        try:
            ttl = self._redis.ttl(f"{self._prefix}{client_id}")
            return float(ttl) if ttl and ttl > 0 else 0.0
        except Exception:
            return self._fallback.retry_after(client_id, tokens)

    def stats(self) -> dict:
        return {"backend": "redis", "max_requests": self._max, "window_seconds": self._window}


# =============================================================================
# Singleton
# =============================================================================

_rate_limiter: RateLimiter | RedisRateLimiter | None = None
_limiter_lock = threading.Lock()


def get_rate_limiter() -> RateLimiter | RedisRateLimiter:
    """Get singleton rate limiter — Redis-backed if REDIS_URL is reachable,
    otherwise in-memory (per-instance)."""
    global _rate_limiter

    if _rate_limiter is None:
        with _limiter_lock:
            if _rate_limiter is None:
                from server.redis_client import get_redis

                redis_client = get_redis()
                if redis_client is not None:
                    _rate_limiter = RedisRateLimiter(redis_client)
                    logger.info("RateLimiter: Redis-backed (%d / %ds)",
                                RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW)
                else:
                    _rate_limiter = RateLimiter()
                    logger.info("RateLimiter: in-memory (%d / %ds)",
                                RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW)

    return _rate_limiter


def reset_rate_limiter() -> None:
    """Reset the singleton (tests)."""
    global _rate_limiter
    with _limiter_lock:
        _rate_limiter = None
