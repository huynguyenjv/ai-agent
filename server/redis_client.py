"""Shared Redis connection — Phase 11.1.

Single lazily-connected Redis client used by the session store and rate limiter
for multi-instance deployments. Returns None (→ in-memory fallback) when
REDIS_URL is unset, the `redis` package is missing, or the server is
unreachable. Never raises on import/connect.
"""

from __future__ import annotations

import logging
import os
import threading

logger = logging.getLogger("server.redis")

_client = None
_checked = False
_lock = threading.Lock()


def get_redis():
    """Return a connected Redis client, or None if unavailable.

    Result is cached after the first call (including the None case).
    """
    global _client, _checked
    if _checked:
        return _client

    with _lock:
        if _checked:
            return _client
        _checked = True

        url = os.environ.get("REDIS_URL")
        if not url:
            logger.info("REDIS_URL not set — using in-memory session/rate-limit.")
            _client = None
            return None

        try:
            import redis  # imported lazily so the dep is optional

            client = redis.from_url(
                url,
                socket_connect_timeout=2,
                socket_timeout=2,
                decode_responses=True,
            )
            client.ping()
            _client = client
            logger.info("Redis connected: %s", url)
        except Exception as e:
            logger.warning("Redis unavailable (%s) — falling back to in-memory.", e)
            _client = None

        return _client


def redis_configured() -> bool:
    """True if REDIS_URL is set (regardless of reachability)."""
    return bool(os.environ.get("REDIS_URL"))


def reset_redis() -> None:
    """Reset the cached client (tests)."""
    global _client, _checked
    with _lock:
        _client = None
        _checked = False
