"""
Redis Cache Service — centralized caching layer.

Provides a unified Redis caching interface for:
  - RAG context caching
  - Session data
  - Generated test code
  - General key-value caching

Falls back to in-memory LRU cache if Redis is unavailable.
"""

from __future__ import annotations

import json
import time
import threading
from collections import OrderedDict
from typing import Optional, Any

import structlog

logger = structlog.get_logger()

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None  # type: ignore[assignment]


class LRUCache:
    """Thread-safe in-memory LRU cache with TTL support."""

    def __init__(self, max_size: int = 1000):
        self._cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()  # key -> (value, expiry)
        self._lock = threading.Lock()
        self._max_size = max_size

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key in self._cache:
                value, expiry = self._cache[key]
                if expiry == 0 or time.time() < expiry:
                    # Move to end (most recently used)
                    self._cache.move_to_end(key)
                    return value
                else:
                    del self._cache[key]
            return None

    def set(self, key: str, value: Any, ttl: int = 0) -> bool:
        with self._lock:
            expiry = time.time() + ttl if ttl > 0 else 0
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = (value, expiry)
            # Evict oldest if over max size
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)
            return True

    def delete(self, key: str) -> bool:
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def exists(self, key: str) -> bool:
        with self._lock:
            if key in self._cache:
                _, expiry = self._cache[key]
                if expiry == 0 or time.time() < expiry:
                    return True
                del self._cache[key]
            return False

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def cleanup_expired(self) -> int:
        """Remove expired entries. Returns count removed."""
        now = time.time()
        with self._lock:
            expired = [k for k, (_, exp) in self._cache.items() if exp > 0 and exp < now]
            for k in expired:
                del self._cache[k]
            return len(expired)

    def size(self) -> int:
        with self._lock:
            return len(self._cache)


class RedisCacheService:
    """Centralized Redis cache service with in-memory fallback.

    Usage::

        cache = RedisCacheService.from_env()
        cache.set("key", {"data": "value"}, ttl=3600)
        result = cache.get("key")
    """

    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_password: Optional[str] = None,
        redis_db: int = 0,
        prefix: str = "ai_agent:",
        fallback_max_size: int = 1000,
    ):
        self._prefix = prefix
        self._redis: Optional[Any] = None  # redis.Redis or None
        self._fallback = LRUCache(max_size=fallback_max_size)
        self._using_redis = False

        if REDIS_AVAILABLE and redis:
            try:
                self._redis = redis.Redis(
                    host=redis_host,
                    port=redis_port,
                    password=redis_password,
                    db=redis_db,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                    retry_on_timeout=True,
                )
                self._redis.ping()
                self._using_redis = True
                logger.info(
                    "Redis cache service initialized",
                    host=redis_host,
                    port=redis_port,
                    db=redis_db,
                )
            except Exception as e:
                logger.warning(
                    "Redis unavailable, using in-memory LRU cache fallback",
                    error=str(e),
                )
                self._redis = None
        else:
            logger.info("Redis package not installed, using in-memory LRU cache")

    @classmethod
    def from_env(cls) -> "RedisCacheService":
        """Create instance from environment variables."""
        import os
        return cls(
            redis_host=os.getenv("REDIS_HOST", "localhost"),
            redis_port=int(os.getenv("REDIS_PORT", "6379")),
            redis_password=os.getenv("REDIS_PASSWORD") or None,
            redis_db=int(os.getenv("REDIS_DB", "0")),
        )

    @property
    def is_redis_connected(self) -> bool:
        """Check if Redis is the active backend."""
        return self._using_redis

    @property
    def backend_name(self) -> str:
        return "redis" if self._using_redis else "memory"

    # ── Core operations ──────────────────────────────────────────────

    def _key(self, key: str) -> str:
        return f"{self._prefix}{key}"

    def get(self, key: str) -> Optional[Any]:
        """Get value by key. Returns None if not found or expired."""
        if self._using_redis:
            return self._redis_get(key)
        return self._fallback.get(key)

    def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set value with TTL (seconds). Default 1 hour."""
        if self._using_redis:
            return self._redis_set(key, value, ttl)
        return self._fallback.set(key, value, ttl)

    def delete(self, key: str) -> bool:
        """Delete a key."""
        if self._using_redis:
            return self._redis_delete(key)
        return self._fallback.delete(key)

    def exists(self, key: str) -> bool:
        """Check if key exists and is not expired."""
        if self._using_redis:
            return self._redis_exists(key)
        return self._fallback.exists(key)

    # ── RAG Context Caching ──────────────────────────────────────────

    def cache_rag_context(self, cache_key: str, chunks: list[dict], ttl: int = 3600) -> bool:
        """Cache RAG search results."""
        return self.set(f"rag:{cache_key}", chunks, ttl)

    def get_rag_context(self, cache_key: str) -> Optional[list[dict]]:
        """Get cached RAG search results."""
        return self.get(f"rag:{cache_key}")

    # ── Generated Test Caching ───────────────────────────────────────

    def cache_generated_test(self, session_id: str, class_name: str, test_code: str, ttl: int = 7200) -> bool:
        """Cache generated test code for refinement. Default 2 hours."""
        return self.set(f"test:{session_id}:{class_name}", test_code, ttl)

    def get_generated_test(self, session_id: str, class_name: str) -> Optional[str]:
        """Get cached generated test code."""
        return self.get(f"test:{session_id}:{class_name}")

    # ── Session Caching ──────────────────────────────────────────────

    def cache_session(self, session_id: str, data: dict, ttl: int = 3600) -> bool:
        """Cache session data."""
        return self.set(f"session:{session_id}", data, ttl)

    def get_session(self, session_id: str) -> Optional[dict]:
        """Get cached session data."""
        return self.get(f"session:{session_id}")

    def delete_session(self, session_id: str) -> bool:
        """Delete cached session data."""
        return self.delete(f"session:{session_id}")

    # ── Stats ────────────────────────────────────────────────────────

    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        if self._using_redis:
            try:
                info = self._redis.info("memory")
                return {
                    "backend": "redis",
                    "connected": True,
                    "used_memory_human": info.get("used_memory_human", "unknown"),
                    "used_memory_bytes": info.get("used_memory", 0),
                    "total_keys": self._redis.dbsize(),
                }
            except Exception as e:
                logger.error("Failed to get Redis stats", error=str(e))
                return {"backend": "redis", "connected": False, "error": str(e)}
        else:
            return {
                "backend": "memory",
                "connected": True,
                "cached_items": self._fallback.size(),
                "max_size": self._fallback._max_size,
            }

    # ── Redis internals ──────────────────────────────────────────────

    def _redis_get(self, key: str) -> Optional[Any]:
        try:
            data = self._redis.get(self._key(key))
            if data:
                return json.loads(data)
            return None
        except json.JSONDecodeError:
            # Return as plain string if not valid JSON
            return data
        except Exception as e:
            logger.error("Redis GET failed, trying fallback", key=key, error=str(e))
            return self._fallback.get(key)

    def _redis_set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        try:
            if isinstance(value, (dict, list)):
                data = json.dumps(value)
            elif isinstance(value, str):
                data = value
            else:
                data = json.dumps(value)
            
            self._redis.setex(self._key(key), ttl, data)
            return True
        except Exception as e:
            logger.error("Redis SET failed, using fallback", key=key, error=str(e))
            return self._fallback.set(key, value, ttl)

    def _redis_delete(self, key: str) -> bool:
        try:
            return self._redis.delete(self._key(key)) > 0
        except Exception as e:
            logger.error("Redis DELETE failed", key=key, error=str(e))
            return self._fallback.delete(key)

    def _redis_exists(self, key: str) -> bool:
        try:
            return self._redis.exists(self._key(key)) > 0
        except Exception as e:
            logger.error("Redis EXISTS failed", key=key, error=str(e))
            return self._fallback.exists(key)


# ── Module-level singleton ───────────────────────────────────────────────

_cache_service: Optional[RedisCacheService] = None
_cache_lock = threading.Lock()


def get_cache_service() -> RedisCacheService:
    """Get or create the global cache service singleton."""
    global _cache_service
    if _cache_service is None:
        with _cache_lock:
            if _cache_service is None:
                _cache_service = RedisCacheService.from_env()
    return _cache_service


def reset_cache_service() -> None:
    """Reset the global cache service (for testing)."""
    global _cache_service
    _cache_service = None
