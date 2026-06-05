"""Simple in-memory session store for multi-turn conversations.

Provides TTL-based session storage for conversation context.
For production, consider Redis-backed implementation.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

logger = logging.getLogger("server.session")


@dataclass
class SessionData:
    """Data stored in a session."""

    conversation_id: str
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    data: dict[str, Any] = field(default_factory=dict)

    def touch(self) -> None:
        """Update last_accessed timestamp."""
        self.last_accessed = datetime.now()


class SessionStore:
    """Thread-safe in-memory session store with TTL.

    Usage:
        store = get_session_store()
        store.set("session-123", {"context": "...", "history": [...]})
        data = store.get("session-123")
    """

    def __init__(self, ttl_minutes: int = 30, max_sessions: int = 1000):
        """Initialize the session store.

        Args:
            ttl_minutes: Session TTL in minutes (default 30)
            max_sessions: Max sessions before cleanup (default 1000)
        """
        self._store: dict[str, SessionData] = {}
        self._ttl = timedelta(minutes=ttl_minutes)
        self._max_sessions = max_sessions
        self._lock = threading.Lock()

    def get(self, session_id: str) -> dict[str, Any] | None:
        """Get session data if it exists and hasn't expired.

        Args:
            session_id: Unique session identifier

        Returns:
            Session data dict or None if not found/expired
        """
        with self._lock:
            entry = self._store.get(session_id)
            if entry is None:
                return None

            if datetime.now() - entry.last_accessed > self._ttl:
                del self._store[session_id]
                logger.debug("Session %s expired", session_id[:8])
                return None

            entry.touch()
            return entry.data

    def set(self, session_id: str, data: dict[str, Any]) -> None:
        """Store session data.

        Args:
            session_id: Unique session identifier
            data: Data to store
        """
        with self._lock:
            if len(self._store) >= self._max_sessions:
                self._cleanup_oldest()

            if session_id in self._store:
                self._store[session_id].data = data
                self._store[session_id].touch()
            else:
                self._store[session_id] = SessionData(
                    conversation_id=session_id,
                    data=data,
                )
            logger.debug("Session %s updated", session_id[:8])

    def delete(self, session_id: str) -> bool:
        """Delete a session.

        Args:
            session_id: Session to delete

        Returns:
            True if session existed and was deleted
        """
        with self._lock:
            if session_id in self._store:
                del self._store[session_id]
                return True
            return False

    def cleanup(self) -> int:
        """Remove expired sessions.

        Returns:
            Number of sessions cleaned up
        """
        now = datetime.now()
        with self._lock:
            expired = [
                k for k, v in self._store.items()
                if now - v.last_accessed >= self._ttl
            ]
            for k in expired:
                del self._store[k]

        if expired:
            logger.info("Cleaned up %d expired sessions", len(expired))
        return len(expired)

    def _cleanup_oldest(self) -> None:
        """Remove oldest 10% of sessions when at capacity."""
        to_remove = max(1, len(self._store) // 10)
        sorted_sessions = sorted(
            self._store.items(),
            key=lambda x: x[1].last_accessed
        )
        for session_id, _ in sorted_sessions[:to_remove]:
            del self._store[session_id]
        logger.info("Cleaned up %d oldest sessions (capacity)", to_remove)

    def stats(self) -> dict:
        """Get session store statistics."""
        with self._lock:
            now = datetime.now()
            active = sum(
                1 for v in self._store.values()
                if now - v.last_accessed < self._ttl
            )
            return {
                "total_sessions": len(self._store),
                "active_sessions": active,
                "ttl_minutes": self._ttl.total_seconds() / 60,
                "max_sessions": self._max_sessions,
            }


# =============================================================================
# Redis-backed session store (Phase 11.1) — shared across instances
# =============================================================================

import json


class RedisSessionStore:
    """Session store in Redis so multi-turn context survives across instances.

    Falls back to an in-memory store on any Redis error.
    """

    def __init__(self, redis_client, ttl_minutes: int = 30, prefix: str = "sess:"):
        self._redis = redis_client
        self._ttl = ttl_minutes * 60
        self._prefix = prefix
        self._fallback = SessionStore(ttl_minutes=ttl_minutes)

    def _key(self, session_id: str) -> str:
        return f"{self._prefix}{session_id}"

    def get(self, session_id: str) -> dict[str, Any] | None:
        try:
            raw = self._redis.get(self._key(session_id))
            if raw is None:
                return None
            self._redis.expire(self._key(session_id), self._ttl)  # touch
            return json.loads(raw)
        except Exception as e:
            logger.warning("Redis session get error, using fallback: %s", e)
            return self._fallback.get(session_id)

    def set(self, session_id: str, data: dict[str, Any]) -> None:
        try:
            self._redis.setex(self._key(session_id), self._ttl, json.dumps(data))
        except Exception as e:
            logger.warning("Redis session set error, using fallback: %s", e)
            self._fallback.set(session_id, data)

    def delete(self, session_id: str) -> bool:
        try:
            return bool(self._redis.delete(self._key(session_id)))
        except Exception:
            return self._fallback.delete(session_id)

    def cleanup(self) -> int:
        # Redis expires keys automatically.
        return 0

    def stats(self) -> dict:
        return {"backend": "redis", "ttl_minutes": self._ttl / 60}


# =============================================================================
# Singleton
# =============================================================================

_session_store: SessionStore | RedisSessionStore | None = None
_store_lock = threading.Lock()


def get_session_store() -> SessionStore | RedisSessionStore:
    """Get singleton session store — Redis-backed if REDIS_URL is reachable,
    otherwise in-memory (per-instance)."""
    global _session_store

    if _session_store is None:
        with _store_lock:
            if _session_store is None:
                from server.redis_client import get_redis

                redis_client = get_redis()
                if redis_client is not None:
                    _session_store = RedisSessionStore(redis_client)
                    logger.info("SessionStore: Redis-backed")
                else:
                    _session_store = SessionStore()
                    logger.info("SessionStore: in-memory")

    return _session_store


def reset_session_store(ttl_minutes: int = 30) -> SessionStore:
    """Reset the singleton (for testing)."""
    global _session_store

    with _store_lock:
        _session_store = SessionStore(ttl_minutes=ttl_minutes)
        return _session_store
