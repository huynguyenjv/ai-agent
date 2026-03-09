"""
Session management for the API server.
Supports Redis-backed persistence for production deployments.
"""

import threading
import uuid
from typing import Optional
from datetime import datetime

import structlog
from pydantic import BaseModel, Field

from utils.cache_service import get_cache_service

logger = structlog.get_logger()


class SessionInfo(BaseModel):
    """Session information model."""

    session_id: str
    created_at: datetime
    last_activity: datetime
    current_class: Optional[str] = None
    current_file: Optional[str] = None
    tests_generated: int = 0
    is_active: bool = True


class SessionManager:
    """Manages API sessions with Redis-backed persistence.
    
    Sessions are stored in Redis (if available) with TTL-based expiration.
    Falls back to in-memory storage if Redis is unavailable.
    """

    def __init__(self, default_timeout: int = 3600):
        self.default_timeout = default_timeout
        # Local cache for fast lookups (always populated)
        self._local_cache: dict[str, SessionInfo] = {}
        # Thread lock for _local_cache access
        self._lock = threading.Lock()
        # Redis cache service (initialized lazily)
        self._cache = None

    @property
    def cache(self):
        """Lazy-init cache service to avoid import ordering issues."""
        if self._cache is None:
            self._cache = get_cache_service()
        return self._cache

    def create_session(self) -> SessionInfo:
        """Create a new session (persisted to Redis if available)."""
        session_id = str(uuid.uuid4())
        now = datetime.utcnow()

        session = SessionInfo(
            session_id=session_id,
            created_at=now,
            last_activity=now,
        )
        
        # Store locally (thread-safe)
        with self._lock:
            self._local_cache[session_id] = session
        
        # Persist to Redis
        self.cache.cache_session(
            session_id, 
            session.model_dump(mode="json"),
            ttl=self.default_timeout,
        )

        logger.info("Session created", session_id=session_id, backend=self.cache.backend_name)
        return session

    def get_session(self, session_id: str) -> Optional[SessionInfo]:
        """Get session by ID (checks local cache first, then Redis)."""
        # Check local cache first (thread-safe)
        with self._lock:
            if session_id in self._local_cache:
                session = self._local_cache[session_id]
                if self._is_expired(session):
                    del self._local_cache[session_id]
                    # Also delete from Redis (outside lock)
                    self.cache.delete_session(session_id)
                    return None
                return session
        
        # Try Redis
        data = self.cache.get_session(session_id)
        if data:
            try:
                session = SessionInfo(**data)
                if not self._is_expired(session):
                    with self._lock:
                        self._local_cache[session_id] = session
                    return session
                else:
                    self.cache.delete_session(session_id)
            except Exception as e:
                logger.error("Failed to deserialize session from Redis", 
                           session_id=session_id, error=str(e))
        
        return None

    def update_session(
        self,
        session_id: str,
        current_class: Optional[str] = None,
        current_file: Optional[str] = None,
        increment_tests: bool = False,
    ) -> Optional[SessionInfo]:
        """Update session information (persisted to Redis)."""
        session = self.get_session(session_id)
        if not session:
            return None

        session.last_activity = datetime.utcnow()

        if current_class:
            session.current_class = current_class
        if current_file:
            session.current_file = current_file
        if increment_tests:
            session.tests_generated += 1

        # Update local (thread-safe) and Redis
        with self._lock:
            self._local_cache[session_id] = session
        self.cache.cache_session(
            session_id, 
            session.model_dump(mode="json"),
            ttl=self.default_timeout,
        )

        return session

    def delete_session(self, session_id: str) -> bool:
        """Delete a session (from both local cache and Redis)."""
        deleted = False
        
        with self._lock:
            if session_id in self._local_cache:
                del self._local_cache[session_id]
                deleted = True

        if self.cache.delete_session(session_id):
            deleted = True

        if deleted:
            logger.info("Session deleted", session_id=session_id)
        return deleted

    def list_sessions(self) -> list[SessionInfo]:
        """List all active sessions."""
        # Remove expired sessions from local cache first
        self._cleanup_local_cache()
        with self._lock:
            return list(self._local_cache.values())

    def cleanup_expired(self) -> int:
        """Remove expired sessions."""
        count = self._cleanup_local_cache()

        if count > 0:
            logger.info("Cleaned up expired sessions", count=count)

        return count

    def _is_expired(self, session: SessionInfo) -> bool:
        """Check if session has expired."""
        now = datetime.utcnow()
        elapsed = (now - session.last_activity).total_seconds()
        return elapsed > self.default_timeout

    def _cleanup_local_cache(self) -> int:
        """Remove expired sessions from local cache."""
        now = datetime.utcnow()

        with self._lock:
            expired = [
                sid for sid, session in self._local_cache.items()
                if (now - session.last_activity).total_seconds() > self.default_timeout
            ]
            for session_id in expired:
                del self._local_cache[session_id]

        # Delete from Redis outside lock (Redis ops can be slow)
        for session_id in expired:
            self.cache.delete_session(session_id)

        return len(expired)

