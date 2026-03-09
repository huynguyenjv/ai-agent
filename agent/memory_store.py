"""
Persistent memory store abstraction for Long-Lived Agent pattern.
Supports both in-memory (development) and Redis (production) backends.
"""

import json
import time
from abc import ABC, abstractmethod
from typing import Optional
from dataclasses import dataclass, field, asdict

import structlog

logger = structlog.get_logger()


@dataclass
class SessionData:
    """Serializable session data for persistent storage."""
    
    session_id: str
    created_at: float
    last_activity: float
    timeout_seconds: int = 3600
    
    # Context
    current_class: Optional[str] = None
    current_file: Optional[str] = None
    current_package: Optional[str] = None
    
    # Conversation history (limited)
    history: list[dict] = field(default_factory=list)
    max_history: int = 20
    
    # Generated tests metadata (not full code to save space)
    generated_tests_meta: list[dict] = field(default_factory=list)
    
    # RAG context cache keys (actual data in separate cache)
    rag_cache_keys: list[str] = field(default_factory=list)
    
    def is_expired(self) -> bool:
        return (time.time() - self.last_activity) > self.timeout_seconds
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "SessionData":
        return cls(**data)


class MemoryStore(ABC):
    """Abstract base class for memory storage backends."""
    
    @abstractmethod
    def save_session(self, session: SessionData) -> bool:
        """Save session data."""
        pass
    
    @abstractmethod
    def load_session(self, session_id: str) -> Optional[SessionData]:
        """Load session data."""
        pass
    
    @abstractmethod
    def delete_session(self, session_id: str) -> bool:
        """Delete session data."""
        pass
    
    @abstractmethod
    def list_sessions(self) -> list[str]:
        """List all session IDs."""
        pass
    
    @abstractmethod
    def cleanup_expired(self) -> int:
        """Remove expired sessions. Returns count of removed sessions."""
        pass
    
    @abstractmethod
    def cache_rag_context(self, key: str, chunks: list[dict], ttl: int = 3600) -> bool:
        """Cache RAG context with TTL."""
        pass
    
    @abstractmethod
    def get_rag_context(self, key: str) -> Optional[list[dict]]:
        """Get cached RAG context."""
        pass
    
    @abstractmethod
    def save_generated_test(self, session_id: str, class_name: str, test_code: str) -> bool:
        """Save generated test code (for refinement)."""
        pass
    
    @abstractmethod
    def get_generated_test(self, session_id: str, class_name: str) -> Optional[str]:
        """Get last generated test code for a class."""
        pass


class InMemoryStore(MemoryStore):
    """In-memory implementation for development/testing."""
    
    def __init__(self):
        self._sessions: dict[str, SessionData] = {}
        self._rag_cache: dict[str, tuple[list[dict], float]] = {}  # key -> (data, expiry)
        self._test_cache: dict[str, str] = {}  # session_id:class_name -> code
        logger.info("InMemoryStore initialized (development mode)")
    
    def save_session(self, session: SessionData) -> bool:
        session.last_activity = time.time()
        self._sessions[session.session_id] = session
        return True
    
    def load_session(self, session_id: str) -> Optional[SessionData]:
        session = self._sessions.get(session_id)
        if session and not session.is_expired():
            return session
        elif session:
            # Auto-cleanup expired
            del self._sessions[session_id]
        return None
    
    def delete_session(self, session_id: str) -> bool:
        if session_id in self._sessions:
            del self._sessions[session_id]
            # Also cleanup related caches
            keys_to_delete = [k for k in self._test_cache if k.startswith(f"{session_id}:")]
            for k in keys_to_delete:
                del self._test_cache[k]
            return True
        return False
    
    def list_sessions(self) -> list[str]:
        return [sid for sid, s in self._sessions.items() if not s.is_expired()]
    
    def cleanup_expired(self) -> int:
        expired = [sid for sid, s in self._sessions.items() if s.is_expired()]
        for sid in expired:
            self.delete_session(sid)
        
        # Also cleanup expired RAG cache
        now = time.time()
        expired_rag = [k for k, (_, exp) in self._rag_cache.items() if exp < now]
        for k in expired_rag:
            del self._rag_cache[k]
        
        return len(expired)
    
    def cache_rag_context(self, key: str, chunks: list[dict], ttl: int = 3600) -> bool:
        expiry = time.time() + ttl
        self._rag_cache[key] = (chunks, expiry)
        return True
    
    def get_rag_context(self, key: str) -> Optional[list[dict]]:
        if key in self._rag_cache:
            data, expiry = self._rag_cache[key]
            if time.time() < expiry:
                return data
            else:
                del self._rag_cache[key]
        return None
    
    def save_generated_test(self, session_id: str, class_name: str, test_code: str) -> bool:
        key = f"{session_id}:{class_name}"
        self._test_cache[key] = test_code
        return True
    
    def get_generated_test(self, session_id: str, class_name: str) -> Optional[str]:
        key = f"{session_id}:{class_name}"
        return self._test_cache.get(key)


class RedisStore(MemoryStore):
    """Redis implementation for production (distributed, persistent)."""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        prefix: str = "ai_agent:",
    ):
        try:
            import redis
            self._redis = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=True,
            )
            self._prefix = prefix
            # Test connection
            self._redis.ping()
            logger.info("RedisStore initialized", host=host, port=port)
        except ImportError:
            raise ImportError("redis package required for RedisStore. Install with: pip install redis")
        except Exception as e:
            logger.error("Failed to connect to Redis", error=str(e))
            raise
    
    def _session_key(self, session_id: str) -> str:
        return f"{self._prefix}session:{session_id}"
    
    def _rag_key(self, key: str) -> str:
        return f"{self._prefix}rag:{key}"
    
    def _test_key(self, session_id: str, class_name: str) -> str:
        return f"{self._prefix}test:{session_id}:{class_name}"
    
    def save_session(self, session: SessionData) -> bool:
        try:
            session.last_activity = time.time()
            key = self._session_key(session.session_id)
            data = json.dumps(session.to_dict())
            # Set with TTL slightly longer than session timeout
            self._redis.setex(key, session.timeout_seconds + 60, data)
            return True
        except Exception as e:
            logger.error("Failed to save session", error=str(e))
            return False
    
    def load_session(self, session_id: str) -> Optional[SessionData]:
        try:
            key = self._session_key(session_id)
            data = self._redis.get(key)
            if data:
                session = SessionData.from_dict(json.loads(data))
                if not session.is_expired():
                    return session
                else:
                    self._redis.delete(key)
            return None
        except Exception as e:
            logger.error("Failed to load session", error=str(e))
            return None
    
    def delete_session(self, session_id: str) -> bool:
        try:
            key = self._session_key(session_id)
            # Also delete related test caches
            pattern = f"{self._prefix}test:{session_id}:*"
            for test_key in self._redis.scan_iter(pattern):
                self._redis.delete(test_key)
            return self._redis.delete(key) > 0
        except Exception as e:
            logger.error("Failed to delete session", error=str(e))
            return False
    
    def list_sessions(self) -> list[str]:
        try:
            pattern = f"{self._prefix}session:*"
            keys = list(self._redis.scan_iter(pattern))
            # Extract session IDs from keys
            prefix_len = len(f"{self._prefix}session:")
            return [k[prefix_len:] for k in keys]
        except Exception as e:
            logger.error("Failed to list sessions", error=str(e))
            return []
    
    def cleanup_expired(self) -> int:
        # Redis handles expiry automatically via TTL
        # This is mainly for manual cleanup if needed
        return 0
    
    def cache_rag_context(self, key: str, chunks: list[dict], ttl: int = 3600) -> bool:
        try:
            redis_key = self._rag_key(key)
            data = json.dumps(chunks)
            self._redis.setex(redis_key, ttl, data)
            return True
        except Exception as e:
            logger.error("Failed to cache RAG context", error=str(e))
            return False
    
    def get_rag_context(self, key: str) -> Optional[list[dict]]:
        try:
            redis_key = self._rag_key(key)
            data = self._redis.get(redis_key)
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.error("Failed to get RAG context", error=str(e))
            return None
    
    def save_generated_test(self, session_id: str, class_name: str, test_code: str) -> bool:
        try:
            key = self._test_key(session_id, class_name)
            # Keep test code for 1 hour (for refinement)
            self._redis.setex(key, 3600, test_code)
            return True
        except Exception as e:
            logger.error("Failed to save generated test", error=str(e))
            return False
    
    def get_generated_test(self, session_id: str, class_name: str) -> Optional[str]:
        try:
            key = self._test_key(session_id, class_name)
            return self._redis.get(key)
        except Exception as e:
            logger.error("Failed to get generated test", error=str(e))
            return None


def create_memory_store(backend: str = "memory", **kwargs) -> MemoryStore:
    """Factory function to create memory store based on configuration.
    
    Args:
        backend: "memory" for in-memory (dev) or "redis" for Redis (prod)
        **kwargs: Backend-specific configuration
    
    Returns:
        MemoryStore instance (falls back to InMemoryStore if Redis is unavailable)
    """
    if backend == "redis":
        try:
            return RedisStore(**kwargs)
        except ImportError:
            logger.warning("Redis package not installed, falling back to in-memory store")
            return InMemoryStore()
        except Exception as e:
            logger.warning("Redis connection failed, falling back to in-memory store", error=str(e))
            return InMemoryStore()
    else:
        return InMemoryStore()

