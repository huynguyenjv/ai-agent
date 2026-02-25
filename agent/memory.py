"""
Session memory for maintaining agent context across interactions.
Supports pluggable storage backends (in-memory for dev, Redis for prod).
"""

import os
import time
from dataclasses import dataclass, field
from typing import Optional
from collections import deque

import structlog

from .memory_store import MemoryStore, SessionData, create_memory_store

logger = structlog.get_logger()


@dataclass
class ConversationTurn:
    """A single turn in the conversation."""

    role: str  # user | assistant
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)


@dataclass
class GeneratedTest:
    """Record of a generated test."""

    class_name: str
    test_code: str
    timestamp: float
    success: bool
    feedback: Optional[str] = None


class SessionMemory:
    """Manages session state and conversation history.
    
    Supports optional persistent storage for Long-Lived Agent pattern.
    """

    def __init__(
        self,
        session_id: str,
        max_history: int = 20,
        timeout_seconds: int = 3600,
        store: Optional[MemoryStore] = None,
    ):
        self.session_id = session_id
        self.max_history = max_history
        self.timeout_seconds = timeout_seconds
        self.created_at = time.time()
        self.last_activity = time.time()
        self._store = store  # Optional persistent store

        # Conversation history
        self.history: deque[ConversationTurn] = deque(maxlen=max_history)

        # Context about the current task
        self.current_class: Optional[str] = None
        self.current_file: Optional[str] = None
        self.current_package: Optional[str] = None

        # Generated tests in this session
        self.generated_tests: list[GeneratedTest] = []

        # RAG context cache (local, backed by store if available)
        self._rag_context_local: dict[str, list[dict]] = {}

        logger.info("Session created", session_id=session_id)

    def add_user_message(self, content: str, metadata: Optional[dict] = None) -> None:
        """Add a user message to history."""
        self.history.append(
            ConversationTurn(
                role="user",
                content=content,
                metadata=metadata or {},
            )
        )
        self.last_activity = time.time()

    def add_assistant_message(
        self, content: str, metadata: Optional[dict] = None
    ) -> None:
        """Add an assistant message to history."""
        self.history.append(
            ConversationTurn(
                role="assistant",
                content=content,
                metadata=metadata or {},
            )
        )
        self.last_activity = time.time()

    def set_context(
        self,
        class_name: Optional[str] = None,
        file_path: Optional[str] = None,
        package: Optional[str] = None,
    ) -> None:
        """Set the current working context."""
        if class_name:
            self.current_class = class_name
        if file_path:
            self.current_file = file_path
        if package:
            self.current_package = package
        self.last_activity = time.time()

    def cache_rag_context(self, key: str, chunks: list[dict]) -> None:
        """Cache RAG results for reuse (with optional persistent storage)."""
        self._rag_context_local[key] = chunks
        self.last_activity = time.time()
        
        # Also persist to store if available
        if self._store:
            self._store.cache_rag_context(key, chunks, ttl=3600)

    def get_cached_rag_context(self, key: str) -> Optional[list[dict]]:
        """Get cached RAG results (checks local cache first, then store)."""
        # Check local cache first
        if key in self._rag_context_local:
            return self._rag_context_local[key]
        
        # Try persistent store
        if self._store:
            cached = self._store.get_rag_context(key)
            if cached:
                self._rag_context_local[key] = cached  # Populate local cache
                return cached
        
        return None

    def record_generated_test(
        self,
        class_name: str,
        test_code: str,
        success: bool = True,
        feedback: Optional[str] = None,
    ) -> None:
        """Record a generated test (with optional persistent storage)."""
        self.generated_tests.append(
            GeneratedTest(
                class_name=class_name,
                test_code=test_code,
                timestamp=time.time(),
                success=success,
                feedback=feedback,
            )
        )
        self.last_activity = time.time()
        
        # Persist test code to store for refinement
        if self._store:
            self._store.save_generated_test(self.session_id, class_name, test_code)

    def get_conversation_context(self, max_turns: int = 5) -> str:
        """Get recent conversation as context string."""
        recent = list(self.history)[-max_turns:]
        context_parts = []

        for turn in recent:
            role_label = "User" if turn.role == "user" else "Assistant"
            # Truncate long messages
            content = turn.content[:500] + "..." if len(turn.content) > 500 else turn.content
            context_parts.append(f"{role_label}: {content}")

        return "\n\n".join(context_parts)

    def get_session_summary(self) -> dict:
        """Get a summary of the session state."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "last_activity": self.last_activity,
            "conversation_turns": len(self.history),
            "current_class": self.current_class,
            "current_file": self.current_file,
            "tests_generated": len(self.generated_tests),
            "successful_tests": sum(1 for t in self.generated_tests if t.success),
            "is_expired": self.is_expired(),
        }

    def is_expired(self) -> bool:
        """Check if the session has expired."""
        return (time.time() - self.last_activity) > self.timeout_seconds

    def clear(self) -> None:
        """Clear all session data."""
        self.history.clear()
        self.current_class = None
        self.current_file = None
        self.current_package = None
        self.generated_tests.clear()
        self._rag_context_local.clear()
        logger.info("Session cleared", session_id=self.session_id)


class MemoryManager:
    """Manages multiple session memories with pluggable storage backend.
    
    For Long-Lived Agent pattern:
    - In-memory store for development (fast, no dependencies)
    - Redis store for production (distributed, persistent)
    
    Configure via environment variables:
    - MEMORY_BACKEND: "memory" (default) or "redis"
    - REDIS_HOST, REDIS_PORT, REDIS_PASSWORD, REDIS_DB (for Redis backend)
    """

    def __init__(self, default_timeout: int = 3600, store: Optional[MemoryStore] = None):
        self.default_timeout = default_timeout
        
        # Local cache for active sessions (reduces store lookups)
        self._local_cache: dict[str, SessionMemory] = {}
        
        # Initialize storage backend
        if store:
            self._store = store
        else:
            backend = os.getenv("MEMORY_BACKEND", "memory")
            if backend == "redis":
                self._store = create_memory_store(
                    backend="redis",
                    host=os.getenv("REDIS_HOST", "localhost"),
                    port=int(os.getenv("REDIS_PORT", "6379")),
                    password=os.getenv("REDIS_PASSWORD"),
                    db=int(os.getenv("REDIS_DB", "0")),
                )
            else:
                self._store = create_memory_store(backend="memory")
        
        logger.info("MemoryManager initialized", backend=type(self._store).__name__)

    def get_or_create_session(self, session_id: str) -> SessionMemory:
        """Get existing session or create new one."""
        # Check local cache first
        if session_id in self._local_cache:
            session = self._local_cache[session_id]
            if not session.is_expired():
                return session
            else:
                del self._local_cache[session_id]
        
        # Try to load from store
        session_data = self._store.load_session(session_id)
        if session_data:
            session = self._restore_session(session_data)
            self._local_cache[session_id] = session
            return session

        # Create new session
        session = SessionMemory(
            session_id=session_id,
            timeout_seconds=self.default_timeout,
            store=self._store,
        )
        self._local_cache[session_id] = session
        self._persist_session(session)
        return session

    def get_session(self, session_id: str) -> Optional[SessionMemory]:
        """Get session if it exists and is not expired."""
        # Check local cache
        if session_id in self._local_cache:
            session = self._local_cache[session_id]
            if not session.is_expired():
                return session
            else:
                del self._local_cache[session_id]
        
        # Try to load from store
        session_data = self._store.load_session(session_id)
        if session_data:
            session = self._restore_session(session_data)
            self._local_cache[session_id] = session
            return session
        
        return None

    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        # Remove from local cache
        if session_id in self._local_cache:
            del self._local_cache[session_id]
        
        # Remove from store
        result = self._store.delete_session(session_id)
        if result:
            logger.info("Session deleted", session_id=session_id)
        return result

    def cleanup_expired(self) -> int:
        """Remove all expired sessions."""
        # Cleanup local cache
        expired_local = [
            sid for sid, session in self._local_cache.items() if session.is_expired()
        ]
        for sid in expired_local:
            del self._local_cache[sid]
        
        # Cleanup store
        store_count = self._store.cleanup_expired()
        
        total = len(expired_local) + store_count
        if total > 0:
            logger.info("Cleaned up expired sessions", count=total)
        return total

    def get_active_sessions(self) -> list[str]:
        """Get list of active session IDs."""
        return self._store.list_sessions()
    
    def _persist_session(self, session: SessionMemory) -> None:
        """Persist session to store."""
        session_data = SessionData(
            session_id=session.session_id,
            created_at=session.created_at,
            last_activity=session.last_activity,
            timeout_seconds=session.timeout_seconds,
            current_class=session.current_class,
            current_file=session.current_file,
            current_package=session.current_package,
            history=[
                {"role": t.role, "content": t.content[:500], "timestamp": t.timestamp}
                for t in list(session.history)[-10:]  # Only persist last 10 turns
            ],
            max_history=session.max_history,
            generated_tests_meta=[
                {"class_name": t.class_name, "timestamp": t.timestamp, "success": t.success}
                for t in session.generated_tests[-5:]  # Only metadata, not full code
            ],
        )
        self._store.save_session(session_data)
    
    def _restore_session(self, data: SessionData) -> SessionMemory:
        """Restore SessionMemory from SessionData."""
        session = SessionMemory(
            session_id=data.session_id,
            max_history=data.max_history,
            timeout_seconds=data.timeout_seconds,
            store=self._store,
        )
        session.created_at = data.created_at
        session.last_activity = data.last_activity
        session.current_class = data.current_class
        session.current_file = data.current_file
        session.current_package = data.current_package
        
        # Restore history
        for h in data.history:
            session.history.append(ConversationTurn(
                role=h["role"],
                content=h["content"],
                timestamp=h.get("timestamp", time.time()),
            ))
        
        # Restore generated tests metadata
        for t in data.generated_tests_meta:
            # Try to get full test code from store
            test_code = self._store.get_generated_test(data.session_id, t["class_name"])
            session.generated_tests.append(GeneratedTest(
                class_name=t["class_name"],
                test_code=test_code or "",
                timestamp=t["timestamp"],
                success=t["success"],
            ))
        
        return session

