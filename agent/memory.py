"""
Session memory for maintaining agent context across interactions.
"""

import time
from dataclasses import dataclass, field
from typing import Optional
from collections import deque

import structlog

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
    """Manages session state and conversation history."""

    def __init__(
        self,
        session_id: str,
        max_history: int = 20,
        timeout_seconds: int = 3600,
    ):
        self.session_id = session_id
        self.max_history = max_history
        self.timeout_seconds = timeout_seconds
        self.created_at = time.time()
        self.last_activity = time.time()

        # Conversation history
        self.history: deque[ConversationTurn] = deque(maxlen=max_history)

        # Context about the current task
        self.current_class: Optional[str] = None
        self.current_file: Optional[str] = None
        self.current_package: Optional[str] = None

        # Generated tests in this session
        self.generated_tests: list[GeneratedTest] = []

        # RAG context cache
        self.rag_context: dict[str, list[dict]] = {}

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
        """Cache RAG results for reuse."""
        self.rag_context[key] = chunks
        self.last_activity = time.time()

    def get_cached_rag_context(self, key: str) -> Optional[list[dict]]:
        """Get cached RAG results."""
        return self.rag_context.get(key)

    def record_generated_test(
        self,
        class_name: str,
        test_code: str,
        success: bool = True,
        feedback: Optional[str] = None,
    ) -> None:
        """Record a generated test."""
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
        self.rag_context.clear()
        logger.info("Session cleared", session_id=self.session_id)


class MemoryManager:
    """Manages multiple session memories."""

    def __init__(self, default_timeout: int = 3600):
        self.sessions: dict[str, SessionMemory] = {}
        self.default_timeout = default_timeout

    def get_or_create_session(self, session_id: str) -> SessionMemory:
        """Get existing session or create new one."""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            if not session.is_expired():
                return session
            else:
                # Clean up expired session
                del self.sessions[session_id]

        # Create new session
        session = SessionMemory(
            session_id=session_id,
            timeout_seconds=self.default_timeout,
        )
        self.sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[SessionMemory]:
        """Get session if it exists and is not expired."""
        session = self.sessions.get(session_id)
        if session and not session.is_expired():
            return session
        return None

    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info("Session deleted", session_id=session_id)
            return True
        return False

    def cleanup_expired(self) -> int:
        """Remove all expired sessions."""
        expired = [
            sid for sid, session in self.sessions.items() if session.is_expired()
        ]
        for sid in expired:
            del self.sessions[sid]

        if expired:
            logger.info("Cleaned up expired sessions", count=len(expired))
        return len(expired)

    def get_active_sessions(self) -> list[str]:
        """Get list of active session IDs."""
        return [
            sid for sid, session in self.sessions.items() if not session.is_expired()
        ]

