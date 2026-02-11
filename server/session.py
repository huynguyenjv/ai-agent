"""
Session management for the API server.
"""

import uuid
from typing import Optional
from datetime import datetime

import structlog
from pydantic import BaseModel, Field

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
    """Manages API sessions."""

    def __init__(self, default_timeout: int = 3600):
        self.sessions: dict[str, SessionInfo] = {}
        self.default_timeout = default_timeout

    def create_session(self) -> SessionInfo:
        """Create a new session."""
        session_id = str(uuid.uuid4())
        now = datetime.utcnow()

        session = SessionInfo(
            session_id=session_id,
            created_at=now,
            last_activity=now,
        )
        self.sessions[session_id] = session

        logger.info("Session created", session_id=session_id)
        return session

    def get_session(self, session_id: str) -> Optional[SessionInfo]:
        """Get session by ID."""
        return self.sessions.get(session_id)

    def update_session(
        self,
        session_id: str,
        current_class: Optional[str] = None,
        current_file: Optional[str] = None,
        increment_tests: bool = False,
    ) -> Optional[SessionInfo]:
        """Update session information."""
        session = self.sessions.get(session_id)
        if not session:
            return None

        session.last_activity = datetime.utcnow()

        if current_class:
            session.current_class = current_class
        if current_file:
            session.current_file = current_file
        if increment_tests:
            session.tests_generated += 1

        return session

    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info("Session deleted", session_id=session_id)
            return True
        return False

    def list_sessions(self) -> list[SessionInfo]:
        """List all active sessions."""
        return list(self.sessions.values())

    def cleanup_expired(self) -> int:
        """Remove expired sessions."""
        now = datetime.utcnow()
        expired = []

        for session_id, session in self.sessions.items():
            elapsed = (now - session.last_activity).total_seconds()
            if elapsed > self.default_timeout:
                expired.append(session_id)

        for session_id in expired:
            del self.sessions[session_id]

        if expired:
            logger.info("Cleaned up expired sessions", count=len(expired))

        return len(expired)

