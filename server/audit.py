"""Audit Logging — Phase 10.5.

Persistent audit trail for security-relevant events: tool executions, auth
events, security violations, admin actions. SQLite by default (file or
in-memory), PostgreSQL when DATABASE_URL is set (mirrors server/metrics).

Audit failures must never break a request — all writes are best-effort.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any

logger = logging.getLogger("server.audit")


class AuditEventType(str, Enum):
    TOOL_EXECUTION = "tool_execution"
    AUTH = "auth"
    SECURITY_VIOLATION = "security_violation"
    ADMIN = "admin"


@dataclass
class AuditEvent:
    """A single audit-trail entry."""
    event_type: str
    action: str
    actor: str = "anonymous"          # api-key id or client IP
    outcome: str = "ok"               # ok | blocked | error
    detail: dict[str, Any] = field(default_factory=dict)
    correlation_id: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type,
            "action": self.action,
            "actor": self.actor,
            "outcome": self.outcome,
            "detail": self.detail,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp.isoformat(),
        }


class AuditStorage(ABC):
    @abstractmethod
    def record(self, event: AuditEvent) -> None: ...

    @abstractmethod
    def recent(self, limit: int = 100) -> list[dict]: ...

    @abstractmethod
    def clear_old(self, days: int = 90) -> int: ...


class SQLiteAuditStorage(AuditStorage):
    """SQLite backend with a single shared connection (works with :memory:)."""

    def __init__(self, db_path: str):
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    action TEXT NOT NULL,
                    actor TEXT,
                    outcome TEXT,
                    detail TEXT,
                    correlation_id TEXT,
                    timestamp TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit_log(timestamp)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_audit_type ON audit_log(event_type)"
            )
            self._conn.commit()

    def record(self, event: AuditEvent) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO audit_log (event_type, action, actor, outcome, detail, "
                "correlation_id, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    event.event_type,
                    event.action,
                    event.actor,
                    event.outcome,
                    json.dumps(event.detail, ensure_ascii=False),
                    event.correlation_id,
                    event.timestamp.isoformat(),
                ),
            )
            self._conn.commit()

    def recent(self, limit: int = 100) -> list[dict]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,)
            )
            rows = [dict(r) for r in cur.fetchall()]
        for r in rows:
            try:
                r["detail"] = json.loads(r["detail"]) if r["detail"] else {}
            except (TypeError, json.JSONDecodeError):
                r["detail"] = {}
        return rows

    def clear_old(self, days: int = 90) -> int:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM audit_log WHERE timestamp < ?", (cutoff,)
            )
            self._conn.commit()
            return cur.rowcount


class PostgresAuditStorage(AuditStorage):
    """PostgreSQL backend (psycopg pool). Schema in init-db/002_audit_schema.sql."""

    def __init__(self, database_url: str):
        from psycopg_pool import ConnectionPool

        self._pool = ConnectionPool(database_url, min_size=1, max_size=4, open=True)
        logger.info("Audit PostgreSQL pool initialized")

    def record(self, event: AuditEvent) -> None:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO audit_log (event_type, action, actor, outcome, detail, "
                    "correlation_id, timestamp) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    (
                        event.event_type,
                        event.action,
                        event.actor,
                        event.outcome,
                        json.dumps(event.detail, ensure_ascii=False),
                        event.correlation_id,
                        event.timestamp,
                    ),
                )

    def recent(self, limit: int = 100) -> list[dict]:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT event_type, action, actor, outcome, detail, correlation_id, "
                    "timestamp FROM audit_log ORDER BY id DESC LIMIT %s",
                    (limit,),
                )
                cols = [c.name for c in cur.description]
                return [dict(zip(cols, row)) for row in cur.fetchall()]

    def clear_old(self, days: int = 90) -> int:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM audit_log WHERE timestamp < NOW() - INTERVAL '%s days'",
                    (days,),
                )
                return cur.rowcount


class AuditLogger:
    """Best-effort audit logger — never raises into the request path."""

    def __init__(self, storage: AuditStorage):
        self._storage = storage

    def log(self, event: AuditEvent) -> None:
        try:
            self._storage.record(event)
        except Exception as e:  # audit must not break requests
            logger.error("audit write failed: %s", e)
        # Mirror to standard logs for SIEM scraping
        log_fn = logger.warning if event.outcome != "ok" else logger.info
        log_fn("AUDIT %s/%s actor=%s outcome=%s detail=%s",
               event.event_type, event.action, event.actor, event.outcome, event.detail)

    # Convenience helpers
    def tool_execution(self, action: str, actor: str, outcome: str = "ok",
                       correlation_id: str = "", **detail) -> None:
        self.log(AuditEvent(AuditEventType.TOOL_EXECUTION.value, action, actor,
                            outcome, detail, correlation_id))

    def auth(self, action: str, actor: str, outcome: str,
             correlation_id: str = "", **detail) -> None:
        self.log(AuditEvent(AuditEventType.AUTH.value, action, actor,
                            outcome, detail, correlation_id))

    def security_violation(self, action: str, actor: str,
                          correlation_id: str = "", **detail) -> None:
        self.log(AuditEvent(AuditEventType.SECURITY_VIOLATION.value, action, actor,
                            "blocked", detail, correlation_id))

    def recent(self, limit: int = 100) -> list[dict]:
        try:
            return self._storage.recent(limit)
        except Exception as e:
            logger.error("audit read failed: %s", e)
            return []

    def clear_old(self, days: int = 90) -> int:
        try:
            return self._storage.clear_old(days)
        except Exception as e:
            logger.error("audit cleanup failed: %s", e)
            return 0


_audit: AuditLogger | None = None
_lock = threading.Lock()


def get_audit_logger() -> AuditLogger:
    """Singleton. Postgres if DATABASE_URL set, else SQLite (AUDIT_DB_PATH)."""
    global _audit
    if _audit is None:
        with _lock:
            if _audit is None:
                database_url = os.environ.get("DATABASE_URL")
                if database_url:
                    try:
                        storage: AuditStorage = PostgresAuditStorage(database_url)
                    except Exception as e:
                        logger.error("Audit Postgres init failed, falling back to SQLite: %s", e)
                        storage = SQLiteAuditStorage(_default_sqlite_path())
                else:
                    storage = SQLiteAuditStorage(_default_sqlite_path())
                _audit = AuditLogger(storage)
    return _audit


def _default_sqlite_path() -> str:
    path = os.environ.get("AUDIT_DB_PATH", "data/audit.db")
    if path != ":memory:":
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    return path
