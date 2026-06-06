"""Cross-session Memory — Phase R10 (Phase 15.4).

Durable per-scope memory that survives across conversations. SQLite by default,
Postgres when DATABASE_URL is set (mirrors server/audit). Recall is keyword +
recency based (no embeddings — RAG/embedder is opt-in/off). Opt-in via
ENABLE_MEMORY (default off) so it never changes default behavior.
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
import threading
import time
from abc import ABC, abstractmethod

logger = logging.getLogger("server.agent.memory")


def enable_memory() -> bool:
    return os.environ.get("ENABLE_MEMORY", "false").lower() in ("1", "true", "yes")


def _tokens(text: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9_]+", (text or "").lower()) if len(t) > 2}


class MemoryStorage(ABC):
    @abstractmethod
    def add(self, scope: str, content: str, ttl: int | None) -> None: ...
    @abstractmethod
    def fetch(self, scope: str) -> list[dict]: ...
    @abstractmethod
    def clear(self, scope: str) -> int: ...


class SQLiteMemoryStorage(MemoryStorage):
    def __init__(self, db_path: str):
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute(
                """CREATE TABLE IF NOT EXISTS agent_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scope TEXT NOT NULL, content TEXT NOT NULL,
                    created_at REAL NOT NULL, expires_at REAL
                )"""
            )
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_mem_scope ON agent_memory(scope)")
            self._conn.commit()

    def add(self, scope, content, ttl):
        exp = time.time() + ttl if ttl else None
        with self._lock:
            self._conn.execute(
                "INSERT INTO agent_memory (scope, content, created_at, expires_at) VALUES (?,?,?,?)",
                (scope, content, time.time(), exp),
            )
            self._conn.commit()

    def fetch(self, scope):
        now = time.time()
        with self._lock:
            rows = self._conn.execute(
                "SELECT content, created_at FROM agent_memory "
                "WHERE scope=? AND (expires_at IS NULL OR expires_at > ?) ORDER BY id DESC LIMIT 200",
                (scope, now),
            ).fetchall()
        return [dict(r) for r in rows]

    def clear(self, scope):
        with self._lock:
            cur = self._conn.execute("DELETE FROM agent_memory WHERE scope=?", (scope,))
            self._conn.commit()
            return cur.rowcount


class PostgresMemoryStorage(MemoryStorage):
    def __init__(self, database_url: str):
        from psycopg_pool import ConnectionPool
        self._pool = ConnectionPool(database_url, min_size=1, max_size=4, open=True)

    def add(self, scope, content, ttl):
        with self._pool.connection() as c, c.cursor() as cur:
            cur.execute(
                "INSERT INTO agent_memory (scope, content, created_at, expires_at) "
                "VALUES (%s,%s,NOW(),%s)",
                (scope, content, None if not ttl else __import__("datetime").datetime.now()
                 + __import__("datetime").timedelta(seconds=ttl)),
            )

    def fetch(self, scope):
        with self._pool.connection() as c, c.cursor() as cur:
            cur.execute(
                "SELECT content, EXTRACT(EPOCH FROM created_at) FROM agent_memory "
                "WHERE scope=%s AND (expires_at IS NULL OR expires_at > NOW()) ORDER BY id DESC LIMIT 200",
                (scope,),
            )
            return [{"content": r[0], "created_at": r[1]} for r in cur.fetchall()]

    def clear(self, scope):
        with self._pool.connection() as c, c.cursor() as cur:
            cur.execute("DELETE FROM agent_memory WHERE scope=%s", (scope,))
            return cur.rowcount


class MemoryStore:
    def __init__(self, storage: MemoryStorage):
        self._storage = storage

    def remember(self, scope: str, content: str, ttl: int | None = None) -> None:
        if not scope or not content:
            return
        try:
            self._storage.add(scope, content, ttl)
        except Exception as e:
            logger.warning("memory remember failed: %s", e)

    def recall(self, scope: str, query: str, top_k: int = 5) -> list[dict]:
        try:
            rows = self._storage.fetch(scope)
        except Exception as e:
            logger.warning("memory recall failed: %s", e)
            return []
        q = _tokens(query)
        scored = []
        for r in rows:
            overlap = len(q & _tokens(r["content"]))
            scored.append((overlap, r["created_at"], r))
        # keyword overlap first, then recency
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return [r for _, _, r in scored[:top_k]]

    def forget(self, scope: str) -> int:
        try:
            return self._storage.clear(scope)
        except Exception:
            return 0


_store: MemoryStore | None = None
_lock = threading.Lock()


def get_memory_store() -> MemoryStore:
    global _store
    if _store is None:
        with _lock:
            if _store is None:
                url = os.environ.get("DATABASE_URL")
                if url:
                    try:
                        storage: MemoryStorage = PostgresMemoryStorage(url)
                    except Exception as e:
                        logger.error("memory Postgres init failed, SQLite fallback: %s", e)
                        storage = SQLiteMemoryStorage(_sqlite_path())
                else:
                    storage = SQLiteMemoryStorage(_sqlite_path())
                _store = MemoryStore(storage)
    return _store


def _sqlite_path() -> str:
    path = os.environ.get("MEMORY_DB_PATH", "data/memory.db")
    if path != ":memory:":
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    return path


def reset_memory_store() -> None:
    global _store
    _store = None
