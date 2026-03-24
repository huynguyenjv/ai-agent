"""HashStore — Section 5, HashStore subsection.

Local SQLite database at ~/.ai-agent/hash_store.db.
A record is only written after the server confirms successful ingestion.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path


class HashStore:
    """SQLite-backed file hash store for cache invalidation.

    Invariant: A record is only written after the server confirms
    successful ingestion. Never written on upload failure.
    """

    def __init__(self, db_path: str | None = None):
        if db_path is None:
            home = Path.home()
            store_dir = home / ".ai-agent"
            store_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(store_dir / "hash_store.db")

        self._db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS file_hashes (
                file_path TEXT PRIMARY KEY,
                md5_hash TEXT NOT NULL,
                indexed_at REAL NOT NULL
            )
            """
        )
        self._conn.commit()

    def get_hash(self, file_path: str) -> str | None:
        """Return stored MD5 hash for a file, or None if absent."""
        cursor = self._conn.execute(
            "SELECT md5_hash FROM file_hashes WHERE file_path = ?",
            (file_path,),
        )
        row = cursor.fetchone()
        return row[0] if row else None

    def set_hash(self, file_path: str, md5_hash: str) -> None:
        """Store or update the hash for a file. Only call after successful upload."""
        self._conn.execute(
            """
            INSERT INTO file_hashes (file_path, md5_hash, indexed_at)
            VALUES (?, ?, ?)
            ON CONFLICT(file_path) DO UPDATE SET
                md5_hash = excluded.md5_hash,
                indexed_at = excluded.indexed_at
            """,
            (file_path, md5_hash, time.time()),
        )
        self._conn.commit()

    def set_hashes_batch(self, entries: list[tuple[str, str]]) -> None:
        """Batch update hashes. Each entry is (file_path, md5_hash)."""
        now = time.time()
        self._conn.executemany(
            """
            INSERT INTO file_hashes (file_path, md5_hash, indexed_at)
            VALUES (?, ?, ?)
            ON CONFLICT(file_path) DO UPDATE SET
                md5_hash = excluded.md5_hash,
                indexed_at = excluded.indexed_at
            """,
            [(fp, h, now) for fp, h in entries],
        )
        self._conn.commit()

    def remove(self, file_path: str) -> None:
        """Remove a file's hash entry."""
        self._conn.execute(
            "DELETE FROM file_hashes WHERE file_path = ?", (file_path,)
        )
        self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
