"""Streaming Optimization — Phase 19.4.

Backpressure (bounded SSE event queue) + token coalescing to reduce per-token
SSE overhead. Client-disconnect detection already lives in routers/chat.py.
"""

from __future__ import annotations

import asyncio
import os

SSE_QUEUE_MAXSIZE = int(os.environ.get("SSE_QUEUE_MAXSIZE", "256"))


def make_event_queue(maxsize: int | None = None) -> asyncio.Queue:
    """Bounded queue → producer awaits when full (backpressure on slow clients)."""
    return asyncio.Queue(maxsize=maxsize if maxsize is not None else SSE_QUEUE_MAXSIZE)


class TokenCoalescer:
    """Coalesce small content tokens into larger chunks.

    Flushes when the buffer reaches `min_chars` (or on explicit flush). Reduces
    the number of SSE frames without adding latency beyond one token.
    """

    def __init__(self, min_chars: int = 24):
        self._buf: list[str] = []
        self._size = 0
        self._min = min_chars

    def add(self, token: str) -> str | None:
        """Add a token; return a chunk to emit if the threshold is reached, else None."""
        if not token:
            return None
        self._buf.append(token)
        self._size += len(token)
        if self._size >= self._min:
            return self.flush()
        return None

    def flush(self) -> str | None:
        """Return and clear any buffered content."""
        if not self._buf:
            return None
        chunk = "".join(self._buf)
        self._buf.clear()
        self._size = 0
        return chunk
