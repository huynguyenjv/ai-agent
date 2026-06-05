"""Request Batching — Phase 19.1.

Generic async micro-batcher: coalesces concurrent `submit()` calls into one
`process_fn(items)` call within a small time/size window. Useful for batching
embeddings (when RAG is on) or any batchable backend op. Each caller still gets
its own result.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable, Sequence, TypeVar

logger = logging.getLogger("server.batch")

I = TypeVar("I")
R = TypeVar("R")


class AsyncBatcher:
    def __init__(
        self,
        process_fn: Callable[[Sequence[I]], Awaitable[Sequence[R]]],
        max_batch: int = 32,
        max_wait: float = 0.05,
    ):
        self._process = process_fn
        self._max_batch = max_batch
        self._max_wait = max_wait
        self._pending: list[tuple] = []   # (item, future)
        self._lock = asyncio.Lock()
        self._timer: asyncio.Task | None = None

    async def submit(self, item: I) -> R:
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        async with self._lock:
            self._pending.append((item, fut))
            if len(self._pending) >= self._max_batch:
                await self._flush_locked()
            elif self._timer is None:
                self._timer = asyncio.create_task(self._delayed_flush())
        return await fut

    async def _delayed_flush(self) -> None:
        try:
            await asyncio.sleep(self._max_wait)
        except asyncio.CancelledError:
            return
        async with self._lock:
            await self._flush_locked()

    async def _flush_locked(self) -> None:
        if not self._pending:
            return
        batch = self._pending
        self._pending = []
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

        items = [b[0] for b in batch]
        try:
            results = await self._process(items)
            for (_, fut), res in zip(batch, results):
                if not fut.done():
                    fut.set_result(res)
        except Exception as e:
            logger.error("batch process failed: %s", e)
            for _, fut in batch:
                if not fut.done():
                    fut.set_exception(e)
