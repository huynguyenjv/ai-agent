"""Retry Strategy — Phase 11.6.

Shared async retry with exponential backoff + jitter for external calls
(vLLM, Qdrant, Postgres, HTTP). Centralizes what was previously ad-hoc per
call site.
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Awaitable, Callable, Sequence, TypeVar

logger = logging.getLogger("server.retry")

T = TypeVar("T")

# Exceptions that are generally safe/worth retrying (transient I/O).
DEFAULT_RETRYABLE: tuple[type[BaseException], ...] = (
    ConnectionError,
    TimeoutError,
    asyncio.TimeoutError,
    OSError,
)


def compute_delay(attempt: int, base_delay: float, max_delay: float, jitter: bool) -> float:
    """Exponential backoff delay for a 0-based attempt index, optionally jittered."""
    delay = min(max_delay, base_delay * (2 ** attempt))
    if jitter:
        # Full jitter: random in [0, delay]
        delay = random.uniform(0, delay)
    return delay


async def retry_async(
    fn: Callable[[], Awaitable[T]],
    *,
    max_retries: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 10.0,
    jitter: bool = True,
    retryable: Sequence[type[BaseException]] = DEFAULT_RETRYABLE,
    on_retry: Callable[[int, BaseException], None] | None = None,
) -> T:
    """Call an async zero-arg function with retries.

    Retries only on `retryable` exceptions. Re-raises the last exception after
    `max_retries` attempts, or immediately for non-retryable exceptions.
    """
    retryable_t = tuple(retryable)
    last_exc: BaseException | None = None

    for attempt in range(max_retries):
        try:
            return await fn()
        except retryable_t as e:
            last_exc = e
            if attempt >= max_retries - 1:
                break
            delay = compute_delay(attempt, base_delay, max_delay, jitter)
            if on_retry:
                on_retry(attempt, e)
            logger.warning(
                "retry %d/%d after %.2fs: %s", attempt + 1, max_retries, delay, e
            )
            await asyncio.sleep(delay)

    assert last_exc is not None
    raise last_exc


def is_retryable(exc: BaseException, retryable: Sequence[type[BaseException]] = DEFAULT_RETRYABLE) -> bool:
    return isinstance(exc, tuple(retryable))
