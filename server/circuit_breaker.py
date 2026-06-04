"""Circuit Breaker — Phase 11.3.

Protects external dependencies (vLLM, Qdrant, Postgres). When failures exceed a
threshold the circuit opens and calls fail fast (CircuitOpenError) until a
recovery timeout elapses; it then half-opens to probe recovery.

States: closed -> open -> half-open -> (closed | open)
"""

from __future__ import annotations

import logging
import threading
import time
from enum import Enum
from typing import Awaitable, Callable, TypeVar

logger = logging.getLogger("server.circuit_breaker")

T = TypeVar("T")


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(Exception):
    """Raised when a call is rejected because the circuit is open."""

    def __init__(self, name: str):
        super().__init__(f"Circuit '{name}' is open")
        self.name = name


class CircuitBreaker:
    """Thread-safe circuit breaker.

    Args:
        name: identifier (for logs/metrics)
        failure_threshold: consecutive failures before opening
        recovery_timeout: seconds to wait before half-opening
        half_open_max: successes in half-open required to close
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max: int = 3,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max = half_open_max

        self._lock = threading.Lock()
        self._state = CircuitState.CLOSED
        self._failures = 0
        self._half_open_successes = 0
        self._opened_at = 0.0

    @property
    def state(self) -> CircuitState:
        with self._lock:
            return self._state

    def _transition_if_recovered(self) -> None:
        """Move OPEN -> HALF_OPEN once recovery_timeout has elapsed. Lock held."""
        if (
            self._state == CircuitState.OPEN
            and time.monotonic() - self._opened_at >= self.recovery_timeout
        ):
            self._state = CircuitState.HALF_OPEN
            self._half_open_successes = 0
            logger.info("circuit '%s' -> half_open (probing)", self.name)

    def allow(self) -> bool:
        """Whether a call may proceed right now."""
        with self._lock:
            self._transition_if_recovered()
            return self._state != CircuitState.OPEN

    def record_success(self) -> None:
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._half_open_successes += 1
                if self._half_open_successes >= self.half_open_max:
                    self._state = CircuitState.CLOSED
                    self._failures = 0
                    logger.info("circuit '%s' -> closed (recovered)", self.name)
            else:
                self._failures = 0

    def record_failure(self) -> None:
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                # A failure while probing re-opens immediately.
                self._state = CircuitState.OPEN
                self._opened_at = time.monotonic()
                logger.warning("circuit '%s' -> open (probe failed)", self.name)
                return
            self._failures += 1
            if self._failures >= self.failure_threshold:
                self._state = CircuitState.OPEN
                self._opened_at = time.monotonic()
                logger.warning(
                    "circuit '%s' -> open (%d failures)", self.name, self._failures
                )

    async def call(self, fn: Callable[[], Awaitable[T]]) -> T:
        """Run an async zero-arg fn through the breaker.

        Raises CircuitOpenError if the circuit is open.
        """
        if not self.allow():
            raise CircuitOpenError(self.name)
        try:
            result = await fn()
        except Exception:
            self.record_failure()
            raise
        self.record_success()
        return result

    def stats(self) -> dict:
        with self._lock:
            return {
                "name": self.name,
                "state": self._state.value,
                "failures": self._failures,
                "failure_threshold": self.failure_threshold,
            }


# Registry of named breakers
_breakers: dict[str, CircuitBreaker] = {}
_registry_lock = threading.Lock()


def get_circuit_breaker(
    name: str,
    failure_threshold: int | None = None,
    recovery_timeout: float | None = None,
) -> CircuitBreaker:
    """Get or create a named circuit breaker (singleton per name)."""
    import os

    with _registry_lock:
        cb = _breakers.get(name)
        if cb is None:
            cb = CircuitBreaker(
                name,
                failure_threshold=failure_threshold
                if failure_threshold is not None
                else int(os.environ.get("CIRCUIT_FAILURE_THRESHOLD", "5")),
                recovery_timeout=recovery_timeout
                if recovery_timeout is not None
                else float(os.environ.get("CIRCUIT_RECOVERY_TIMEOUT", "60")),
            )
            _breakers[name] = cb
        return cb


def all_circuit_stats() -> list[dict]:
    with _registry_lock:
        return [cb.stats() for cb in _breakers.values()]
