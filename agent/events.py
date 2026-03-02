"""
Event Bus — decoupled publish/subscribe for agent lifecycle events.

Components publish events (state changes, validation results, metrics)
without knowing who consumes them.  Consumers register handlers
that react to specific event types.

Thread-safe and synchronous by default; async adapters can wrap handlers.

Usage::

    bus = EventBus()
    bus.subscribe(EventType.STATE_CHANGED, my_handler)
    bus.publish(Event(EventType.STATE_CHANGED, data={...}))
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

import structlog

logger = structlog.get_logger()


# ── Event types ──────────────────────────────────────────────────────

class EventType(str, Enum):
    """All event types emitted by the agent."""

    # Lifecycle
    STATE_CHANGED = "state_changed"
    PLAN_CREATED = "plan_created"
    STEP_STARTED = "step_started"
    STEP_COMPLETED = "step_completed"
    STEP_FAILED = "step_failed"

    # Generation
    GENERATION_STARTED = "generation_started"
    GENERATION_COMPLETED = "generation_completed"
    TOKEN_GENERATED = "token_generated"  # For streaming

    # Validation & Repair
    VALIDATION_COMPLETED = "validation_completed"
    REPAIR_STARTED = "repair_started"
    REPAIR_COMPLETED = "repair_completed"

    # Context
    CONTEXT_RETRIEVED = "context_retrieved"

    # Session
    SESSION_CREATED = "session_created"
    SESSION_EXPIRED = "session_expired"

    # Metrics
    METRIC_RECORDED = "metric_recorded"

    # Errors
    ERROR_OCCURRED = "error_occurred"


# ── Event data ───────────────────────────────────────────────────────

@dataclass
class Event:
    """A single event emitted by the system."""

    type: EventType
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    source: str = ""           # Component that emitted the event
    plan_id: Optional[str] = None
    session_id: Optional[str] = None

    def __str__(self) -> str:
        return f"Event({self.type.value}, source={self.source})"


# ── Handler type ─────────────────────────────────────────────────────

EventHandler = Callable[[Event], None]


# ── Event Bus ────────────────────────────────────────────────────────

class EventBus:
    """Thread-safe publish/subscribe event bus.

    Supports:
      - Subscribe to specific event types
      - Subscribe to ALL events (wildcard)
      - Unsubscribe
      - Publish events synchronously
    """

    def __init__(self) -> None:
        self._handlers: dict[EventType, list[EventHandler]] = {}
        self._wildcard_handlers: list[EventHandler] = []
        self._lock = threading.Lock()
        self._event_count = 0

    def subscribe(
        self,
        event_type: EventType,
        handler: EventHandler,
    ) -> None:
        """Subscribe to a specific event type."""
        with self._lock:
            if event_type not in self._handlers:
                self._handlers[event_type] = []
            self._handlers[event_type].append(handler)

    def subscribe_all(self, handler: EventHandler) -> None:
        """Subscribe to ALL event types (wildcard)."""
        with self._lock:
            self._wildcard_handlers.append(handler)

    def unsubscribe(
        self,
        event_type: EventType,
        handler: EventHandler,
    ) -> None:
        """Unsubscribe from a specific event type."""
        with self._lock:
            handlers = self._handlers.get(event_type, [])
            if handler in handlers:
                handlers.remove(handler)

    def publish(self, event: Event) -> None:
        """Publish an event to all subscribers.

        Handlers are called synchronously.  If a handler raises,
        the error is logged and the next handler is still called.
        """
        with self._lock:
            self._event_count += 1
            handlers = list(self._handlers.get(event.type, []))
            wildcards = list(self._wildcard_handlers)

        for handler in handlers + wildcards:
            try:
                handler(event)
            except Exception as e:
                logger.warning(
                    "Event handler failed",
                    event_type=event.type.value,
                    handler=handler.__name__ if hasattr(handler, "__name__") else str(handler),
                    error=str(e),
                )

    @property
    def event_count(self) -> int:
        """Total events published."""
        return self._event_count

    @property
    def subscriber_count(self) -> int:
        """Total subscribers across all event types."""
        with self._lock:
            return sum(len(h) for h in self._handlers.values()) + len(self._wildcard_handlers)

    def clear(self) -> None:
        """Remove all subscribers."""
        with self._lock:
            self._handlers.clear()
            self._wildcard_handlers.clear()


# ── Global singleton ─────────────────────────────────────────────────

_global_bus: Optional[EventBus] = None
_bus_lock = threading.Lock()


def get_event_bus() -> EventBus:
    """Get or create the global event bus singleton."""
    global _global_bus
    if _global_bus is None:
        with _bus_lock:
            if _global_bus is None:
                _global_bus = EventBus()
    return _global_bus


def reset_event_bus() -> None:
    """Reset the global event bus (for testing)."""
    global _global_bus
    with _bus_lock:
        if _global_bus:
            _global_bus.clear()
        _global_bus = None
