"""
Metrics Collector — observability for the agent pipeline.

Collects timing, counts, and quality metrics across all agent runs.
Subscribes to the EventBus for automatic data collection.

Metrics are kept in-memory and exposed via ``get_metrics()`` for
the health/status API endpoint.

Usage::

    from agent.events import get_event_bus
    from agent.metrics import MetricsCollector

    collector = MetricsCollector(get_event_bus())
    # ... agent runs ...
    print(collector.get_metrics())
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import structlog

from .events import EventBus, Event, EventType

logger = structlog.get_logger()


# ── Metrics data ─────────────────────────────────────────────────────

@dataclass
class TimingStats:
    """Accumulated timing statistics for an operation."""

    count: int = 0
    total_ms: float = 0.0
    min_ms: float = float("inf")
    max_ms: float = 0.0

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.count if self.count > 0 else 0.0

    def record(self, duration_ms: float) -> None:
        self.count += 1
        self.total_ms += duration_ms
        if duration_ms < self.min_ms:
            self.min_ms = duration_ms
        if duration_ms > self.max_ms:
            self.max_ms = duration_ms

    def to_dict(self) -> dict:
        return {
            "count": self.count,
            "avg_ms": round(self.avg_ms, 1),
            "min_ms": round(self.min_ms, 1) if self.min_ms != float("inf") else 0,
            "max_ms": round(self.max_ms, 1),
            "total_ms": round(self.total_ms, 1),
        }


# ── Metrics Collector ────────────────────────────────────────────────

class MetricsCollector:
    """Collects and aggregates agent metrics via EventBus subscription.

    Automatically subscribes to lifecycle events and records:
      - Generation counts (success / failure)
      - Timing per step (retrieve, generate, validate, ...)
      - Repair rates
      - Token usage
      - Validation pass rates
    """

    def __init__(self, bus: Optional[EventBus] = None) -> None:
        self._lock = threading.Lock()
        self._started_at = time.time()

        # Counters
        self._generations_total = 0
        self._generations_success = 0
        self._generations_failed = 0
        self._repairs_total = 0
        self._repairs_success = 0
        self._validations_total = 0
        self._validations_passed = 0
        self._tokens_total = 0

        # Timing
        self._step_timings: dict[str, TimingStats] = defaultdict(TimingStats)
        self._generation_timing = TimingStats()

        # Track in-flight steps
        self._step_start_times: dict[str, float] = {}

        # Subscribe to events if bus provided
        if bus:
            self._subscribe(bus)

    def _subscribe(self, bus: EventBus) -> None:
        """Subscribe to relevant events."""
        bus.subscribe(EventType.GENERATION_STARTED, self._on_generation_started)
        bus.subscribe(EventType.GENERATION_COMPLETED, self._on_generation_completed)
        bus.subscribe(EventType.STEP_STARTED, self._on_step_started)
        bus.subscribe(EventType.STEP_COMPLETED, self._on_step_completed)
        bus.subscribe(EventType.STEP_FAILED, self._on_step_failed)
        bus.subscribe(EventType.VALIDATION_COMPLETED, self._on_validation_completed)
        bus.subscribe(EventType.REPAIR_STARTED, self._on_repair_started)
        bus.subscribe(EventType.REPAIR_COMPLETED, self._on_repair_completed)
        bus.subscribe(EventType.ERROR_OCCURRED, self._on_error)

    # ── Event handlers ───────────────────────────────────────────────

    def _on_generation_started(self, event: Event) -> None:
        with self._lock:
            self._generations_total += 1
            key = event.data.get("plan_id", "unknown")
            self._step_start_times[f"gen:{key}"] = time.time()

    def _on_generation_completed(self, event: Event) -> None:
        with self._lock:
            if event.data.get("success", True):
                self._generations_success += 1
            else:
                self._generations_failed += 1

            self._tokens_total += event.data.get("tokens_used", 0)

            key = f"gen:{event.data.get('plan_id', 'unknown')}"
            start = self._step_start_times.pop(key, None)
            if start:
                self._generation_timing.record((time.time() - start) * 1000)

    def _on_step_started(self, event: Event) -> None:
        step_action = event.data.get("action", "unknown")
        step_id = event.data.get("step_id", 0)
        key = f"{step_action}:{step_id}"
        with self._lock:
            self._step_start_times[key] = time.time()

    def _on_step_completed(self, event: Event) -> None:
        step_action = event.data.get("action", "unknown")
        step_id = event.data.get("step_id", 0)
        key = f"{step_action}:{step_id}"
        with self._lock:
            start = self._step_start_times.pop(key, None)
            if start:
                duration = (time.time() - start) * 1000
                self._step_timings[step_action].record(duration)

    def _on_step_failed(self, event: Event) -> None:
        # Also record as completed (for timing) + count failure
        self._on_step_completed(event)

    def _on_validation_completed(self, event: Event) -> None:
        with self._lock:
            self._validations_total += 1
            if event.data.get("passed", False):
                self._validations_passed += 1

    def _on_repair_started(self, event: Event) -> None:
        with self._lock:
            self._repairs_total += 1

    def _on_repair_completed(self, event: Event) -> None:
        with self._lock:
            if event.data.get("success", False):
                self._repairs_success += 1

    def _on_error(self, event: Event) -> None:
        with self._lock:
            self._generations_failed += 1

    # ── Manual recording (for components not on the bus) ─────────────

    def record_generation(self, success: bool, tokens: int = 0, duration_ms: float = 0) -> None:
        """Manually record a generation outcome."""
        with self._lock:
            self._generations_total += 1
            if success:
                self._generations_success += 1
            else:
                self._generations_failed += 1
            self._tokens_total += tokens
            if duration_ms > 0:
                self._generation_timing.record(duration_ms)

    def record_step(self, action: str, duration_ms: float) -> None:
        """Manually record a step timing."""
        with self._lock:
            self._step_timings[action].record(duration_ms)

    # ── Query ────────────────────────────────────────────────────────

    def get_metrics(self) -> dict:
        """Get all collected metrics as a dict (for API/health endpoint)."""
        with self._lock:
            uptime = time.time() - self._started_at
            return {
                "uptime_seconds": round(uptime, 1),
                "generations": {
                    "total": self._generations_total,
                    "success": self._generations_success,
                    "failed": self._generations_failed,
                    "success_rate": (
                        round(self._generations_success / self._generations_total, 3)
                        if self._generations_total > 0 else 0
                    ),
                },
                "validation": {
                    "total": self._validations_total,
                    "passed": self._validations_passed,
                    "pass_rate": (
                        round(self._validations_passed / self._validations_total, 3)
                        if self._validations_total > 0 else 0
                    ),
                },
                "repair": {
                    "total": self._repairs_total,
                    "success": self._repairs_success,
                    "success_rate": (
                        round(self._repairs_success / self._repairs_total, 3)
                        if self._repairs_total > 0 else 0
                    ),
                },
                "tokens": {
                    "total": self._tokens_total,
                    "avg_per_generation": (
                        round(self._tokens_total / self._generations_total)
                        if self._generations_total > 0 else 0
                    ),
                },
                "timing": {
                    "generation": self._generation_timing.to_dict(),
                    "steps": {
                        action: stats.to_dict()
                        for action, stats in self._step_timings.items()
                    },
                },
            }

    def reset(self) -> None:
        """Reset all metrics (for testing)."""
        with self._lock:
            self._started_at = time.time()
            self._generations_total = 0
            self._generations_success = 0
            self._generations_failed = 0
            self._repairs_total = 0
            self._repairs_success = 0
            self._validations_total = 0
            self._validations_passed = 0
            self._tokens_total = 0
            self._step_timings.clear()
            self._generation_timing = TimingStats()
            self._step_start_times.clear()
