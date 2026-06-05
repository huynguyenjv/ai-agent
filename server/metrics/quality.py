"""Online Quality Metrics — Phase 16.5.

Tracks production quality signals: user satisfaction, task completion, retry
rate, code-acceptance rate, response length. Complements raw perf metrics
(latency/tokens) with quality-of-output signals.
"""

from __future__ import annotations

import statistics
import threading
from collections import defaultdict


class QualityTracker:
    """In-memory rolling quality counters (per-intent + overall)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._total = 0
        self._satisfied = 0
        self._completed = 0
        self._retried = 0
        self._code_offered = 0
        self._code_accepted = 0
        self._lengths: list[int] = []
        self._by_intent: dict[str, dict[str, int]] = defaultdict(
            lambda: {"n": 0, "satisfied": 0, "retried": 0, "completed": 0}
        )

    def record(
        self,
        *,
        intent: str = "",
        satisfied: bool | None = None,
        completed: bool | None = None,
        retried: bool = False,
        code_offered: bool = False,
        code_accepted: bool = False,
        response_length: int | None = None,
    ) -> None:
        with self._lock:
            self._total += 1
            if satisfied:
                self._satisfied += 1
            if completed:
                self._completed += 1
            if retried:
                self._retried += 1
            if code_offered:
                self._code_offered += 1
            if code_accepted:
                self._code_accepted += 1
            if response_length is not None:
                self._lengths.append(response_length)

            bi = self._by_intent[intent or "unknown"]
            bi["n"] += 1
            if satisfied:
                bi["satisfied"] += 1
            if retried:
                bi["retried"] += 1
            if completed:
                bi["completed"] += 1

    def _rate(self, num: int, den: int) -> float:
        return round(num / den, 4) if den else 0.0

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "total": self._total,
                "satisfaction_rate": self._rate(self._satisfied, self._total),
                "task_completion_rate": self._rate(self._completed, self._total),
                "retry_rate": self._rate(self._retried, self._total),
                "code_acceptance_rate": self._rate(self._code_accepted, self._code_offered),
                "response_length": {
                    "avg": round(statistics.mean(self._lengths), 1) if self._lengths else 0,
                    "p50": round(statistics.median(self._lengths), 1) if self._lengths else 0,
                    "max": max(self._lengths) if self._lengths else 0,
                },
                "by_intent": {
                    k: {
                        "n": v["n"],
                        "satisfaction_rate": self._rate(v["satisfied"], v["n"]),
                        "retry_rate": self._rate(v["retried"], v["n"]),
                        "completion_rate": self._rate(v["completed"], v["n"]),
                    }
                    for k, v in self._by_intent.items()
                },
            }


_tracker: QualityTracker | None = None
_lock = threading.Lock()


def get_quality_tracker() -> QualityTracker:
    global _tracker
    if _tracker is None:
        with _lock:
            if _tracker is None:
                _tracker = QualityTracker()
    return _tracker


def reset_quality_tracker() -> None:
    global _tracker
    with _lock:
        _tracker = None
