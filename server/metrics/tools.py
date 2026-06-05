"""Tool Usage Analytics — Phase 17.5.

Per-tool success/failure rates, call counts, optional latency, and selection
frequency. Fed from the tool-result validation path in chat. (Tools run
client-side, so latency may be unavailable.)
"""

from __future__ import annotations

import statistics
import threading
from collections import defaultdict


class ToolAnalytics:
    def __init__(self):
        self._lock = threading.Lock()
        self._stats: dict[str, dict] = defaultdict(
            lambda: {"calls": 0, "success": 0, "failure": 0, "latencies": []}
        )

    def record(self, tool_name: str, *, success: bool, latency_ms: float | None = None) -> None:
        name = tool_name or "unknown"
        with self._lock:
            s = self._stats[name]
            s["calls"] += 1
            if success:
                s["success"] += 1
            else:
                s["failure"] += 1
            if latency_ms is not None:
                s["latencies"].append(latency_ms)

    def snapshot(self) -> dict:
        with self._lock:
            total = sum(s["calls"] for s in self._stats.values())
            tools = {}
            for name, s in self._stats.items():
                calls = s["calls"] or 1
                lat = s["latencies"]
                tools[name] = {
                    "calls": s["calls"],
                    "success_rate": round(s["success"] / calls, 4),
                    "failure_rate": round(s["failure"] / calls, 4),
                    "avg_latency_ms": round(statistics.mean(lat), 1) if lat else None,
                }
            most_used = max(self._stats.items(), key=lambda kv: kv[1]["calls"])[0] if self._stats else None
            return {"total_calls": total, "tools": tools, "most_used": most_used}


_analytics: ToolAnalytics | None = None
_lock = threading.Lock()


def get_tool_analytics() -> ToolAnalytics:
    global _analytics
    if _analytics is None:
        with _lock:
            if _analytics is None:
                _analytics = ToolAnalytics()
    return _analytics


def reset_tool_analytics() -> None:
    global _analytics
    with _lock:
        _analytics = None
