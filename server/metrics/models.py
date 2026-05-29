"""Data models for metrics tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class RequestMetrics:
    """Metrics for a single request."""

    # Identifiers
    request_id: str
    correlation_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

    # Model info
    model: str = ""
    intent: str = ""

    # Token counts
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    # Timing (milliseconds)
    time_to_first_token_ms: float = 0.0
    total_time_ms: float = 0.0
    tokens_per_second: float = 0.0

    # Tool usage
    tool_calls_count: int = 0
    tool_names: list[str] = field(default_factory=list)

    # RAG usage
    rag_chunks_used: int = 0

    # Validation
    validation_warnings: list[str] = field(default_factory=list)

    # Status
    success: bool = True
    error_message: str = ""

    # Additional context
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage/API response."""
        return {
            "request_id": self.request_id,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp.isoformat(),
            "model": self.model,
            "intent": self.intent,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "time_to_first_token_ms": round(self.time_to_first_token_ms, 2),
            "total_time_ms": round(self.total_time_ms, 2),
            "tokens_per_second": round(self.tokens_per_second, 2),
            "tool_calls_count": self.tool_calls_count,
            "tool_names": self.tool_names,
            "rag_chunks_used": self.rag_chunks_used,
            "validation_warnings": self.validation_warnings,
            "success": self.success,
            "error_message": self.error_message,
            "metadata": self.metadata,
        }

    def calculate_derived_metrics(self) -> None:
        """Calculate derived metrics like tokens_per_second."""
        self.total_tokens = self.input_tokens + self.output_tokens
        if self.total_time_ms > 0 and self.output_tokens > 0:
            self.tokens_per_second = (self.output_tokens / self.total_time_ms) * 1000


@dataclass
class AggregatedMetrics:
    """Aggregated metrics for analysis."""

    model: str
    period_start: datetime
    period_end: datetime

    # Counts
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0

    # Token stats
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    avg_input_tokens: float = 0.0
    avg_output_tokens: float = 0.0

    # Timing stats (ms)
    avg_ttft_ms: float = 0.0
    avg_total_time_ms: float = 0.0
    p50_total_time_ms: float = 0.0
    p95_total_time_ms: float = 0.0
    p99_total_time_ms: float = 0.0

    # Performance
    avg_tokens_per_second: float = 0.0

    # Tool usage
    total_tool_calls: int = 0
    avg_tool_calls_per_request: float = 0.0

    # Intent distribution
    intent_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "model": self.model,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": round(self.successful_requests / max(1, self.total_requests) * 100, 2),
            "tokens": {
                "total_input": self.total_input_tokens,
                "total_output": self.total_output_tokens,
                "avg_input": round(self.avg_input_tokens, 1),
                "avg_output": round(self.avg_output_tokens, 1),
            },
            "latency_ms": {
                "avg_ttft": round(self.avg_ttft_ms, 2),
                "avg_total": round(self.avg_total_time_ms, 2),
                "p50": round(self.p50_total_time_ms, 2),
                "p95": round(self.p95_total_time_ms, 2),
                "p99": round(self.p99_total_time_ms, 2),
            },
            "performance": {
                "avg_tokens_per_second": round(self.avg_tokens_per_second, 2),
            },
            "tool_usage": {
                "total_calls": self.total_tool_calls,
                "avg_per_request": round(self.avg_tool_calls_per_request, 2),
            },
            "intent_distribution": self.intent_counts,
        }
