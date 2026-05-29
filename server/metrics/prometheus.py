"""Prometheus metrics for AI Agent monitoring.

Exposes metrics at /metrics/prometheus endpoint.
"""

from __future__ import annotations

import os
import time

from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest, CONTENT_TYPE_LATEST

# =============================================================================
# Counters
# =============================================================================

REQUEST_COUNT = Counter(
    "ai_agent_requests_total",
    "Total requests processed",
    ["intent", "status"]
)

TOOL_CALLS = Counter(
    "ai_agent_tool_calls_total",
    "Tool calls made",
    ["tool_name"]
)

TOKEN_COUNT = Counter(
    "ai_agent_tokens_total",
    "Tokens processed",
    ["type", "model"]
)

FEEDBACK_COUNT = Counter(
    "ai_agent_feedback_total",
    "Feedback received",
    ["feedback_type", "rating_bucket"]
)

VERIFICATION_COUNT = Counter(
    "ai_agent_verification_total",
    "Verification results",
    ["result"]  # passed, failed, max_retries
)

# =============================================================================
# Histograms
# =============================================================================

REQUEST_LATENCY = Histogram(
    "ai_agent_request_duration_seconds",
    "Request latency distribution",
    ["intent"],
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30, 60, 120]
)

TTFT = Histogram(
    "ai_agent_ttft_seconds",
    "Time to first token",
    ["intent"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10]
)

RAG_LATENCY = Histogram(
    "ai_agent_rag_search_seconds",
    "RAG search latency",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1]
)

RAG_CHUNKS = Histogram(
    "ai_agent_rag_chunks_returned",
    "Number of RAG chunks returned",
    buckets=[0, 1, 2, 3, 5, 10, 20]
)

# =============================================================================
# Gauges
# =============================================================================

ACTIVE_REQUESTS = Gauge(
    "ai_agent_active_requests",
    "Currently active requests"
)

SESSION_COUNT = Gauge(
    "ai_agent_sessions_active",
    "Active sessions"
)

CACHE_SIZE = Gauge(
    "ai_agent_cache_size",
    "Cache entries",
    ["cache_type"]
)

# =============================================================================
# Info
# =============================================================================

VERSION_INFO = Info(
    "ai_agent",
    "AI Agent version info"
)

_version = os.environ.get("APP_VERSION", "2.0.0")
VERSION_INFO.info({"version": _version, "phase": "3"})

# =============================================================================
# Helper Functions
# =============================================================================

def record_request(intent: str, status: str, duration: float, ttft: float | None = None):
    """Record request metrics."""
    REQUEST_COUNT.labels(intent=intent, status=status).inc()
    REQUEST_LATENCY.labels(intent=intent).observe(duration)
    if ttft is not None:
        TTFT.labels(intent=intent).observe(ttft)


def record_tool_call(tool_name: str):
    """Record tool call."""
    TOOL_CALLS.labels(tool_name=tool_name).inc()


def record_tokens(input_tokens: int, output_tokens: int, model: str):
    """Record token usage."""
    TOKEN_COUNT.labels(type="input", model=model).inc(input_tokens)
    TOKEN_COUNT.labels(type="output", model=model).inc(output_tokens)


def record_feedback(feedback_type: str, rating: int):
    """Record user feedback."""
    if feedback_type == "thumbs":
        bucket = "positive" if rating > 0 else ("negative" if rating < 0 else "neutral")
    else:
        bucket = "5" if rating == 5 else ("4" if rating == 4 else ("3" if rating == 3 else "1-2"))
    FEEDBACK_COUNT.labels(feedback_type=feedback_type, rating_bucket=bucket).inc()


def record_verification(passed: bool, max_retries: bool = False):
    """Record verification result."""
    if max_retries:
        VERIFICATION_COUNT.labels(result="max_retries").inc()
    elif passed:
        VERIFICATION_COUNT.labels(result="passed").inc()
    else:
        VERIFICATION_COUNT.labels(result="failed").inc()


def record_rag_search(duration: float, chunks: int):
    """Record RAG search metrics."""
    RAG_LATENCY.observe(duration)
    RAG_CHUNKS.observe(chunks)


def get_metrics() -> bytes:
    """Generate Prometheus metrics output."""
    return generate_latest()


def get_content_type() -> str:
    """Get Prometheus content type."""
    return CONTENT_TYPE_LATEST
