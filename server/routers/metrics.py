"""Metrics API router."""

from __future__ import annotations

from fastapi import APIRouter, Query
from fastapi.responses import Response

from server.metrics import get_metrics_counter
from server.metrics.prometheus import get_metrics as get_prometheus_metrics, get_content_type

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.get("")
async def get_metrics_summary(
    hours: int = Query(default=24, ge=1, le=168, description="Hours to look back"),
):
    """Get aggregated metrics summary for all models."""
    counter = get_metrics_counter()
    comparison = counter.compare_models(hours=hours)

    return {
        "period_hours": hours,
        "models": comparison,
    }


@router.get("/recent")
async def get_recent_metrics(
    limit: int = Query(default=50, ge=1, le=500, description="Number of records"),
    model: str | None = Query(default=None, description="Filter by model"),
):
    """Get recent request metrics."""
    counter = get_metrics_counter()
    recent = counter.get_recent(limit=limit, model=model)

    return {
        "count": len(recent),
        "requests": recent,
    }


@router.get("/compare")
async def compare_models(
    hours: int = Query(default=24, ge=1, le=168, description="Hours to look back"),
):
    """Compare metrics across models."""
    counter = get_metrics_counter()
    comparison = counter.compare_models(hours=hours)

    # Build comparison table
    models = list(comparison.keys())
    if not models:
        return {"message": "No metrics data available"}

    table = {
        "models": models,
        "metrics": {},
    }

    # Extract comparable metrics
    metric_keys = [
        ("total_requests", "Total Requests"),
        ("success_rate", "Success Rate (%)"),
        ("tokens.avg_input", "Avg Input Tokens"),
        ("tokens.avg_output", "Avg Output Tokens"),
        ("latency_ms.avg_ttft", "Avg TTFT (ms)"),
        ("latency_ms.avg_total", "Avg Total Time (ms)"),
        ("latency_ms.p95", "P95 Latency (ms)"),
        ("performance.avg_tokens_per_second", "Avg Tokens/sec"),
        ("tool_usage.avg_per_request", "Avg Tool Calls"),
    ]

    for key_path, label in metric_keys:
        values = []
        for model in models:
            data = comparison[model]
            # Navigate nested keys
            value = data
            for k in key_path.split("."):
                value = value.get(k, 0) if isinstance(value, dict) else 0
            values.append(value)
        table["metrics"][label] = dict(zip(models, values))

    return table


@router.delete("/clear")
async def clear_old_metrics(
    days: int = Query(default=30, ge=1, le=365, description="Delete metrics older than X days"),
):
    """Clear old metrics data."""
    counter = get_metrics_counter()
    deleted = counter.clear_old(days=days)

    return {
        "deleted": deleted,
        "message": f"Cleared {deleted} metrics older than {days} days",
    }


@router.get("/prometheus")
async def prometheus_metrics():
    """Prometheus metrics endpoint for scraping."""
    return Response(
        content=get_prometheus_metrics(),
        media_type=get_content_type(),
    )
