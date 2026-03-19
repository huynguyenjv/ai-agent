"""
Health, status, and observability endpoints.

Includes: /health, /v1/health, /v1/models, /cache/stats,
          /v1/agent/status, /v1/agent/metrics, /v1/agent/events/stream,
          /v1/events (Tabby)
"""

import asyncio
import json
import time
import queue as queue_mod
from typing import Optional

import structlog
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from agent.events import Event
from ..dependencies import (
    graph_orchestrator, rate_limiter, _executor,
    run_in_executor,
)
from ..schemas import HealthResponse, ModelInfo, ModelsResponse

logger = structlog.get_logger()

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check the health of all services (non-blocking)."""
    vllm_healthy = False
    qdrant_healthy = False
    index_stats = None

    if graph_orchestrator:
        # Run sync health checks in thread pool to avoid blocking the event loop
        loop = asyncio.get_running_loop()

        vllm_healthy = await loop.run_in_executor(None, graph_orchestrator.vllm.health_check)

        try:
            stats = await loop.run_in_executor(None, graph_orchestrator.rag.get_stats)
            # Qdrant status có thể là "green", "yellow", hoặc tên khác tùy version
            qdrant_healthy = stats.status in ("green", "yellow", "Green", "Yellow") or stats.total_points >= 0
            index_stats = {
                "collection": stats.collection_name,
                "total_points": stats.total_points,
                "status": stats.status,
                "type_distribution": stats.type_distribution,
            }
        except Exception as e:
            logger.warning("Qdrant health check failed", error=str(e))
            # Try direct connection check
            try:
                collections = await loop.run_in_executor(
                    None, graph_orchestrator.rag.qdrant.get_collections
                )
                qdrant_healthy = True
                index_stats = {"collection": "java_codebase", "total_points": 0, "status": "connected"}
            except Exception as e:
                logger.warning("Qdrant direct connection check failed", error=str(e))

    status = "healthy" if (vllm_healthy and qdrant_healthy) else "degraded"

    # Check Redis/cache health
    from utils.cache_service import get_cache_service
    cache = get_cache_service()
    redis_healthy = cache.is_redis_connected
    cache_backend = cache.backend_name

    return HealthResponse(
        status=status,
        vllm_healthy=vllm_healthy,
        qdrant_healthy=qdrant_healthy,
        redis_healthy=redis_healthy,
        cache_backend=cache_backend,
        index_stats=index_stats,
    )


@router.get("/cache/stats")
async def cache_stats():
    """Get cache service statistics."""
    from utils.cache_service import get_cache_service
    cache = get_cache_service()
    loop = asyncio.get_running_loop()
    stats = await loop.run_in_executor(_executor, cache.get_cache_stats)
    
    # Add rate limiter info
    if rate_limiter:
        stats["rate_limiter"] = {
            "backend": "redis" if rate_limiter.redis_client else "memory",
            "requests_per_window": rate_limiter.requests_per_window,
            "window_seconds": rate_limiter.window_seconds,
        }
    
    return stats


# ── Tabby-specific ──────────────────────────────────────────────────

@router.get("/v1/health")
async def tabby_health():
    """Health check endpoint for Tabby."""
    return {"status": "ok"}


@router.post("/v1/events")
async def tabby_events(request: dict):
    """Receive events from Tabby (telemetry, etc.)."""
    logger.debug("Received Tabby event", event=request)
    return {"status": "ok"}


# ── OpenAI models ───────────────────────────────────────────────────

@router.get("/v1/models", response_model=ModelsResponse)
async def list_models():
    """List available models (OpenAI-compatible)."""
    return ModelsResponse(
        data=[
            ModelInfo(id="ai-agent", created=int(time.time())),
            ModelInfo(id="ai-agent-test-generator", created=int(time.time())),
        ]
    )


# ── Agent Status & Metrics (Phase 4 observability) ──────────────────

@router.get("/v1/agent/status")
async def agent_status():
    """Get agent pipeline status and metrics (Phase 4 observability)."""
    if not graph_orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")

    metrics_data = graph_orchestrator.metrics.get_metrics() if graph_orchestrator.metrics else {}
    event_bus = graph_orchestrator.event_bus

    return {
        "status": "running",
        "event_bus": {
            "total_events": event_bus.event_count,
            "subscribers": event_bus.subscriber_count,
        },
        "metrics": metrics_data,
    }


@router.get("/v1/agent/metrics")
async def agent_metrics():
    """Get detailed agent metrics (Phase 4 observability)."""
    if not graph_orchestrator or not graph_orchestrator.metrics:
        raise HTTPException(status_code=503, detail="Service not initialized")

    return graph_orchestrator.metrics.get_metrics()


@router.get("/v1/agent/events/stream")
async def agent_event_stream():
    """SSE stream of real-time agent events (Phase 4).

    Clients connect via EventSource / SSE and receive live events
    as the agent processes requests.
    """
    if not graph_orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")

    event_queue: queue_mod.Queue = queue_mod.Queue(maxsize=100)
    keepalive_counter = 0  # Track iterations for keepalive spacing

    def _forward_to_queue(event: Event) -> None:
        try:
            event_queue.put_nowait({
                "type": event.type.value,
                "data": event.data,
                "timestamp": event.timestamp,
                "source": event.source,
                "plan_id": event.plan_id,
            })
        except queue_mod.Full:
            logger.debug("Event queue full, dropping event")

    # Subscribe to all events
    graph_orchestrator.event_bus.subscribe_all(_forward_to_queue)

    async def _event_generator():
        nonlocal keepalive_counter
        try:
            while True:
                try:
                    evt = event_queue.get_nowait()
                    keepalive_counter = 0
                    yield f"data: {json.dumps(evt)}\n\n"
                except queue_mod.Empty:
                    keepalive_counter += 1
                    # Send keepalive every ~15s (15 iterations × 1s sleep)
                    if keepalive_counter >= 15:
                        yield ": keepalive\n\n"
                        keepalive_counter = 0
                    await asyncio.sleep(1)
        finally:
            # Cleanup: thread-safe unsubscribe via EventBus lock
            try:
                with graph_orchestrator.event_bus._lock:
                    handlers = graph_orchestrator.event_bus._wildcard_handlers
                    if _forward_to_queue in handlers:
                        handlers.remove(_forward_to_queue)
            except Exception:
                logger.debug("Event handler cleanup failed (non-critical)")

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )
