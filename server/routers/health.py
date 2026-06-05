"""Health Checks — Phase 11.5.

- /health/live   : liveness — process is up (no dependency checks)
- /health/ready  : readiness — vLLM reachable (the hard dependency)
- /health/deep   : full dependency probe (vLLM, Qdrant, Postgres) + circuit state
"""

from __future__ import annotations

import asyncio
import logging
import os
import time

from fastapi import APIRouter, Request, Response

from server.circuit_breaker import all_circuit_stats

logger = logging.getLogger("server.health")

router = APIRouter()

VERSION = "2.0.0"

# Per-dependency probe timeout so /health/deep never hangs on a dead backend.
PROBE_TIMEOUT = float(os.environ.get("HEALTH_PROBE_TIMEOUT", "3"))


@router.get("/health/live")
async def liveness() -> dict:
    """Liveness probe — process is running. No external calls."""
    return {"status": "ok", "version": VERSION}


async def _check_vllm(req: Request) -> dict:
    client = getattr(req.app.state, "vllm_client", None)
    if client is None:
        return {"status": "error", "detail": "vLLM client not initialized"}
    start = time.monotonic()
    try:
        await asyncio.wait_for(client.models.list(), timeout=PROBE_TIMEOUT)
        return {"status": "ok", "latency_ms": round((time.monotonic() - start) * 1000, 1)}
    except (Exception, asyncio.TimeoutError) as e:
        return {"status": "error", "detail": str(e)[:200] or "timeout"}


async def _check_qdrant(req: Request) -> dict:
    qdrant = getattr(req.app.state, "qdrant", None)
    if qdrant is None:
        return {"status": "disabled", "detail": "RAG off (ENABLE_RAG=false)"}
    start = time.monotonic()
    try:
        await asyncio.wait_for(qdrant._client.get_collections(), timeout=PROBE_TIMEOUT)
        return {"status": "ok", "latency_ms": round((time.monotonic() - start) * 1000, 1)}
    except (Exception, asyncio.TimeoutError) as e:
        return {"status": "error", "detail": str(e)[:200] or "timeout"}


def _check_redis() -> dict:
    from server.redis_client import redis_configured, get_redis

    if not redis_configured():
        return {"status": "disabled", "detail": "REDIS_URL not set (in-memory)"}
    client = get_redis()
    if client is None:
        return {"status": "error", "detail": "Redis configured but unreachable"}
    start = time.monotonic()
    try:
        client.ping()
        return {"status": "ok", "latency_ms": round((time.monotonic() - start) * 1000, 1)}
    except Exception as e:
        return {"status": "error", "detail": str(e)[:200]}


def _check_postgres() -> dict:
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        return {"status": "disabled", "detail": "DATABASE_URL not set (SQLite fallback)"}
    try:
        import psycopg

        start = time.monotonic()
        with psycopg.connect(database_url, connect_timeout=3) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
        return {"status": "ok", "latency_ms": round((time.monotonic() - start) * 1000, 1)}
    except Exception as e:
        return {"status": "error", "detail": str(e)[:200]}


@router.get("/health/ready")
async def readiness(req: Request, response: Response) -> dict:
    """Readiness probe — vLLM (the hard dependency) must be reachable."""
    vllm = await _check_vllm(req)
    ready = vllm["status"] == "ok"
    if not ready:
        response.status_code = 503
    return {"status": "ok" if ready else "not_ready", "vllm": vllm}


@router.get("/health/deep")
async def deep_health(req: Request, response: Response) -> dict:
    """Full dependency probe + circuit-breaker states."""
    checks = {
        "vllm": await _check_vllm(req),
        "qdrant": await _check_qdrant(req),
        "postgres": _check_postgres(),
        "redis": _check_redis(),
    }
    # Only 'error' is unhealthy; 'disabled' is an intentional state.
    healthy = all(c["status"] != "error" for c in checks.values())
    if not healthy:
        response.status_code = 503
    return {
        "status": "ok" if healthy else "degraded",
        "version": VERSION,
        "checks": checks,
        "circuits": all_circuit_stats(),
    }
