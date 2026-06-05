"""Connection Pooling — Phase 19.2.

Builds the vLLM client with an explicitly bounded httpx connection pool to avoid
socket exhaustion under load. (Postgres uses psycopg's pool in server/metrics;
Redis uses redis-py's pool in server/redis_client — both already pooled.)
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger("server.connections")

VLLM_MAX_CONNECTIONS = int(os.environ.get("VLLM_MAX_CONNECTIONS", "100"))
VLLM_MAX_KEEPALIVE = int(os.environ.get("VLLM_MAX_KEEPALIVE", "20"))
VLLM_TIMEOUT = float(os.environ.get("VLLM_TIMEOUT", "120"))


def pool_limits():
    """httpx.Limits for the vLLM client (also used to assert config in tests)."""
    import httpx

    return httpx.Limits(
        max_connections=VLLM_MAX_CONNECTIONS,
        max_keepalive_connections=VLLM_MAX_KEEPALIVE,
    )


def build_vllm_client(base_url: str, api_key: str = "not-needed"):
    """Create an AsyncOpenAI client backed by a bounded, keep-alive httpx pool."""
    import httpx
    from openai import AsyncOpenAI

    http_client = httpx.AsyncClient(
        limits=pool_limits(),
        timeout=httpx.Timeout(VLLM_TIMEOUT),
    )
    logger.info(
        "vLLM client pool: max_connections=%d keepalive=%d timeout=%.0fs",
        VLLM_MAX_CONNECTIONS, VLLM_MAX_KEEPALIVE, VLLM_TIMEOUT,
    )
    return AsyncOpenAI(base_url=base_url, api_key=api_key, http_client=http_client)
