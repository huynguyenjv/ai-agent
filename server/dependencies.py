"""
Shared dependencies, global state, and utility functions for the server.

All routers import from here instead of maintaining their own globals.
"""

import asyncio
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, TypeVar, Callable, Any

import structlog
from fastapi import HTTPException

from agent.graph_adapter import GraphOrchestrator
from agent.tool_orchestrator import ToolOrchestrator
from agent.metrics import MetricsCollector
from indexer.build_index import IndexBuilder
from utils.rate_limiter import RateLimiter
from .session import SessionManager

logger = structlog.get_logger()

# ============================================================================
# Global Instances
# ============================================================================

graph_orchestrator: Optional[GraphOrchestrator] = None
index_builder: Optional[IndexBuilder] = None
metrics_collector: Optional[MetricsCollector] = None
session_manager: SessionManager = SessionManager()
rate_limiter: Optional[RateLimiter] = None
tool_orchestrator: ToolOrchestrator = ToolOrchestrator()

# Thread pool for running blocking operations
# Use limited workers to prevent resource exhaustion
_executor: Optional[ThreadPoolExecutor] = None
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "8"))

# Rate limiting configuration
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "10"))  # requests per window
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))  # window in seconds
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "300"))  # max request time in seconds

# Track active executor tasks for observability
_active_tasks: int = 0
_active_tasks_lock = threading.Lock()


# ============================================================================
# Factory Functions
# ============================================================================

def create_index_builder() -> IndexBuilder:
    """Create and configure the index builder."""
    return IndexBuilder(
        qdrant_host=os.getenv("QDRANT_HOST", "localhost"),
        qdrant_port=int(os.getenv("QDRANT_PORT", "6333")),
        collection_name=os.getenv("QDRANT_COLLECTION", "java_codebase"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2-onnx"),
    )


# ============================================================================
# Utility Functions
# ============================================================================

T = TypeVar('T')

async def run_in_executor(func: Callable[..., T], *args: Any) -> T:
    """Run a blocking function in the thread pool with timeout."""
    global _active_tasks
    loop = asyncio.get_running_loop()

    with _active_tasks_lock:
        _active_tasks += 1
        current = _active_tasks

    logger.debug(
        "Submitting task to executor",
        func=func.__name__,
        active_tasks=current,
        max_workers=MAX_WORKERS,
    )

    try:
        return await asyncio.wait_for(
            loop.run_in_executor(_executor, func, *args),
            timeout=REQUEST_TIMEOUT
        )
    except asyncio.TimeoutError:
        logger.error(
            "Request timed out",
            func=func.__name__,
            timeout=REQUEST_TIMEOUT,
            active_tasks=_active_tasks,
        )
        raise HTTPException(
            status_code=504,
            detail=f"Request timed out after {REQUEST_TIMEOUT} seconds"
        )
    finally:
        with _active_tasks_lock:
            _active_tasks -= 1


def check_rate_limit(client_ip: str) -> tuple[bool, int]:
    """Check if client has exceeded rate limit.

    Returns:
        (allowed: bool, remaining: int) - Whether request is allowed and remaining requests

    NOTE: This is a sync function that may do Redis I/O.
    Must be called via run_in_executor() from async context.
    """
    if not rate_limiter:
        return True, RATE_LIMIT_REQUESTS  # Allow if rate limiter not initialized
    
    allowed = rate_limiter.check_rate_limit(client_ip)
    remaining = rate_limiter.get_remaining_requests(client_ip)
    
    return allowed, remaining


def _get_orchestrator() -> Optional[GraphOrchestrator]:
    """Get the active graph_orchestrator."""
    return graph_orchestrator
