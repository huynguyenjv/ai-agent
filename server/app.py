"""
FastAPI application factory and lifespan management.

This is the main app module — creates the FastAPI app, registers
middleware, and includes all routers.
"""

import asyncio
import os
import time

import structlog
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from agent.graph_adapter import create_graph_orchestrator
from utils.rate_limiter import RateLimiter
from . import dependencies as deps
from .routers import health, chat, test_gen, index, embeddings

# Load environment variables
load_dotenv()

logger = structlog.get_logger()


# ============================================================================
# Lifespan
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("Starting AI Agent server...", backend="langgraph")

    # Initialize thread pool for blocking operations
    deps._executor = ThreadPoolExecutor(max_workers=deps.MAX_WORKERS)
    logger.info("Thread pool initialized", max_workers=deps.MAX_WORKERS)

    # Initialize rate limiter
    deps.rate_limiter = RateLimiter(
        redis_host=os.getenv("REDIS_HOST", "localhost"),
        redis_port=int(os.getenv("REDIS_PORT", "6379")),
        redis_password=os.getenv("REDIS_PASSWORD") or None,
        redis_db=int(os.getenv("REDIS_DB", "0")),
        requests_per_window=deps.RATE_LIMIT_REQUESTS,
        window_seconds=deps.RATE_LIMIT_WINDOW,
    )

    # Initialize LangGraph graph_orchestrator
    deps.graph_orchestrator = await create_graph_orchestrator()
    deps.index_builder = deps.create_index_builder()

    # Phase 4: metrics collector
    deps.metrics_collector = deps.graph_orchestrator.metrics if deps.graph_orchestrator else None

    # Start background task for session cleanup
    cleanup_task = asyncio.create_task(_periodic_session_cleanup())

    logger.info("Server initialized successfully", backend="langgraph")

    yield

    # Cleanup
    logger.info("Shutting down server...")
    
    # Cancel cleanup task
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass

    # Close vLLM client connection
    if deps.graph_orchestrator and deps.graph_orchestrator.vllm:
        deps.graph_orchestrator.vllm.close()
        logger.info("vLLM client closed")

    # Close Qdrant connections
    if deps.graph_orchestrator and deps.graph_orchestrator.rag:
        try:
            deps.graph_orchestrator.rag.qdrant.close()
            logger.info("Qdrant client (RAG) closed")
        except Exception as e:
            logger.warning("Failed to close RAG Qdrant client", error=str(e))
    if deps.index_builder:
        try:
            deps.index_builder.qdrant.close()
            logger.info("Qdrant client (indexer) closed")
        except Exception as e:
            logger.warning("Failed to close indexer Qdrant client", error=str(e))

    # Close Redis connections
    if deps.rate_limiter and deps.rate_limiter.redis_client:
        try:
            deps.rate_limiter.redis_client.close()
            logger.info("Redis client (rate limiter) closed")
        except Exception as e:
            logger.warning("Failed to close rate limiter Redis", error=str(e))

    # Shutdown thread pool
    if deps._executor:
        deps._executor.shutdown(wait=True, cancel_futures=True)
        logger.info("Thread pool shutdown complete")


async def _periodic_session_cleanup():
    """Background task to cleanup expired sessions periodically."""
    while True:
        try:
            await asyncio.sleep(300)  # Every 5 minutes
            loop = asyncio.get_running_loop()
            count = await loop.run_in_executor(
                deps._executor, deps.session_manager.cleanup_expired
            )
            if count > 0:
                logger.info("Cleaned up expired sessions", count=count)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error("Session cleanup failed", error=str(e))


# ============================================================================
# App Factory
# ============================================================================

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="AI Coding Agent",
        description="Self-hosted AI agent for generating JUnit5 + Mockito unit tests",
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS middleware - configurable for production
    # In production, set CORS_ORIGINS env var to comma-separated list of allowed origins
    cors_origins_str = os.getenv("CORS_ORIGINS", "*")
    if cors_origins_str == "*":
        # Development mode - allow all
        allow_origins = ["*"]
        allow_credentials = False  # Cannot use credentials with wildcard origin
    else:
        # Production mode - specific origins only
        allow_origins = [origin.strip() for origin in cors_origins_str.split(",")]
        allow_credentials = True

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=allow_credentials,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
    )

    # Rate limiting middleware
    @app.middleware("http")
    async def rate_limit_middleware(request: Request, call_next):
        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/v1/health", "/v1/models", "/metrics"]:
            return await call_next(request)
        
        # Get client IP (handle proxy headers)
        client_ip = request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
        if not client_ip:
            client_ip = request.client.host if request.client else "unknown"
        
        # Check rate limit — run in executor to avoid blocking event loop
        loop = asyncio.get_running_loop()
        allowed, remaining = await loop.run_in_executor(
            deps._executor, deps.check_rate_limit, client_ip
        )
        if not allowed:
            logger.warning("Rate limit exceeded", client_ip=client_ip, remaining=remaining)
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "message": f"Too many requests. Max {deps.RATE_LIMIT_REQUESTS} requests per {deps.RATE_LIMIT_WINDOW} seconds.",
                    "retry_after": deps.RATE_LIMIT_WINDOW
                },
                headers={
                    "Retry-After": str(deps.RATE_LIMIT_WINDOW),
                    "X-RateLimit-Limit": str(deps.RATE_LIMIT_REQUESTS),
                    "X-RateLimit-Remaining": str(remaining),
                    "X-RateLimit-Reset": str(int(time.time() + deps.RATE_LIMIT_WINDOW)),
                }
            )
        
        response = await call_next(request)
        
        # Add rate limit headers to successful responses
        response.headers["X-RateLimit-Limit"] = str(deps.RATE_LIMIT_REQUESTS)
        response.headers["X-RateLimit-Remaining"] = str(remaining - 1)  # Account for current request 
        response.headers["X-RateLimit-Reset"] = str(int(time.time() + deps.RATE_LIMIT_WINDOW))
        
        return response

    # ── 422 validation error handler — log details for debugging ─────
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        logger.error(
            "Request validation failed (422)",
            path=request.url.path,
            errors=exc.errors(),
        )
        return JSONResponse(
            status_code=422,
            content={"detail": exc.errors()},
        )

    # ── Include Routers ──────────────────────────────────────────────
    app.include_router(health.router)
    app.include_router(chat.router)
    app.include_router(test_gen.router)
    app.include_router(index.router)
    app.include_router(embeddings.router)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server.app:app",
        host=os.getenv("SERVER_HOST", "0.0.0.0"),
        port=int(os.getenv("SERVER_PORT", "8080")),
        reload=True,
    )
