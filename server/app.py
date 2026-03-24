"""FastAPI Application — Section 7 + Phase 7.

Cloud VM server with:
- POST /v1/chat/completions (SSE stream)
- POST /index (chunk ingestion)
- Authentication middleware
- Lifespan handler for Qdrant and embedder initialization
- Structured JSON logging with correlation ID
- Health check
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI

from server.logging_config import correlation_id_var
from server.rag.embedder import Embedder
from server.rag.qdrant_client import QdrantService
from server.routers.chat import router as chat_router
from server.routers.index import router as index_router

logger = logging.getLogger("server")

# Config from environment — Section 17
QDRANT_URL = os.environ.get("QDRANT_URL", "http://127.0.0.1:6333")
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8080/v1")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan handler — initialize and cleanup resources."""
    logger.info("Starting AI Coding Agent server...")

    # Initialize Qdrant (non-blocking — will retry on first request if unavailable)
    qdrant = QdrantService(url=QDRANT_URL)
    try:
        await qdrant.ensure_collection()
        logger.info("Qdrant connected: %s", QDRANT_URL)
    except Exception as e:
        logger.warning("Qdrant not available at startup (%s). Will retry on first request.", e)
    app.state.qdrant = qdrant

    # Initialize Embedder (lazy load on first use if model not cached)
    try:
        embedder = Embedder()
        logger.info("Embedder initialized.")
    except Exception as e:
        logger.warning("Embedder init failed (%s). Will retry on first request.", e)
        embedder = None
    app.state.embedder = embedder

    # Initialize vLLM client
    app.state.vllm_client = AsyncOpenAI(
        base_url=VLLM_BASE_URL,
        api_key="not-needed",  # vLLM local, no auth required
    )
    app.state.vllm_model = os.environ.get("VLLM_MODEL", "qwen2.5-coder")

    logger.info("Server ready. vLLM: %s", VLLM_BASE_URL)

    yield

    # Cleanup
    await qdrant.close()
    logger.info("Server shutdown complete.")


def create_app() -> FastAPI:
    """Create the FastAPI application."""
    app = FastAPI(
        title="AI Coding Agent",
        version="2.0.0",
        lifespan=lifespan,
    )

    # CORS — restrict in production via CORS_ORIGINS env var
    cors_origins = os.environ.get("CORS_ORIGINS", "*").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request logging middleware with correlation ID (Phase 7)
    @app.middleware("http")
    async def logging_middleware(request: Request, call_next):
        cid = str(uuid.uuid4())[:8]
        request.state.correlation_id = cid

        # Propagate correlation_id to all loggers in this async context
        token = correlation_id_var.set(cid)
        try:
            start = time.monotonic()
            response = await call_next(request)
            elapsed_ms = (time.monotonic() - start) * 1000

            logger.info(
                "[%s] %s %s -> %d (%.1fms)",
                cid,
                request.method,
                request.url.path,
                response.status_code,
                elapsed_ms,
            )

            response.headers["X-Correlation-ID"] = cid
            return response
        finally:
            correlation_id_var.reset(token)

    # Include routers
    app.include_router(chat_router)
    app.include_router(index_router)

    # Health check
    @app.get("/health")
    async def health():
        return {"status": "ok", "version": "2.0.0"}

    return app


app = create_app()
