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
# NB: server.rag.embedder / qdrant_client are imported lazily inside lifespan
# (only when ENABLE_RAG is on) so RAG-off startup never imports torch.
from server.routers.chat import router as chat_router
from server.routers.index import router as index_router
from server.routers.review import router as review_router
from server.routers.metrics import router as metrics_router
from server.routers.feedback import router as feedback_router
from server.routers.health import router as health_router
from server.routers.jobs import router as jobs_router

logger = logging.getLogger("server")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan handler — initialize and cleanup resources."""
    logger.info("Starting AI Coding Agent server...")

    # Read env vars here (after dotenv loaded in main.py)
    qdrant_url = os.environ.get("QDRANT_URL", "http://127.0.0.1:6333")
    vllm_base_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")

    # RAG (Qdrant + embedder) is opt-in (agentic-first default). When RAG is
    # off, skip loading the embedding model and connecting Qdrant entirely —
    # faster startup, less memory, no Qdrant dependency.
    rag_enabled = os.environ.get("ENABLE_RAG", "false").lower() in ("1", "true", "yes")

    qdrant = None
    embedder = None
    if rag_enabled:
        # Lazy imports — only pull in torch/sentence-transformers when RAG is on.
        from server.rag.embedder import Embedder
        from server.rag.qdrant_client import QdrantService

        # Initialize Qdrant (non-blocking — will retry on first request if unavailable)
        qdrant = QdrantService(url=qdrant_url)
        try:
            await qdrant.ensure_collection()
            logger.info("Qdrant connected: %s", qdrant_url)
        except Exception as e:
            logger.warning("Qdrant not available at startup (%s). Will retry on first request.", e)

        # Initialize Embedder (lazy load on first use if model not cached)
        try:
            embedder = Embedder()
            logger.info("Embedder initialized.")
        except Exception as e:
            logger.warning("Embedder init failed (%s). Will retry on first request.", e)
            embedder = None
    else:
        logger.info("RAG disabled (ENABLE_RAG=false) — skipping Qdrant + embedder init.")

    app.state.qdrant = qdrant
    app.state.embedder = embedder

    # Initialize vLLM client. DEV_MODE → mock (no real model server needed).
    from server.dev_mode import is_dev_mode, build_mock_vllm_client

    if is_dev_mode():
        app.state.vllm_client = build_mock_vllm_client()
    else:
        from server.connections import build_vllm_client

        app.state.vllm_client = build_vllm_client(vllm_base_url)
    app.state.vllm_model = os.environ.get("VLLM_MODEL", "qwen2.5-coder")

    # Phase 19.3: optional speculative pre-warm (off by default)
    from server.speculative import prewarm_enabled, prewarm_vllm

    if prewarm_enabled():
        await prewarm_vllm(app.state.vllm_client, app.state.vllm_model)

    logger.info("Server ready. vLLM: %s", vllm_base_url)

    yield

    # Cleanup
    if qdrant is not None:
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

    # Correlation ID + request logging middleware (Phase 14.3)
    from server.middleware.correlation import register_correlation_middleware

    register_correlation_middleware(app)

    # OpenTelemetry tracing (Phase 14.1) — no-op unless configured
    from server.tracing import setup_tracing

    setup_tracing(app)

    # Include routers
    app.include_router(chat_router)
    app.include_router(index_router)
    app.include_router(review_router)
    app.include_router(metrics_router)
    app.include_router(feedback_router)
    app.include_router(health_router)
    app.include_router(jobs_router)

    # Health check
    @app.get("/health")
    async def health():
        return {"status": "ok", "version": "2.0.0"}

    return app


app = create_app()
