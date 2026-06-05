"""Correlation ID middleware — Phase 14.3.

Honors an incoming `X-Correlation-ID` (so a trace id survives across services),
otherwise generates one; propagates it to all logs in the request's async
context, echoes it on the response, and times the request.
"""

from __future__ import annotations

import logging
import time
import uuid

from fastapi import FastAPI, Request

from server.logging_config import correlation_id_var

logger = logging.getLogger("server.request")

CORRELATION_HEADER = "X-Correlation-ID"


def _new_id() -> str:
    return str(uuid.uuid4())[:8]


def register_correlation_middleware(app: FastAPI) -> None:
    """Attach the correlation-id + request-logging middleware to the app."""

    @app.middleware("http")
    async def correlation_middleware(request: Request, call_next):
        cid = request.headers.get(CORRELATION_HEADER) or _new_id()
        request.state.correlation_id = cid
        token = correlation_id_var.set(cid)
        try:
            start = time.monotonic()
            response = await call_next(request)
            elapsed_ms = (time.monotonic() - start) * 1000
            logger.info(
                "[%s] %s %s -> %d (%.1fms)",
                cid, request.method, request.url.path,
                response.status_code, elapsed_ms,
            )
            response.headers[CORRELATION_HEADER] = cid
            return response
        finally:
            correlation_id_var.reset(token)
