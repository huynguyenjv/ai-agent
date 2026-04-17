"""Authentication middleware — Phase 7.

Constant-time API key comparison.
Supports both X-Api-Key header and Authorization: Bearer token.
"""

from __future__ import annotations

import hmac
import logging
import os

from fastapi import HTTPException, Request

logger = logging.getLogger("server.auth")

API_KEY = os.environ.get("API_KEY", "")


def verify_api_key(
    request: Request,
    x_api_key: str | None = None,
    authorization: str | None = None,
) -> None:
    """Verify API key from headers. Raises HTTPException(403) on failure.

    Uses hmac.compare_digest for constant-time comparison (Phase 7).
    """
    token = x_api_key
    if not token and authorization and authorization.startswith("Bearer "):
        token = authorization[7:]

    if not token or not API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")

    if not hmac.compare_digest(token.encode(), API_KEY.encode()):
        raise HTTPException(status_code=403, detail="Invalid API key")
