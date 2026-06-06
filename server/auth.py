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


def _audit_auth_failure(request: Request, reason: str) -> None:
    """Best-effort audit of an auth failure (R3)."""
    try:
        from server.audit import get_audit_logger

        actor = request.client.host if request.client else "unknown"
        cid = getattr(request.state, "correlation_id", "")
        get_audit_logger().auth("verify_api_key", actor=actor, outcome="error",
                                correlation_id=cid, reason=reason)
    except Exception:
        pass


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
        _audit_auth_failure(request, "missing_key_or_server_unconfigured")
        raise HTTPException(status_code=403, detail="Invalid API key")

    if not hmac.compare_digest(token.encode(), API_KEY.encode()):
        _audit_auth_failure(request, "bad_key")
        raise HTTPException(status_code=403, detail="Invalid API key")
