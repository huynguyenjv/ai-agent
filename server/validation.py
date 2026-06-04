"""Input Validation — Phase 10.6.

Schema/size/path validation for incoming requests. Rejects oversized payloads
(DoS), path-traversal in file references, and malformed tool arguments before
they reach the agent.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger("server.validation")

# Limits (override via env)
MAX_MESSAGES = int(os.environ.get("MAX_MESSAGES", "300"))
MAX_MESSAGE_CHARS = int(os.environ.get("MAX_MESSAGE_CHARS", "200000"))      # per message
MAX_TOTAL_CHARS = int(os.environ.get("MAX_TOTAL_CHARS", "1000000"))         # whole conversation
MAX_PATH_LEN = int(os.environ.get("MAX_PATH_LEN", "4096"))


class ValidationError(ValueError):
    """Raised when an input fails validation. Maps to HTTP 422."""


def _content_len(content) -> int:
    """Length of a message content that may be str, list (multimodal), or None."""
    if content is None:
        return 0
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        total = 0
        for item in content:
            if isinstance(item, dict):
                total += len(str(item.get("text", "")))
            else:
                total += len(str(item))
        return total
    return len(str(content))


def is_safe_relative_path(path: str) -> bool:
    """True if a path is a safe repo-relative reference (no traversal/abs/NUL)."""
    if not path:
        return True
    if len(path) > MAX_PATH_LEN:
        return False
    if "\x00" in path:
        return False
    normalized = path.replace("\\", "/")
    # Reject absolute paths and drive letters
    if normalized.startswith("/") or (len(normalized) > 1 and normalized[1] == ":"):
        return False
    # Reject parent-directory traversal
    parts = normalized.split("/")
    if ".." in parts:
        return False
    return True


def validate_chat_request(request) -> None:
    """Validate a ChatRequest. Raises ValidationError on violation.

    Checks: message count, per-message size, total size, and path safety of
    active_file (a repo-relative reference).
    """
    messages = getattr(request, "messages", None) or []

    if not messages:
        raise ValidationError("messages must not be empty")

    if len(messages) > MAX_MESSAGES:
        raise ValidationError(f"too many messages: {len(messages)} > {MAX_MESSAGES}")

    total = 0
    for i, msg in enumerate(messages):
        n = _content_len(getattr(msg, "content", None))
        if n > MAX_MESSAGE_CHARS:
            raise ValidationError(
                f"message[{i}] too large: {n} chars > {MAX_MESSAGE_CHARS}"
            )
        total += n

    if total > MAX_TOTAL_CHARS:
        raise ValidationError(f"conversation too large: {total} chars > {MAX_TOTAL_CHARS}")

    active_file = getattr(request, "active_file", None)
    if active_file and not is_safe_relative_path(active_file):
        raise ValidationError(f"unsafe active_file path: {active_file!r}")
