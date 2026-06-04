"""Graceful Degradation — Phase 11.4.

Fallback responses when the LLM backend is unavailable (e.g. its circuit
breaker is open), so the user gets a clear, useful message instead of a hard
error or a hang.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("server.agent.fallback")

LLM_UNAVAILABLE_MESSAGE = (
    "⚠️ Trợ lý tạm thời chưa thể xử lý yêu cầu do dịch vụ mô hình (LLM) đang quá tải "
    "hoặc không phản hồi. Vui lòng thử lại sau giây lát.\n\n"
    "(The model service is temporarily unavailable. Please retry shortly.)"
)


def llm_unavailable_draft(detail: str | None = None) -> dict:
    """A graph-node-shaped result used when the LLM backend is unavailable.

    Returns the same dict shape as `generate()` so the graph can continue to
    post_process / END without special casing.
    """
    if detail:
        logger.warning("Serving LLM-unavailable fallback: %s", detail)
    return {
        "draft": LLM_UNAVAILABLE_MESSAGE,
        "pending_tool_calls": [],
        "degraded": True,
    }


def is_degraded(state: dict) -> bool:
    return bool(state.get("degraded"))
