"""Node: emit_tool_calls — Agent Tool Call Flow.

Pure SSE side-effect node. Emits thinking comments and OpenAI tool_calls chunks.
Connected to END — handles exceptions internally (no graph fallback exists).
"""

from __future__ import annotations

import json
import logging

from server.agent.state import AgentState

logger = logging.getLogger("server.emit_tool_calls")


async def emit_tool_calls(state: AgentState, sse_callback=None) -> dict:
    """Emit SSE events for pending_tool_calls.

    Sequence:
    1. thinking comment (immediate, < 50ms)
    2. tool_start comment
    3. tool_calls event (two-chunk OpenAI format via sse_callback)
    4. done event

    Returns {} — no state changes, pure side-effect node.
    """
    if sse_callback is None:
        return {}

    tool_calls = state.get("pending_tool_calls", [])
    if not tool_calls:
        return {}

    tool_names = ", ".join(
        tc["function"]["name"] for tc in tool_calls if "function" in tc
    )

    try:
        # Thinking comment — immediate
        await sse_callback("thinking", "Chuẩn bị công cụ...")

        # Tool start comment
        await sse_callback("thinking", f"Gọi: {tool_names}")

        # Emit tool_calls in OpenAI format (two chunks handled by sse_callback -> tool_calls_event)
        await sse_callback("tool_calls", json.dumps(tool_calls))

    except Exception as exc:
        logger.exception("emit_tool_calls failed: %s", exc)
        # Surface error as visible content delta (no graph fallback from this node)
        try:
            await sse_callback("error", f"Tool call failed: {exc}")
        except Exception:
            pass

    return {}
