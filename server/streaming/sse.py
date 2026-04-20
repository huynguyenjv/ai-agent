"""SSE Streaming — Section 10.

Three-layer event model: thinking, tool steps, content.

Continue IDE compatibility:
- Continue only understands OpenAI streaming format (data: {"choices":[...]})
- Thinking/tool events are sent as OpenAI content deltas so Continue renders them
- SSE comments (: prefix) are invisible to Continue — used only for heartbeat
"""

from __future__ import annotations

import json


def sse_event(data: dict) -> str:
    """Format a dict as an SSE data line."""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _content_chunk(content: str, chunk_id: str = "chatcmpl-agent") -> str:
    """Create an OpenAI-compatible content delta chunk."""
    return sse_event({
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "choices": [{
            "index": 0,
            "delta": {"content": content},
            "finish_reason": None,
        }],
    })


# ---------------------------------------------------------------------------
# Layer 1 — Thinking events
# Rendered as faint text in Continue by wrapping in content delta
# ---------------------------------------------------------------------------

def thinking_event(content: str) -> str:
    """Layer 1 — Thinking event. Sent as empty content to keep connection alive.

    Continue will render this as text. We use a subtle format.
    """
    # Send as SSE comment — invisible to Continue but keeps connection alive
    # Don't pollute the actual chat output with thinking text
    return f": thinking: {content}\n\n"


# ---------------------------------------------------------------------------
# Layer 2 — Tool step events
# ---------------------------------------------------------------------------

def tool_start_event(tool: str, label: str, detail: str = "") -> str:
    """Layer 2 — Tool start. SSE comment (invisible to Continue)."""
    return f": tool_start: {tool} — {label}\n\n"


def tool_progress_event(step: str, detail: str = "", pct: int = -1) -> str:
    """Layer 2 — Tool progress. SSE comment."""
    return f": tool_progress: {step}\n\n"


def tool_done_event(tool: str, summary: str, ms: int = 0) -> str:
    """Layer 2 — Tool done. SSE comment."""
    return f": tool_done: {tool} — {summary}\n\n"


def tool_error_event(tool: str, error: str) -> str:
    """Layer 2 — Tool error. Sent as content delta so user sees the error."""
    return _content_chunk(f"\n\n⚠️ Error ({tool}): {error}")


# ---------------------------------------------------------------------------
# Layer 3 — Content events (OpenAI delta format)
# ---------------------------------------------------------------------------

def content_delta_event(content: str, chunk_id: str = "chatcmpl-agent") -> str:
    """Layer 3 — Content event in OpenAI delta format. Continue renders this."""
    return _content_chunk(content, chunk_id)


# ---------------------------------------------------------------------------
# Control events
# ---------------------------------------------------------------------------

def done_event() -> str:
    """Terminal event signaling stream completion."""
    return "data: [DONE]\n\n"


def tool_calls_event(
    tool_calls: list[dict],
    chunk_id: str = "chatcmpl-agent",
    *,
    native: bool = False,
) -> str:
    """Emit tool calls in native OpenAI format or <tool_call> text tags.

    Args:
        native: True  → OpenAI tool_calls object (for /review/pr, API clients)
                False → <tool_call> text in content (for Continue IDE)
    """
    if native:
        chunk1 = sse_event({
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "choices": [{
                "index": 0,
                "delta": {"tool_calls": tool_calls},
                "finish_reason": None,
            }],
        })
        chunk2 = sse_event({
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "tool_calls",
            }],
        })
        return chunk1 + chunk2

    lines: list[str] = []
    for tc in tool_calls:
        fn = tc.get("function") or {}
        payload = {"name": fn.get("name", ""), "arguments": json.loads(fn.get("arguments", "{}"))}
        lines.append(f"<tool_call>\n{json.dumps(payload, ensure_ascii=False)}\n</tool_call>")
    return _content_chunk("\n".join(lines), chunk_id)


def heartbeat_comment() -> str:
    """SSE heartbeat to prevent proxy timeout. Every 15 seconds."""
    return ": keep-alive\n\n"
