"""POST /v1/chat/completions — Section 7 + Section 10.

SSE streaming endpoint. Always streams regardless of stream field.
Authentication via X-Api-Key header.

Continue IDE compatibility:
- active_file extracted from code fences in message content
- thinking/tool events wrapped in OpenAI delta format (SSE comments invisible to Continue)
- All events are valid OpenAI streaming chunks
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import AsyncGenerator

_SENTINEL = object()  # Completion signal for SSE queue

from fastapi import APIRouter, Header, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from server.auth import verify_api_key
from server.continue_compat import extract_active_file
from server.streaming.sse import (
    thinking_event,
    tool_start_event,
    tool_done_event,
    tool_error_event,
    content_delta_event,
    done_event,
    heartbeat_comment,
    tool_calls_event,
)

logger = logging.getLogger("server.chat")

router = APIRouter()


class ChatMessage(BaseModel):
    role: str
    content: str | None = None        # null in assistant tool_calls messages
    tool_calls: list[dict] | None = None  # present in assistant messages for Turn 2
    tool_call_id: str | None = None   # present in role:"tool" result messages


class ChatRequest(BaseModel):
    """OpenAI-compatible chat completions request.

    Continue sends standard OpenAI format with extra fields like
    temperature, max_tokens, etc. We accept and ignore them.
    """
    model_config = {"extra": "allow"}

    messages: list[ChatMessage]
    model: str = ""
    stream: bool = True
    active_file: str | None = None
    repo_path: str | None = None


@router.post("/v1/chat/completions")
async def chat_completions(
    request: ChatRequest,
    req: Request,
    x_api_key: str = Header(None),
    authorization: str = Header(None),
) -> StreamingResponse:
    """Stream SSE response through the LangGraph agent."""
    verify_api_key(req, x_api_key, authorization)

    return StreamingResponse(
        _stream_response(request, req),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


async def _stream_response(
    request: ChatRequest,
    req: Request,
) -> AsyncGenerator[str, None]:
    """Generate SSE events through the agent pipeline.

    Section 10 timing constraints:
    - First thinking event: within 50ms
    - First tool_start: within 100ms of tool decision
    - First content token: within 200ms of LLM start
    - Max blank screen: 500ms
    - Heartbeat: every 15 seconds
    """
    # Emit first thinking event immediately (within 50ms)
    # Wrapped in OpenAI delta format so Continue renders it as content
    yield thinking_event("Phân tích intent...")

    # Build event queue for SSE callback
    event_queue: asyncio.Queue = asyncio.Queue()
    content_streamed = False

    async def sse_callback(event_type: str, content: str) -> None:
        """Callback for agent nodes to emit SSE events."""
        nonlocal content_streamed
        if event_type == "content":
            content_streamed = True
            await event_queue.put(content_delta_event(content))
        elif event_type == "error":
            await event_queue.put(tool_error_event("generate", content))
        elif event_type == "tool_calls":
            # content is JSON-serialized list of tool call dicts
            await event_queue.put(tool_calls_event(json.loads(content)))

    # Convert messages to LangChain format
    messages = []
    for msg in request.messages:
        if msg.role == "user":
            messages.append(HumanMessage(content=msg.content or ""))
        elif msg.role == "tool":
            messages.append(ToolMessage(
                content=msg.content or "",
                tool_call_id=msg.tool_call_id or "",
            ))
        else:
            messages.append(AIMessage(content=msg.content or ""))

    # Extract active_file from Continue's message content if not explicitly provided
    active_file = extract_active_file(messages, request.active_file)
    if active_file:
        logger.info("Detected active_file from message content: %s", active_file)

    # Initial agent state
    initial_state = {
        "messages": messages,
        "intent": "",
        "active_file": active_file,
        "mentioned_files": [],
        "freshness_signal": False,
        "force_reindex": False,
        "rag_chunks": [],
        "rag_hit": False,
        "hash_verified": False,
        "tool_results": [],
        "context_assembled": "",
        "draft": "",
        "emitted_steps": [],
        "volatile_rejected": False,
        "pending_tool_calls": [],
        "is_tool_result_turn": False,
        "tool_turns_used": 0,
    }

    # Build and run agent graph
    from server.agent.graph import build_agent_graph

    vllm_client = req.app.state.vllm_client
    model = req.app.state.vllm_model
    qdrant = req.app.state.qdrant
    embedder = req.app.state.embedder

    agent = build_agent_graph(
        vllm_client=vllm_client,
        model=model,
        qdrant=qdrant,
        embedder=embedder,
        sse_callback=sse_callback,
    )

    # Run agent in background task, signal completion via queue
    async def _run_agent():
        try:
            result = await agent.ainvoke(initial_state)
            await event_queue.put((_SENTINEL, result))
        except Exception as exc:
            await event_queue.put((_SENTINEL, exc))

    asyncio.create_task(_run_agent())

    yield thinking_event("Đang xử lý...")

    # Stream events from queue + heartbeat
    last_event_time = time.monotonic()
    agent_result = None

    while True:
        try:
            event = await asyncio.wait_for(event_queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            if time.monotonic() - last_event_time > 15:
                yield heartbeat_comment()
                last_event_time = time.monotonic()
            continue

        # Completion signal
        if isinstance(event, tuple) and len(event) == 2 and event[0] is _SENTINEL:
            agent_result = event[1]
            break

        yield event
        last_event_time = time.monotonic()

    # If agent failed, emit error
    if isinstance(agent_result, Exception):
        logger.error("Agent execution failed: %s", agent_result)
        yield tool_error_event("agent", str(agent_result))
    elif agent_result and not content_streamed:
        # If no content was streamed via callback, stream the draft directly
        draft = agent_result.get("draft", "")
        if draft:
            yield content_delta_event(draft)

    # Terminal event
    yield done_event()
