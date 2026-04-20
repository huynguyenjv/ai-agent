"""POST /v1/chat/completions — native tool-call streaming.

SSE streaming endpoint. Always streams regardless of stream field.
Authentication via X-Api-Key header.

Client (Continue) sends full conversation history each request — server is
stateless. Tool schemas from server registry are merged with client-provided
tools and forwarded to vLLM; model decides tool_calls natively. Continue
executes tools client-side and sends results back as Turn 2.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import time
from typing import AsyncGenerator

_SENTINEL = object()

from fastapi import APIRouter, Header, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

from server.auth import verify_api_key
from server.continue_compat import extract_active_file
from server.streaming.sse import (
    thinking_event,
    tool_error_event,
    content_delta_event,
    done_event,
    heartbeat_comment,
    tool_calls_event,
)

def _native_tool_calls() -> bool:
    """True → OpenAI native tool_calls; False → <tool_call> text tags (Continue)."""
    return os.environ.get("TOOL_CALL_FORMAT", "text").lower() == "native"

logger = logging.getLogger("server.chat")

router = APIRouter()


class ChatMessage(BaseModel):
    role: str
    content: str | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None


class ChatRequest(BaseModel):
    model_config = {"extra": "allow"}

    messages: list[ChatMessage]
    model: str = ""
    stream: bool = True
    tools: list[dict] | None = None
    tool_choice: str | dict | None = None
    active_file: str | None = None
    repo_path: str | None = None


@router.post("/v1/chat/completions")
async def chat_completions(
    request: ChatRequest,
    req: Request,
    x_api_key: str = Header(None),
    authorization: str = Header(None),
) -> StreamingResponse:
    verify_api_key(req, x_api_key, authorization)

    return StreamingResponse(
        _stream_response(request, req),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def _convert_messages(request_messages: list[ChatMessage]):
    out = []
    for msg in request_messages:
        if msg.role == "user":
            out.append(HumanMessage(content=msg.content or ""))
        elif msg.role == "tool":
            out.append(ToolMessage(
                content=msg.content or "",
                tool_call_id=msg.tool_call_id or "",
            ))
        elif msg.role == "assistant":
            ai = AIMessage(content=msg.content or "")
            if msg.tool_calls:
                ai.additional_kwargs["tool_calls"] = msg.tool_calls
            out.append(ai)
        elif msg.role == "system":
            out.append(SystemMessage(content=msg.content or ""))
        else:
            out.append(HumanMessage(content=msg.content or ""))
    return out


def _enable_rag() -> bool:
    return os.environ.get("ENABLE_RAG", "false").lower() in ("1", "true", "yes")


async def _stream_response(
    request: ChatRequest,
    req: Request,
) -> AsyncGenerator[str, None]:
    yield thinking_event("Phân tích intent...")

    event_queue: asyncio.Queue = asyncio.Queue()
    content_streamed = False

    async def sse_callback(event_type: str, content: str) -> None:
        nonlocal content_streamed
        if event_type == "content":
            content_streamed = True
            await event_queue.put(content_delta_event(content))
        elif event_type == "error":
            await event_queue.put(tool_error_event("generate", content))

    messages = _convert_messages(request.messages)

    active_file = extract_active_file(messages, request.active_file)
    if active_file:
        logger.info("Detected active_file from message content: %s", active_file)

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
        "client_tools": request.tools or [],
        "tool_choice": request.tool_choice,
    }

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
        enable_rag=_enable_rag(),
    )

    async def _run_agent():
        try:
            result = await agent.ainvoke(initial_state)
            await event_queue.put((_SENTINEL, result))
        except Exception as exc:
            await event_queue.put((_SENTINEL, exc))

    run_task = asyncio.create_task(_run_agent())

    yield thinking_event("Đang xử lý...")

    last_event_time = time.monotonic()
    agent_result = None

    try:
        while True:
            if await req.is_disconnected():
                logger.info("Client disconnected, cancelling agent")
                break
            try:
                event = await asyncio.wait_for(event_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                if time.monotonic() - last_event_time > 15:
                    yield heartbeat_comment()
                    last_event_time = time.monotonic()
                continue

            if isinstance(event, tuple) and len(event) == 2 and event[0] is _SENTINEL:
                agent_result = event[1]
                break

            yield event
            last_event_time = time.monotonic()
    finally:
        if not run_task.done():
            run_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await run_task

    if isinstance(agent_result, Exception):
        logger.error("Agent execution failed: %s", agent_result)
        yield tool_error_event("agent", str(agent_result))
    elif isinstance(agent_result, dict):
        tc = agent_result.get("pending_tool_calls") or []
        if tc and not content_streamed:
            yield tool_calls_event(tc, native=_native_tool_calls())
        elif not content_streamed:
            draft = agent_result.get("draft", "")
            if draft:
                yield content_delta_event(draft)

    yield done_event()
