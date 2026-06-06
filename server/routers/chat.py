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
from server.metrics.counter import RequestTimer
from server.metrics.prometheus import record_request, record_tokens, ACTIVE_REQUESTS
from server.rate_limit import get_rate_limiter
from server.session import get_session_store
from server.agent.input_guard import get_input_guard
from server.agent.tool_validator import validate_tool_result, annotate_invalid
from server.audit import get_audit_logger
from server.metrics.tools import get_tool_analytics
from server.utils.content import normalize_content
from server.utils.sanitize import sanitize_user_input, sanitize_tool_output
from server.utils import secret_scanner
from server.validation import validate_chat_request, ValidationError
from server.streaming.sse import (
    thinking_event,
    tool_error_event,
    content_delta_event,
    done_event,
    heartbeat_comment,
    tool_calls_event,
)

logger = logging.getLogger("server.chat")

router = APIRouter()


class ChatMessage(BaseModel):
    model_config = {"extra": "allow"}

    role: str
    content: str | list | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None
    name: str | None = None


class ChatRequest(BaseModel):
    model_config = {"extra": "allow"}

    messages: list[ChatMessage]
    model: str = ""
    stream: bool = True
    tools: list[dict] | None = None
    tool_choice: str | dict | None = None
    active_file: str | None = None
    repo_path: str | None = None
    conversation_id: str | None = None  # Session tracking for multi-turn


@router.post("/v1/chat/completions")
async def chat_completions(
    request: ChatRequest,
    req: Request,
    x_api_key: str = Header(None),
    authorization: str = Header(None),
) -> StreamingResponse:
    verify_api_key(req, x_api_key, authorization)

    # Phase 10.6: input validation (size/count/path) before any processing
    try:
        validate_chat_request(request)
    except ValidationError as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=422, detail=str(e))

    # Rate limiting by API key or IP
    client_id = x_api_key or authorization or req.client.host if req.client else "unknown"
    limiter = get_rate_limiter()
    if not limiter.allow(client_id):
        retry_after = limiter.retry_after(client_id)
        from fastapi import HTTPException
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Retry after {retry_after:.1f}s",
            headers={"Retry-After": str(int(retry_after) + 1)},
        )

    logger.info(
        "chat request: model=%s tools=%d tool_names=%s",
        request.model,
        len(request.tools or []),
        [t.get("function", {}).get("name") for t in (request.tools or [])],
    )

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
        text = normalize_content(msg.content)
        if msg.role == "user":
            # Sanitize user input for prompt injection defense (Phase 4 markers
            # + Phase 10.2 InputGuard: neutralize role-hijack/delimiter/unicode).
            sanitized = sanitize_user_input(text)
            guarded_text, _ = get_input_guard().check_and_sanitize(sanitized.text)
            out.append(HumanMessage(content=guarded_text))
        elif msg.role == "tool":
            # Sanitize tool output (injection) then redact secrets (Phase 10.3)
            # before tool results enter the model context.
            safe = sanitize_tool_output(text)
            redacted, findings = secret_scanner.redact(safe)
            if findings:
                logger.warning("Redacted %d secret(s) from tool output", len(findings))
            # Phase 17.4/17.5: validate the tool result + record analytics.
            tool_name = getattr(msg, "name", None) or ""
            validation = validate_tool_result(tool_name, redacted)
            get_tool_analytics().record(tool_name or "unknown", success=validation.valid)
            out.append(ToolMessage(
                content=annotate_invalid(redacted, validation),
                tool_call_id=msg.tool_call_id or "",
            ))
        elif msg.role == "assistant":
            ai = AIMessage(content=text)
            if msg.tool_calls:
                ai.additional_kwargs["tool_calls"] = msg.tool_calls
            out.append(ai)
        elif msg.role == "system":
            out.append(SystemMessage(content=text))
        else:
            out.append(HumanMessage(content=msg.content or ""))
    return out


import re as _re

_AGENTS_PREFIX = _re.compile(r"^\s*/agents\b\s*", _re.IGNORECASE)


def _parse_agents_directive(messages) -> tuple[list, bool]:
    """Detect & strip a leading /agents on the latest user turn (R9).

    Returns (messages, multi_agent). /agents → run the reviewer + refine loop.
    """
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            content = m.content if isinstance(m.content, str) else normalize_content(m.content)
            if _AGENTS_PREFIX.match(content or ""):
                m.content = _AGENTS_PREFIX.sub("", content, count=1)
                return messages, True
            return messages, False
    return messages, False


def _has_tool_context(messages) -> bool:
    """True if any message is a tool result or an assistant with tool_calls."""
    for m in messages:
        if isinstance(m, ToolMessage):
            return True
        if isinstance(m, AIMessage) and (m.additional_kwargs or {}).get("tool_calls"):
            return True
    return False


async def _maybe_summarize(messages, vllm_client, model):
    """Phase R5: summarize long PURE-CHAT conversations to preserve context.

    Conservative — skips entirely when any tool message is present so the
    required assistant/tool_call pairing is never broken.
    """
    from server.agent.summarize import should_summarize, summarize_conversation

    if not should_summarize(messages) or _has_tool_context(messages):
        return messages
    keep_recent = 4
    if len(messages) <= keep_recent:
        return messages
    older, recent = messages[:-keep_recent], messages[-keep_recent:]
    summary = await summarize_conversation(older, vllm_client, model)
    if not summary:
        return messages
    logger.info("Summarized %d older messages into context", len(older))
    return [SystemMessage(content=f"[Earlier conversation summary: {summary}]")] + recent


def _enable_rag() -> bool:
    # Agentic-first: RAG (Qdrant) is OPT-IN. Default OFF — the agent gathers
    # context via client-side tools (grep/search_symbol/read_file) which are
    # always fresh and naturally isolated per user. Enable RAG only for large
    # repos needing semantic recall by setting ENABLE_RAG=true.
    return os.environ.get("ENABLE_RAG", "false").lower() in ("1", "true", "yes")


async def _stream_response(
    request: ChatRequest,
    req: Request,
) -> AsyncGenerator[str, None]:
    yield thinking_event("Phân tích intent...")

    # Initialize metrics tracking
    request_id = getattr(req.state, "correlation_id", str(time.time()))
    model = req.app.state.vllm_model
    metrics_timer = RequestTimer(
        request_id=request_id,
        model=model,
        correlation_id=request_id,
    )
    first_token_recorded = False

    from server.streaming.optimized import make_event_queue

    event_queue: asyncio.Queue = make_event_queue()  # bounded → backpressure (19.4)
    content_streamed = False
    output_tokens_estimate = 0

    async def sse_callback(event_type: str, content: str) -> None:
        nonlocal content_streamed, first_token_recorded, output_tokens_estimate
        if event_type == "content":
            # Track first token
            if not first_token_recorded:
                metrics_timer.mark_first_token()
                first_token_recorded = True
            content_streamed = True
            output_tokens_estimate += len(content) // 4  # Rough estimate
            await event_queue.put(content_delta_event(content))
        elif event_type == "error":
            await event_queue.put(tool_error_event("generate", content))

    messages = _convert_messages(request.messages)

    # R9: /agents opt-in → thorough multi-agent (reviewer + refine loop)
    messages, multi_agent = _parse_agents_directive(messages)

    # Phase 10.2: hard-block when the latest user turn is a critical prompt
    # injection (InputGuard blocks on CRITICAL by default). Lower-severity
    # threats are neutralized in _convert_messages, not blocked.
    latest_user = next(
        (normalize_content(m.content) for m in reversed(request.messages) if m.role == "user"),
        "",
    )
    guard_result = get_input_guard().check(latest_user)
    if guard_result.blocked:
        logger.warning(
            "Blocked prompt injection (level=%s, threats=%d)",
            guard_result.threat_level.value, len(guard_result.threats),
        )
        actor = req.headers.get("x-api-key") or (req.client.host if req.client else "unknown")
        get_audit_logger().security_violation(
            action="prompt_injection",
            actor=actor,
            correlation_id=request_id,
            threat_level=guard_result.threat_level.value,
            threat_count=len(guard_result.threats),
        )
        yield content_delta_event(
            "⚠️ Yêu cầu bị từ chối: phát hiện dấu hiệu prompt injection."
        )
        yield done_event()
        return

    # R5: condense long pure-chat history (skipped when tools are involved)
    messages = await _maybe_summarize(
        messages, req.app.state.vllm_client, req.app.state.vllm_model
    )

    # R10: recall durable cross-session memory (opt-in, off by default)
    from server.agent.memory_store import enable_memory, get_memory_store
    _mem_scope = request.conversation_id or req.headers.get("x-api-key") or "default"
    if enable_memory():
        recalled = get_memory_store().recall(_mem_scope, latest_user, top_k=3)
        if recalled:
            note = "Relevant memory from earlier sessions:\n" + "\n".join(
                f"- {m['content'][:200]}" for m in recalled
            )
            messages = [SystemMessage(content=note)] + messages

    # Load session context if conversation_id provided
    session_store = get_session_store()
    session_data = {}
    if request.conversation_id:
        session_data = session_store.get(request.conversation_id) or {}
        if session_data:
            logger.info("Loaded session context for %s", request.conversation_id[:8])

    active_file = extract_active_file(messages, request.active_file)
    if active_file:
        logger.info("Detected active_file from message content: %s", active_file)

    # Count tool turns from conversation history
    # Each pair of (assistant with tool_calls, tool result) = 1 turn
    tool_turns_used = sum(
        1 for msg in request.messages
        if msg.role == "assistant" and msg.tool_calls
    )
    logger.info("Detected %d tool turns from conversation history", tool_turns_used)

    # Phase 16.2: assign an A/B prompt variant (stable per conversation/client).
    from server.experiment import get_experiment_manager
    ab_unit = request.conversation_id or request_id
    experiment_variant = get_experiment_manager().get_variant("code_gen_prompt", ab_unit)

    # R11 RBAC: resolve role → tools this caller is allowed to use.
    from server.auth_rbac import role_for_key, permissions_for, allowed_tool_names
    _role = role_for_key(req.headers.get("x-api-key"))
    allowed_tools = list(allowed_tool_names(permissions_for(_role)))

    initial_state = {
        "messages": messages,
        "intent": session_data.get("last_intent", ""),  # Carry over from session
        "experiment_variant": experiment_variant,
        "multi_agent": multi_agent,
        "active_file": active_file or session_data.get("active_file"),
        "repo_path": request.repo_path or "",
        "mentioned_files": session_data.get("mentioned_files", []),
        "freshness_signal": False,
        "force_reindex": False,
        "rag_chunks": [],
        "rag_hit": False,
        "rag_enabled": _enable_rag(),
        "hash_verified": False,
        "tool_results": [],
        "context_assembled": session_data.get("context_summary", ""),
        "draft": "",
        "emitted_steps": [],
        "volatile_rejected": False,
        "pending_tool_calls": [],
        "is_tool_result_turn": False,
        "tool_turns_used": tool_turns_used,
        "client_tools": request.tools or [],
        "tool_choice": request.tool_choice,
        "allowed_tools": allowed_tools,
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

    from server.tracing import span

    async def _run_agent():
        try:
            with span("agent.invoke", **{"agent.multi": False}):
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
        # Record failed metrics
        metrics_timer.metrics.success = False
        metrics_timer.metrics.error_message = str(agent_result)
    elif isinstance(agent_result, dict):
        tc = agent_result.get("pending_tool_calls") or []
        if tc:
            # Native OpenAI tool_calls format for Continue with tool-call enabled
            yield tool_calls_event(tc, native=True)
            metrics_timer.set_tool_calls(tc)
        elif not content_streamed:
            draft = agent_result.get("draft", "")
            if draft:
                yield content_delta_event(draft)
                output_tokens_estimate += len(draft) // 4

        # Set intent from result
        metrics_timer.set_intent(agent_result.get("intent", ""))

        # Save session context for multi-turn
        if request.conversation_id:
            session_store.set(request.conversation_id, {
                "last_intent": agent_result.get("intent", ""),
                "active_file": agent_result.get("active_file"),
                "mentioned_files": agent_result.get("mentioned_files", []),
                "context_summary": agent_result.get("context_assembled", "")[:2000],
            })
            logger.debug("Saved session context for %s", request.conversation_id[:8])

        # R10: persist durable cross-session memory (opt-in)
        if enable_memory():
            summary = agent_result.get("context_assembled") or agent_result.get("draft", "")
            if summary:
                get_memory_store().remember(_mem_scope, summary[:1000])

    # Record metrics
    metrics_timer.metrics.total_time_ms = metrics_timer.get_elapsed_ms()
    metrics_timer.metrics.output_tokens = output_tokens_estimate
    # Estimate input tokens from messages
    input_text = " ".join(normalize_content(msg.content) for msg in request.messages)
    metrics_timer.metrics.input_tokens = len(input_text) // 4

    from server.metrics import get_metrics_counter
    get_metrics_counter().record(metrics_timer.metrics)

    # Record Prometheus metrics
    intent = agent_result.get("intent", "unknown") if isinstance(agent_result, dict) else "error"
    status = "success" if not isinstance(agent_result, Exception) else "error"
    duration = metrics_timer.get_elapsed_ms() / 1000.0
    ttft = metrics_timer.metrics.time_to_first_token_ms / 1000.0 if metrics_timer.metrics.time_to_first_token_ms else None
    record_request(intent, status, duration, ttft)
    record_tokens(metrics_timer.metrics.input_tokens, output_tokens_estimate, model)

    yield done_event()
