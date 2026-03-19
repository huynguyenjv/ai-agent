"""
SSE streaming utilities for the AI Agent API.

Extracted from api.py — used by chat and test_gen routers.
"""

import json
import uuid
from typing import Optional

import structlog
from fastapi.responses import StreamingResponse

from agent.models import GenerationRequest, StreamEvent, StreamPhase
from utils.tokenizer import count_tokens
from ..dependencies import _get_orchestrator, graph_orchestrator
from ..schemas import (
    ChatMessage, ChatCompletionResponse, ChatCompletionChoice,
    ChatCompletionUsage,
)

logger = structlog.get_logger()


def _sse_chunk(
    response_id: str,
    created_time: int,
    model: str,
    delta: dict,
    finish_reason: Optional[str] = None,
) -> str:
    """Format a single SSE data line."""
    chunk = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created_time,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason,
        }],
    }
    return f"data: {json.dumps(chunk)}\n\n"


def _non_streaming_response(
    response_id: str,
    created_time: int,
    model: str,
    content: str,
    user_content: str,
    tokens_used: int,
) -> ChatCompletionResponse:
    """Build a standard non-streaming ChatCompletionResponse."""
    return ChatCompletionResponse(
        id=response_id,
        created=created_time,
        model=model,
        choices=[
            ChatCompletionChoice(
                message=ChatMessage(role="assistant", content=content),
                finish_reason="stop",
            )
        ],
        usage=ChatCompletionUsage(
            prompt_tokens=count_tokens(user_content),
            completion_tokens=count_tokens(content),
            total_tokens=tokens_used or (count_tokens(user_content) + count_tokens(content)),
        ),
    )


def _stream_buffered(
    response_id: str,
    created_time: int,
    model: str,
    content: str,
) -> StreamingResponse:
    """Stream already-buffered content as SSE chunks (for test-gen)."""

    async def _generate():
        # Role chunk
        yield _sse_chunk(response_id, created_time, model,
                         delta={"role": "assistant", "content": ""})
        # Content in ~80-char pieces
        chunk_size = 80
        for i in range(0, len(content), chunk_size):
            yield _sse_chunk(response_id, created_time, model,
                             delta={"content": content[i:i + chunk_size]})
        # Finish
        yield _sse_chunk(response_id, created_time, model,
                         delta={}, finish_reason="stop")
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


def _stream_test_generation(
    response_id: str,
    created_time: int,
    model: str,
    gen_request: GenerationRequest,
) -> StreamingResponse:
    """Stream test generation with progress phases + token-by-token code.

    Pipes ``graph_orchestrator.generate_test_streaming()`` events directly as SSE:
    - Phase status messages (planning, retrieving, validating) → full text deltas
    - Code tokens (generating) → individual token deltas
    - Done/Error → finish reason

    The client sees output appear progressively, just like Claude or Copilot.
    """

    async def _generate():
        # Role chunk
        yield _sse_chunk(response_id, created_time, model,
                         delta={"role": "assistant", "content": ""})

        active = _get_orchestrator()
        if not active:
            yield _sse_chunk(response_id, created_time, model,
                             delta={"content": "\n\n❌ Error: Service not initialized"})
            return

        try:
            async for event in active.generate_test_streaming(gen_request):
                if event.phase == StreamPhase.ERROR:
                    yield _sse_chunk(response_id, created_time, model,
                                     delta={"content": f"\n\n❌ {event.content}"})
                    break

                if event.phase == StreamPhase.DONE:
                    # Append metadata block
                    meta = event.metadata or {}
                    meta_lines = ["\n\n---\n"]

                    # Strategy & complexity
                    strategy = meta.get("strategy_used", "single_pass")
                    if strategy == "two_phase":
                        meta_lines.append("- **Strategy:** 🔄 Two-Phase Generation")
                    else:
                        meta_lines.append("- **Strategy:** ⚡ Single Pass")
                    complexity = meta.get("complexity_score", 0)
                    if complexity:
                        meta_lines.append(f"- **Complexity:** {complexity}")

                    # Validation
                    meta_lines.append(
                        f"- **Validation:** {'✅ passed' if meta.get('validation_passed') else '❌ failed'}"
                    )
                    issues: list[str] = meta.get("validation_issues", [])
                    if issues:
                        meta_lines.append(f"- **Issues:** {', '.join(issues[:5])}")

                    # Repairs
                    repairs = meta.get("repair_attempts", 0)
                    if repairs:
                        meta_lines.append(f"- **Repair attempts:** {repairs}")

                    # RAG & tokens
                    meta_lines.append(f"- **RAG chunks:** {meta.get('rag_chunks_used', 0)}")
                    meta_lines.append(f"- **Tokens used:** {meta.get('tokens_used', 0)}")

                    # Elapsed time
                    elapsed = meta.get("elapsed_ms", 0)
                    if elapsed:
                        elapsed_s = elapsed / 1000
                        meta_lines.append(f"- **Total time:** {elapsed_s:.1f}s")

                    meta_block = "\n".join(meta_lines) + "\n"
                    yield _sse_chunk(response_id, created_time, model,
                                     delta={"content": meta_block})
                    break

                if event.delta:
                    # Token-by-token code delta — forward directly
                    yield _sse_chunk(response_id, created_time, model,
                                     delta={"content": event.content})
                else:
                    # Phase status message — send as full content chunk
                    yield _sse_chunk(response_id, created_time, model,
                                     delta={"content": event.content})
        except Exception as exc:
             yield _sse_chunk(response_id, created_time, model,
                              delta={"content": f"\n\n❌ Error: {exc}"})

        # Finish
        yield _sse_chunk(response_id, created_time, model,
                         delta={}, finish_reason="stop")
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


def _stream_from_vllm(
    response_id: str,
    created_time: int,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: Optional[float],
    max_tokens: Optional[int],
    messages: Optional[list] = None,
) -> StreamingResponse:
    """Pipe vLLM streaming directly to the client — real TTFT.

    Includes tool-call interception: when the model outputs ``<tool_call>``
    XML, the server buffers it, parses the JSON, and emits an
    OpenAI-compatible ``tool_calls`` delta so Continue can execute the tool.
    """

    async def _generate():
        # Role chunk
        yield _sse_chunk(response_id, created_time, model,
                         delta={"role": "assistant", "content": ""})

        # ── Tool-call interception state ─────────────────────────
        START_TAG = "<tool_call>"
        END_TAG = "</tool_call>"
        TAG_MAX_LEN = 20

        full_text = ""        # entire accumulated response
        in_tool_tag = False   # True while buffering inside <tool_call>
        tool_buffer = ""      # content between tags
        tool_call_index = 0   # increments for each parsed call
        has_tool_calls = False # any tool call emitted?
        pending_text = ""     # text waiting to be yielded (may contain partial tag)
        json_warn_logged = False # flag to suppress repetitive warnings

        try:
            async for item in graph_orchestrator.vllm.astream_generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=messages,
            ):
                if item is None:
                    break

                full_text += item

                if in_tool_tag:
                    # Accumulate inside <tool_call> buffer
                    tool_buffer += item
                    if END_TAG in tool_buffer:
                        # Full tool call received
                        raw_call = tool_buffer.split(END_TAG)[0].strip()
                        remainder = tool_buffer.split(END_TAG, 1)[1] if END_TAG in tool_buffer else ""

                        try:
                            # Robustness: sometimes model adds text before JSON in the tag
                            if "{" in raw_call:
                                raw_call = raw_call[raw_call.find("{"):]
                            call_data = json.loads(raw_call)
                            call_id = f"call_{uuid.uuid4().hex[:12]}"
                            yield _sse_chunk(response_id, created_time, model,
                                delta={
                                    "tool_calls": [{
                                        "index": tool_call_index,
                                        "id": call_id,
                                        "type": "function",
                                        "function": {
                                            "name": call_data.get("name", ""),
                                            "arguments": json.dumps(
                                                call_data.get("arguments", {}),
                                                ensure_ascii=False,
                                            )
                                        }
                                    }]
                                })
                            tool_call_index += 1
                            has_tool_calls = True
                        except Exception as exc:
                            logger.error("Failed to parse tool call",
                                         error=str(exc), raw=raw_call[:200])
                            yield _sse_chunk(response_id, created_time, model,
                                             delta={"content": f"\n\n[Error parsing tool call: {exc}]"})
                        
                        in_tool_tag = False
                        tool_buffer = ""
                        # If there's text after </tool_call>, put it back as pending
                        if remainder.strip():
                            pending_text = remainder
                    continue

                # ── Not inside a tool tag — check for tag start ──────
                pending_text += item
                
                if not has_tool_calls and not json_warn_logged and pending_text.strip().startswith("{"):
                     # Heuristic: Model output raw JSON instead of XML
                     logger.warning("Model seems to be outputting raw JSON without <tool_call> tags", text=pending_text[:50])
                     json_warn_logged = True
                
                # Look for <tool_call> in the pending text
                if START_TAG in pending_text:
                    # Split: yield text before the tag, buffer after
                    before, after = pending_text.split(START_TAG, 1)
                    if before:
                        yield _sse_chunk(response_id, created_time, model,
                                         delta={"content": before})
                    pending_text = ""

                    # If after already contains the end tag, process immediately
                    # (happens when tag was split across chunks and reassembled)
                    if END_TAG in after:
                        raw_call = after.split(END_TAG)[0].strip()
                        remainder = after.split(END_TAG, 1)[1]
                        try:
                            if "{" in raw_call:
                                raw_call = raw_call[raw_call.find("{"):]
                            call_data = json.loads(raw_call)
                            call_id = f"call_{uuid.uuid4().hex[:12]}"
                            yield _sse_chunk(response_id, created_time, model,
                                delta={
                                    "tool_calls": [{
                                        "index": tool_call_index,
                                        "id": call_id,
                                        "type": "function",
                                        "function": {
                                            "name": call_data.get("name", ""),
                                            "arguments": json.dumps(
                                                call_data.get("arguments", {}),
                                                ensure_ascii=False,
                                            )
                                        }
                                    }]
                                })
                            tool_call_index += 1
                            has_tool_calls = True
                        except Exception as exc:
                            logger.error("Failed to parse tool call",
                                         error=str(exc), raw=raw_call[:200])
                            yield _sse_chunk(response_id, created_time, model,
                                             delta={"content": f"\n\n[Error parsing tool call: {exc}]"})
                        if remainder.strip():
                            pending_text = remainder
                    else:
                        in_tool_tag = True
                        tool_buffer = after
                    continue

                # Check if pending_text ends with a partial tag start
                # e.g. "<tool_" or "<tool_ca" — don't yield yet, wait for more
                might_be_tag = False
                for i in range(1, min(TAG_MAX_LEN, len(pending_text) + 1)):
                    if pending_text.endswith(START_TAG[:i]):
                        might_be_tag = True
                        break

                if might_be_tag:
                    # Hold the potential partial tag — yield everything before it
                    safe_end = len(pending_text) - TAG_MAX_LEN
                    if safe_end > 0:
                        yield _sse_chunk(response_id, created_time, model,
                                         delta={"content": pending_text[:safe_end]})
                        pending_text = pending_text[safe_end:]
                    continue

                # ── FALLBACK: Buffer potential raw JSON if it starts with { ──
                stripped_pending = pending_text.strip()
                if not has_tool_calls and stripped_pending.startswith("{"):
                    # We have something that looks like JSON — buffer until it closes or gets too long
                    if "}" in stripped_pending:
                        try:
                            # Try to parse the WHOLE stripped buffer
                            call_data = json.loads(stripped_pending)
                            if "name" in call_data:
                                logger.info("Fallback: Detected multi-chunk JSON tool call", name=call_data.get("name"))
                                call_id = f"call_{uuid.uuid4().hex[:12]}"
                                yield _sse_chunk(response_id, created_time, model,
                                    delta={
                                        "tool_calls": [{
                                            "index": tool_call_index,
                                            "id": call_id,
                                            "type": "function",
                                            "function": {
                                                "name": call_data.get("name", ""),
                                                "arguments": json.dumps(call_data.get("arguments", {}), ensure_ascii=False)
                                            }
                                        }]
                                    })
                                tool_call_index += 1
                                has_tool_calls = True
                                pending_text = ""
                                continue
                        except json.JSONDecodeError:
                            # Still incomplete or not valid JSON yet — keep buffering
                            pass

                    # If it doesn't close or isn't valid yet, we might want to wait
                    # BUT we shouldn't wait forever if it's just regular text starting with {
                    if len(stripped_pending) < 1000: # Max buffer for raw JSON
                        continue

                # No tag or valid JSON tool call — yield everything
                yield _sse_chunk(response_id, created_time, model,
                                 delta={"content": pending_text})
                pending_text = ""

        except Exception as e:
            logger.error("Streaming generation error", error=str(e))
            yield _sse_chunk(response_id, created_time, model,
                             delta={"content": f"\n\n[Error: {e}]"})

        # Finish — use "tool_calls" finish reason if any tool was emitted
        finish = "tool_calls" if has_tool_calls else "stop"
        yield _sse_chunk(response_id, created_time, model,
                         delta={}, finish_reason=finish)
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


def _build_metadata_block(result) -> str:
    """Build a human-readable metadata block for test generation results."""
    parts = []
    parts.append("---")
    parts.append(f"**Validation:** {'✅ passed' if getattr(result, 'validation_passed', False) else ' failed'}")

    issues = getattr(result, 'validation_issues', [])
    if issues:
        parts.append(f"**Issues:** {', '.join(issues[:5])}")

    repair = getattr(result, 'repair_attempts', 0)
    if repair:
        parts.append(f"**Repair attempts:** {repair}")

    rag = getattr(result, 'rag_chunks_used', 0)
    if rag:
        parts.append(f"**RAG context chunks:** {rag}")

    vs = getattr(result, 'validation_summary', None)
    if vs:
        parts.append(
            f"**Details:** {vs.get('errors', 0)} errors, "
            f"{vs.get('warnings', 0)} warnings, "
            f"{vs.get('test_count', '?')} test(s)"
        )

    parts.append(f"**Tokens used:** {getattr(result, 'tokens_used', 0)}")
    
    # Two-Phase Strategy metadata
    strategy = getattr(result, 'strategy_used', 'single_pass')
    if strategy == "two_phase":
        parts.append(f"**Strategy:** ✅ Two-Phase Generation")
        complexity = getattr(result, 'complexity_score', 0)
        if complexity:
            parts.append(f"**Complexity score:** {complexity}")
    else:
        parts.append(f"**Strategy:** ✅ Single-Pass")
    
    return "\n".join(parts)
