"""
Chat completions endpoints (OpenAI-compatible).

Includes: /v1/chat/completions, /v1/completions, /v1/rag-context
"""

import re
import time
import uuid
import re as _re
from typing import Optional

import structlog
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from agent.models import GenerationRequest
from ..dependencies import (
    graph_orchestrator, tool_orchestrator, _executor,
    _get_orchestrator, run_in_executor,
)
from ..schemas import (
    ChatMessage, ChatCompletionRequest, ChatCompletionResponse,
)
from ..services.stream import (
    _sse_chunk, _stream_from_vllm, _stream_test_generation,
)
from ..services.rag_resolver import _resolve_collection, _resolve_repo_path, _enrich_with_rag

logger = structlog.get_logger()

router = APIRouter()


def _extract_java_file_path_from_message(content: str) -> Optional[str]:
    """Extract Java file path from inline code blocks sent by Continue IDE.

    Continue IDE format when user uses @ClassName:
        ```src/main/java/com/example/ClassName.java
        package com.example;
        ...
        ```

    Returns the file path if found, None otherwise.
    """
    # Pattern 1: code block with file path as info string
    # ```src/main/java/.../ClassName.java  or  ```java src/.../ClassName.java
    code_block_pattern = r'```(?:java\s+)?([^\n]*?\.java)\s*\n'
    match = re.search(code_block_pattern, content)
    if match:
        return match.group(1).strip()

    # Pattern 2: look for class declaration to infer class name as fallback
    class_pattern = r'(?:public\s+)?class\s+(\w+)'
    class_match = re.search(class_pattern, content)
    if class_match:
        class_name = class_match.group(1)
        # Try to find package declaration for a better path
        pkg_match = re.search(r'package\s+([\w.]+)\s*;', content)
        if pkg_match:
            package_path = pkg_match.group(1).replace('.', '/')
            return f"src/main/java/{package_path}/{class_name}.java"
        return f"{class_name}.java"

    return None


@router.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint."""
    # ── DEBUGGING: Log incoming request from Continue ────────────────
    logger.debug("Incoming ChatCompletionRequest",
                 model=request.model,
                 stream=request.stream,
                 msg_count=len(request.messages),
                 has_tools=bool(request.tools))
    if request.tools:
        logger.debug("Tools provided by IDE", count=len(request.tools), tools=[t.function.name for t in request.tools])

    active = _get_orchestrator()
    if not active:
        raise HTTPException(status_code=503, detail="Service not initialized")

    # ── Parse messages ───────────────────────────────────────────────
    user_messages = [m for m in request.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message provided")

    user_content = user_messages[-1].content or ""
    system_messages = [m for m in request.messages if m.role == "system"]
    system_content = system_messages[0].content if system_messages else ""

    # Filter overly-long IDE agent system prompts
    if system_content and any(marker in system_content for marker in [
        "<important_rules>", "You are in agent mode",
        "TOOL_NAME:", "read_file tool", "create_new_file tool",
    ]):
        logger.debug("Filtering overly-long IDE system prompt", original_len=len(system_content))
        system_content = ""

    # ── Handle tool calls from Continue ──────────────────────────────
    # If Continue sent tool results, include them in the message history
    # and forward to vLLM for the next response.
    tool_messages = [m for m in request.messages if m.role == "tool"]

    # ── Detect request intent ────────────────────────────────────────
    is_test_request = any(
        kw in user_content.lower()
        for kw in ["test", "junit", "unit test", "mockito", "generate test"]
    )

    file_path = request.file_path
    if not file_path:
        file_path = _extract_java_file_path_from_message(user_content)

    # ── Resolve Repository and Collection (3-tier) ───────────────────
    resolved_collection = _resolve_collection(
        explicit=request.collection,
        file_path=file_path,
        workspace_path=request.workspace_path,
    )
    resolved_repo_path = _resolve_repo_path(
        workspace_path=request.workspace_path,
        collection=resolved_collection,
        file_path=file_path,
    )
    logger.info(
        "Resolved context",
        collection=resolved_collection,
        repo_path=resolved_repo_path,
    )

    response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created_time = int(time.time())

    # ═════════════════════════════════════════════════════════════════
    # PATH 1: Test generation pipeline (supports Two-Phase Strategy)
    # ═════════════════════════════════════════════════════════════════
    if is_test_request and file_path:
        logger.info(
            "Using test generation pipeline",
            file_path=file_path,
            collection=resolved_collection,
            force_two_phase=request.force_two_phase,
        )
        
        # Extract source code from user message if present
        source_code = None
        code_match = _re.search(r'```(?:[^\n]*\.java)?\s*\n(.*?)```', user_content, _re.DOTALL)
        if code_match:
            source_code = code_match.group(1).strip()
        
        gen_request = GenerationRequest(
            file_path=file_path,
            task_description=user_content,
            collection_name=resolved_collection,
            source_code=source_code,
            repo_path=resolved_repo_path,
            # Two-Phase Strategy: always on
            force_two_phase=True,
            force_single_pass=False,
            complexity_threshold=request.complexity_threshold or 10,
        )

        # ── REAL STREAMING: progress phases + token-by-token code ────
        # Chat completions always stream; ignore request.stream here.
        return _stream_test_generation(
            response_id, created_time, request.model, gen_request,
        )

    # ═════════════════════════════════════════════════════════════════
    # PATH 2: General chat — supports real streaming from vLLM
    # ═════════════════════════════════════════════════════════════════

    # Build the enhanced prompt (with optional RAG context)
    enhanced_prompt = user_content
    if file_path:
        enhanced_prompt = await _enrich_with_rag(user_content, file_path, resolved_collection)

    # ── Resolve System Prompt ────────────────────────────────────────
    # If system_content was filtered (empty), use a concise default.
    if not system_content:
        effective_system = (
            "You are a Senior Java Backend Developer (Java 21, Spring Boot 3.x).\n"
            "Provide concise, high-quality code and answers. "
            "Use available tools when necessary to fulfill user requests."
        )
    else:
        effective_system = system_content

    logger.info("Effective system prompt resolved", length=len(effective_system))
    logger.debug("Effective system prompt content", content=effective_system[:200] + "...")

    # ── ADVANCED: Inject tool-calling instructions ──────────────────
    if request.tools:
        tool_rules = tool_orchestrator.build_tool_system_prompt(request.tools)
        effective_system += f"\n\n{tool_rules}"
        logger.info("Injected tool calling instructions", tool_count=len(request.tools))
        logger.debug("Effective System Prompt", content=effective_system[:500] + "...")

    # ── Build multi-turn messages ───────────────────────────────────
    # We maintain full history so vLLM sees previous user/assistant 
    # messages and tool results.
    multi_turn_messages = [{"role": "system", "content": effective_system}]
    for msg in request.messages:
        if msg.role == "system":
            continue
        m: dict = {"role": msg.role, "content": msg.content or ""}
        if msg.tool_call_id:
            m["tool_call_id"] = msg.tool_call_id
        
        # Convert assistant's tool_calls back to XML for model context
        if msg.role == "assistant" and msg.tool_calls:
            tc_text = "\n".join(
                f'<tool_call>\n{{"name": "{tc.function.name}", "arguments": {tc.function.arguments}}}\n</tool_call>'
                for tc in msg.tool_calls
            )
            m["content"] = (m["content"] or "") + "\n" + tc_text
        multi_turn_messages.append(m)

    # ── REAL STREAMING ───────────────────────────────────────────────
    return _stream_from_vllm(
        response_id, created_time, request.model,
        effective_system, enhanced_prompt,
        request.temperature, request.max_tokens,
        messages=multi_turn_messages,
    )


@router.post("/v1/completions")
async def completions(request: dict):
    """OpenAI-compatible completions endpoint (legacy). Redirects to chat."""
    prompt = request.get("prompt", "")
    messages = [ChatMessage(role="user", content=prompt)]

    chat_request = ChatCompletionRequest(
        model=request.get("model", "ai-agent"),
        messages=messages,
        temperature=request.get("temperature", 0.2),
        max_tokens=request.get("max_tokens", 4096),
    )

    response = await chat_completions(chat_request)

    # If it's a streaming response, return as-is
    if isinstance(response, StreamingResponse):
        return response

    return {
        "id": response.id,
        "object": "text_completion",
        "created": response.created,
        "model": response.model,
        "choices": [
            {
                "text": response.choices[0].message.content if response.choices[0].message else "",
                "index": 0,
                "finish_reason": "stop",
            }
        ],
        "usage": response.usage.model_dump() if response.usage else {},
    }


# ============================================================================
# RAG Context Inspection
# ============================================================================

@router.get("/v1/rag-context")
async def get_rag_context(
    class_name: str,
    file_path: str = "",
    session_id: str = "",
):
    """Return RAG chunks used to build the prompt for a given class.

    Useful for debugging and transparency — lets callers see
    exactly what context the agent retrieves from the vector DB.
    """
    if not graph_orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")

    session = None
    if session_id:
        session = await run_in_executor(
            graph_orchestrator.memory_manager.get_session, session_id
        )

    chunks = await run_in_executor(
        graph_orchestrator._get_rag_context, class_name, file_path, session
    )
    return chunks
