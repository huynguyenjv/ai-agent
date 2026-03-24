"""Node: generate — Section 8.

Assembles system prompt with RAG context, streams tokens via vLLM.
"""

from __future__ import annotations

import logging

from openai import AsyncOpenAI

from server.agent.state import AgentState

logger = logging.getLogger("server.agent.generate")

BASE_SYSTEM_PROMPT = """You are an expert coding assistant. You provide accurate, well-structured code responses grounded in the actual codebase context provided below. Always reference the codebase context when generating code. If the context is insufficient, say so clearly.

Respond in the same language as the user's query (Vietnamese or English)."""


async def generate(
    state: AgentState,
    vllm_client: AsyncOpenAI,
    model: str,
    sse_callback=None,
) -> dict:
    """Assemble prompt and stream LLM response.

    Section 8:
    1. System prompt + RAG context + plan (for code_gen)
    2. Conversation history
    3. Stream via vLLM with stream=True
    4. Emit content SSE delta events
    """
    intent = state.get("intent", "code_gen")
    rag_chunks = state.get("rag_chunks", [])
    tool_results = state.get("tool_results", [])

    # Build system prompt
    parts = [BASE_SYSTEM_PROMPT]

    # Add RAG context as labeled code blocks
    if rag_chunks:
        parts.append("\n## Codebase Context\n")
        for chunk in rag_chunks:
            symbol = chunk.get("symbol_name", "unknown")
            lang = chunk.get("lang", "")
            file_path = chunk.get("file_path", "")
            body = chunk.get("body", "")
            embed = chunk.get("embed_text", "")

            text = body if body else embed
            if text:
                parts.append(f"### {symbol} ({file_path})")
                parts.append(f"```{lang}\n{text}\n```\n")

    # Add plan for code_gen
    if intent == "code_gen" and tool_results:
        for result in tool_results:
            if "plan" in result:
                parts.append(f"\n## Execution Plan\n{result['plan']}\n")

    system_prompt = "\n".join(parts)

    # Build messages
    llm_messages = [{"role": "system", "content": system_prompt}]

    for msg in state.get("messages", []):
        if hasattr(msg, "type"):
            role = "assistant" if msg.type == "ai" else "user"
            content = msg.content
        elif isinstance(msg, dict):
            role = msg.get("role", "user")
            content = msg.get("content", "")
        else:
            role = "user"
            content = str(msg)
        llm_messages.append({"role": role, "content": content})

    # Stream from vLLM
    full_response = []
    try:
        stream = await vllm_client.chat.completions.create(
            model=model,
            messages=llm_messages,
            max_tokens=4096,
            temperature=0.3,
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                full_response.append(token)
                if sse_callback:
                    await sse_callback("content", token)

    except Exception as e:
        logger.error("vLLM generation failed: %s", e)
        error_msg = f"Generation error: {str(e)}"
        full_response.append(error_msg)
        if sse_callback:
            await sse_callback("error", error_msg)

    draft = "".join(full_response)
    return {
        "draft": draft,
        "context_assembled": system_prompt,
    }
