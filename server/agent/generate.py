"""Node: generate — native tool-call streaming.

Forwards merged tools (server registry + client) to vLLM, streams content
tokens, and captures tool_call deltas to accumulate a final pending_tool_calls
list for Turn 2.
"""

from __future__ import annotations

import logging

from openai import AsyncOpenAI

from server.agent.state import AgentState

logger = logging.getLogger("server.agent.generate")

BASE_SYSTEM_PROMPT = (
    "You are an expert coding assistant. Provide accurate, well-structured answers. "
    "If you need to inspect files or search the codebase, call the provided tools. "
    "Respond in the same language as the user's query (Vietnamese or English)."
)


def _to_openai_messages(state: AgentState) -> list[dict]:
    out: list[dict] = [{"role": "system", "content": BASE_SYSTEM_PROMPT}]
    for msg in state.get("messages", []):
        mtype = getattr(msg, "type", None)
        if mtype == "human":
            out.append({"role": "user", "content": msg.content or ""})
        elif mtype == "ai":
            item: dict = {"role": "assistant", "content": msg.content or ""}
            tc = (getattr(msg, "additional_kwargs", {}) or {}).get("tool_calls")
            if tc:
                item["tool_calls"] = tc
                item["content"] = None
            out.append(item)
        elif mtype == "tool":
            out.append({
                "role": "tool",
                "content": msg.content or "",
                "tool_call_id": getattr(msg, "tool_call_id", "") or "",
            })
        elif mtype == "system":
            out.append({"role": "system", "content": msg.content or ""})
        else:
            out.append({"role": "user", "content": str(getattr(msg, "content", msg))})
    return out


def _merge_tool_call_delta(acc: list[dict], delta_list: list) -> None:
    for d in delta_list:
        if hasattr(d, "model_dump"):
            d = d.model_dump()
        idx = d.get("index", 0)
        while len(acc) <= idx:
            acc.append({"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
        slot = acc[idx]
        if d.get("id"):
            slot["id"] = d["id"]
        if d.get("type"):
            slot["type"] = d["type"]
        fn = d.get("function") or {}
        name = fn.get("name")
        if name:
            slot["function"]["name"] += name
        args = fn.get("arguments")
        if args:
            slot["function"]["arguments"] += args


async def generate(
    state: AgentState,
    vllm_client: AsyncOpenAI,
    model: str,
    sse_callback=None,
) -> dict:
    client_tools = state.get("client_tools") or []
    messages = _to_openai_messages(state)

    kwargs: dict = {
        "model": model,
        "messages": messages,
        "stream": True,
        **({"tools": client_tools} if client_tools else {}),
        "max_tokens": 4096,
        "temperature": 0.3,
    }
    tc = state.get("tool_choice")
    if tc is not None:
        kwargs["tool_choice"] = tc

    content_buf: list[str] = []
    tool_calls_acc: list[dict] = []

    try:
        stream = await vllm_client.chat.completions.create(**kwargs)
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if getattr(delta, "content", None):
                token = delta.content
                content_buf.append(token)
                if sse_callback:
                    await sse_callback("content", token)
            if getattr(delta, "tool_calls", None):
                _merge_tool_call_delta(tool_calls_acc, delta.tool_calls)
    except Exception as e:
        logger.error("vLLM generation failed: %s", e)
        if sse_callback:
            await sse_callback("error", str(e))
        return {"draft": f"Generation error: {e}", "pending_tool_calls": []}

    return {
        "draft": "".join(content_buf),
        "pending_tool_calls": tool_calls_acc,
    }
