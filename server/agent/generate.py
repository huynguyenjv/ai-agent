"""Node: generate — native tool-call streaming.

Forwards merged tools (server registry + client) to vLLM, streams content
tokens, and captures tool_call deltas to accumulate a final pending_tool_calls
list for Turn 2.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from openai import AsyncOpenAI

from server.agent.state import AgentState

logger = logging.getLogger("server.agent.generate")

# =============================================================================
# MCP Tools Schema
# =============================================================================

MCP_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "vtrip_read_file",
            "description": "Read a contiguous range of lines from a file. Used when you need exact, guaranteed-fresh content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path relative to repo root"},
                    "start_line": {"type": "integer", "description": "Start line (1-based), default 1", "default": 1},
                    "end_line": {"type": "integer", "description": "End line (1-based), default 150", "default": 150},
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "vtrip_search_symbol",
            "description": "Locate a class, function, or method by name anywhere in the repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Symbol name to search for"},
                    "type_filter": {"type": "string", "enum": ["class", "function", "method", "any"], "default": "any"},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "vtrip_get_project_skeleton",
            "description": "Return a compact structural overview of the entire repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "include_methods": {"type": "boolean", "default": True},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "vtrip_index_with_deps",
            "description": "Parse a specific file and its dependencies, upload chunks for embedding.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path relative to repo root"},
                    "depth": {"type": "integer", "default": 2},
                },
                "required": ["file_path"],
            },
        },
    },
]

MCP_TOOL_NAMES = {t["function"]["name"] for t in MCP_TOOLS}

# =============================================================================
# Intent-based System Prompts
# =============================================================================

INTENT_PROMPTS = {
    "code_gen": "You are an expert coding assistant. Write clean, efficient code. Respond in the user's language.",
    "code_review": "You are a senior code reviewer. Analyze for bugs, security, performance. Respond in the user's language.",
    "explain": "You are a patient coding teacher. Explain clearly with examples. Respond in the user's language.",
    "search": "You are a codebase navigator. Find symbols and files precisely. Respond in the user's language.",
    "structural_analysis": "You are a software architect. Analyze structure and dependencies. Respond in the user's language.",
    "refine": "You are a refactoring expert. Improve code quality. Respond in the user's language.",
    "unit_test": "You are a testing expert. Write comprehensive tests. Respond in the user's language.",
}

DEFAULT_PROMPT = "You are an expert coding assistant. Respond in the user's language."

TOOL_INSTRUCTIONS = """

You have access to these tools - use them when needed:
- vtrip_read_file: Read file content (file_path, start_line, end_line)
- vtrip_search_symbol: Find symbols in codebase (name, type_filter)
- vtrip_get_project_skeleton: Get project structure overview
- vtrip_index_with_deps: Index file with dependencies"""

# =============================================================================
# Tool Name Mapping (for models trained on different tool sets)
# =============================================================================

TOOL_NAME_MAP = {
    "ls": "vtrip_get_project_skeleton",
    "list_files": "vtrip_get_project_skeleton",
    "list_directory": "vtrip_get_project_skeleton",
    "read_file": "vtrip_read_file",
    "cat": "vtrip_read_file",
    "view_file": "vtrip_read_file",
    "search": "vtrip_search_symbol",
    "grep": "vtrip_search_symbol",
    "find_symbol": "vtrip_search_symbol",
    "index_file": "vtrip_index_with_deps",
}

# Argument name mapping per tool
ARG_NAME_MAP = {
    "vtrip_read_file": {
        "path": "file_path",
        "filePath": "file_path",
        "dirPath": "file_path",
        "filename": "file_path",
    },
    "vtrip_search_symbol": {
        "query": "name",
        "symbol": "name",
        "search": "name",
    },
}

# =============================================================================
# Configuration
# =============================================================================

MAX_TOOL_TURNS = 15  # Allow more tool turns for complex tasks
MAX_INPUT_TOKENS = 24000


# =============================================================================
# Helper Functions
# =============================================================================

def _estimate_tokens(text: str) -> int:
    """Rough token estimate."""
    if not text:
        return 0
    return len(text) // 3


def _normalize_json(obj: Any) -> str:
    """Normalize JSON for comparison (sorted keys, no extra spaces)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


# Pattern to match <tool_call>...</tool_call> or <tool_name>...</tool_name> tags
_TOOL_CALL_TAG_PATTERN = re.compile(
    r"<tool_call>\s*\{.*?\}\s*</tool_call>|"  # <tool_call>{"name":...}</tool_call>
    r"<(vtrip_\w+)>.*?</\1>",  # <vtrip_read_file>...</vtrip_read_file>
    re.DOTALL | re.IGNORECASE
)


def _strip_tool_call_tags(content: str) -> str:
    """Remove <tool_call> and <vtrip_*> tags from content.

    Models sometimes output both native tool_calls AND text-based tags.
    We strip the text tags to avoid showing them to the user.
    """
    if not content:
        return content

    cleaned = _TOOL_CALL_TAG_PATTERN.sub("", content)
    # Clean up extra whitespace left behind
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _parse_json_safe(s: str) -> dict:
    """Parse JSON safely, return empty dict on error."""
    if not s:
        return {}
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return {}


def _to_openai_messages(state: AgentState, tools_disabled: bool = False) -> list[dict]:
    """Convert LangChain messages to OpenAI format, with truncation."""
    intent = state.get("intent", "code_gen")
    system_prompt = INTENT_PROMPTS.get(intent, DEFAULT_PROMPT)

    if tools_disabled:
        system_prompt += (
            "\n\nIMPORTANT: You have already used all available tool calls. "
            "DO NOT output any <tool_call> tags or attempt to call tools. "
            "Based on the information you have gathered, provide your final response directly."
        )
    else:
        system_prompt += TOOL_INSTRUCTIONS

    out: list[dict] = [{"role": "system", "content": system_prompt}]

    # Collect messages (skip system messages from client)
    messages: list[dict] = []
    for msg in state.get("messages", []):
        mtype = getattr(msg, "type", None)

        if mtype == "system":
            continue  # Skip client system prompts

        elif mtype == "human":
            content = msg.content
            if isinstance(content, list):
                content = " ".join(
                    item.get("text", "") if isinstance(item, dict) else str(item)
                    for item in content
                )
            messages.append({"role": "user", "content": content or ""})

        elif mtype == "ai":
            item: dict = {"role": "assistant", "content": msg.content or ""}
            tc = (getattr(msg, "additional_kwargs", {}) or {}).get("tool_calls")
            if tc:
                item["tool_calls"] = tc
                item["content"] = None
            messages.append(item)

        elif mtype == "tool":
            tool_call_id = getattr(msg, "tool_call_id", "") or "unknown"
            messages.append({
                "role": "tool",
                "content": msg.content or "",
                "tool_call_id": tool_call_id,
            })

        else:
            content = getattr(msg, "content", str(msg))
            messages.append({"role": "user", "content": content or ""})

    # Truncate old messages to fit token budget (keep most recent)
    total_tokens = _estimate_tokens(system_prompt)
    kept_messages: list[dict] = []

    for msg in reversed(messages):
        content = msg.get("content") or ""
        msg_tokens = _estimate_tokens(content)

        if total_tokens + msg_tokens > MAX_INPUT_TOKENS:
            # Truncate this message if it's the last one we can include
            remaining = MAX_INPUT_TOKENS - total_tokens
            if remaining > 500 and content:
                half = (remaining * 3) // 2
                msg["content"] = content[:half] + "\n...[truncated]...\n" + content[-half:]
                kept_messages.append(msg)
            break

        total_tokens += msg_tokens
        kept_messages.append(msg)

    # Reverse to restore chronological order and add after system prompt
    out.extend(reversed(kept_messages))

    logger.info("_to_openai_messages: %d messages, ~%d tokens", len(out), total_tokens)
    return out


def _merge_tool_call_delta(acc: list[dict], delta_list: list) -> None:
    """Merge streaming tool_call deltas into accumulator."""
    for d in delta_list:
        if hasattr(d, "model_dump"):
            d = d.model_dump()

        idx = d.get("index", 0)

        # Expand accumulator if needed
        while len(acc) <= idx:
            acc.append({
                "id": "",
                "type": "function",
                "function": {"name": "", "arguments": ""},
            })

        slot = acc[idx]

        if d.get("id"):
            slot["id"] = d["id"]
        if d.get("type"):
            slot["type"] = d["type"]

        fn = d.get("function") or {}
        if fn.get("name"):
            slot["function"]["name"] += fn["name"]
        if fn.get("arguments"):
            slot["function"]["arguments"] += fn["arguments"]


def _map_and_validate_tool_calls(tool_calls: list[dict]) -> list[dict]:
    """Map tool names and validate arguments. Filter out invalid calls."""
    valid_calls: list[dict] = []

    for tc in tool_calls:
        fn = tc.get("function") or {}
        name = fn.get("name", "").strip()
        args = _parse_json_safe(fn.get("arguments", "{}"))

        # Skip empty tool calls
        if not name:
            logger.warning("Skipping tool call with empty name")
            continue

        # Map tool name if needed
        original_name = name
        if name in TOOL_NAME_MAP:
            name = TOOL_NAME_MAP[name]
            logger.info("Mapped tool: %s -> %s", original_name, name)

        # Skip unknown tools
        if name not in MCP_TOOL_NAMES:
            logger.warning("Skipping unknown tool: %s", name)
            continue

        # Map argument names
        arg_map = ARG_NAME_MAP.get(name, {})
        mapped_args = {}
        for k, v in args.items():
            mapped_key = arg_map.get(k, k)
            mapped_args[mapped_key] = v

        # Validate required arguments
        if name == "vtrip_read_file":
            file_path = mapped_args.get("file_path", "")
            if not file_path:
                logger.warning("Skipping vtrip_read_file: empty file_path")
                continue
            # Set defaults for optional args
            mapped_args.setdefault("start_line", 1)
            mapped_args.setdefault("end_line", 150)

        elif name == "vtrip_search_symbol":
            symbol_name = mapped_args.get("name", "")
            if not symbol_name:
                logger.warning("Skipping vtrip_search_symbol: empty name")
                continue
            mapped_args.setdefault("type_filter", "any")

        elif name == "vtrip_get_project_skeleton":
            mapped_args.setdefault("include_methods", True)

        elif name == "vtrip_index_with_deps":
            file_path = mapped_args.get("file_path", "")
            if not file_path:
                logger.warning("Skipping vtrip_index_with_deps: empty file_path")
                continue
            mapped_args.setdefault("depth", 2)

        valid_calls.append({
            "id": tc.get("id") or f"call_{len(valid_calls)}",
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(mapped_args),
            },
        })

    return valid_calls


def _get_already_called_tools(state: AgentState) -> set[str]:
    """Get normalized signatures of already-called tools."""
    already_called: set[str] = set()

    for msg in state.get("messages", []):
        if getattr(msg, "type", None) == "ai":
            tc_list = (getattr(msg, "additional_kwargs", {}) or {}).get("tool_calls", [])
            for tc in tc_list:
                fn = tc.get("function", {})
                name = fn.get("name", "")
                args = _parse_json_safe(fn.get("arguments", "{}"))
                # Normalize: sorted keys, no whitespace variations
                call_key = f"{name}:{_normalize_json(args)}"
                already_called.add(call_key)

    return already_called


def _deduplicate_tool_calls(
    tool_calls: list[dict],
    already_called: set[str],
) -> list[dict]:
    """Remove duplicate tool calls."""
    deduped: list[dict] = []

    for tc in tool_calls:
        fn = tc.get("function", {})
        name = fn.get("name", "")
        args = _parse_json_safe(fn.get("arguments", "{}"))
        call_key = f"{name}:{_normalize_json(args)}"

        if call_key in already_called:
            logger.warning("Skipping duplicate tool call: %s", name)
            continue

        already_called.add(call_key)  # Prevent duplicates within same batch
        deduped.append(tc)

    return deduped


# =============================================================================
# Main Generate Function
# =============================================================================

async def generate(
    state: AgentState,
    vllm_client: AsyncOpenAI,
    model: str,
    sse_callback=None,
) -> dict:
    """Generate response, possibly with tool calls."""
    tool_turns_used = state.get("tool_turns_used", 0)

    # Check tool turn limit
    tools_disabled = tool_turns_used >= MAX_TOOL_TURNS
    if tools_disabled:
        logger.warning(
            "Tool turn limit reached (%d/%d), generating without tools",
            tool_turns_used, MAX_TOOL_TURNS
        )
        all_tools = None
    else:
        # Merge MCP tools with client tools (MCP takes priority)
        client_tools = state.get("client_tools") or []
        extra_tools = [
            t for t in client_tools
            if t.get("function", {}).get("name") not in MCP_TOOL_NAMES
        ]
        all_tools = MCP_TOOLS + extra_tools if MCP_TOOLS or extra_tools else None

    messages = _to_openai_messages(state, tools_disabled=tools_disabled)

    # Calculate max_tokens dynamically based on estimated input
    # Model context: ~32k, reserve enough for output
    estimated_input = sum(_estimate_tokens(m.get("content") or "") for m in messages)
    MODEL_CONTEXT = 32000
    max_tokens = min(8192, max(1024, MODEL_CONTEXT - estimated_input - 1000))

    logger.info("generate: estimated_input=%d, max_tokens=%d", estimated_input, max_tokens)

    # Build request kwargs
    kwargs: dict = {
        "model": model,
        "messages": messages,
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }

    # Only add tools if we have them
    if all_tools:
        kwargs["tools"] = all_tools
        tool_choice = state.get("tool_choice")
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice

    tool_names = [t["function"]["name"] for t in (all_tools or [])]
    logger.info("generate: tools=%s, tool_turns=%d/%d", tool_names, tool_turns_used, MAX_TOOL_TURNS)

    # Stream response
    content_buf: list[str] = []
    tool_calls_acc: list[dict] = []

    try:
        stream = await vllm_client.chat.completions.create(**kwargs)

        async for chunk in stream:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            # Collect content
            if getattr(delta, "content", None):
                token = delta.content
                content_buf.append(token)
                if sse_callback:
                    await sse_callback("content", token)

            # Collect tool calls
            if getattr(delta, "tool_calls", None):
                _merge_tool_call_delta(tool_calls_acc, delta.tool_calls)

    except Exception as e:
        logger.error("vLLM generation failed: %s", e)
        if sse_callback:
            await sse_callback("error", str(e))
        return {
            "draft": f"Generation error: {e}",
            "pending_tool_calls": [],
            "tool_turns_used": tool_turns_used,
        }

    # Process results
    draft = "".join(content_buf)

    # Strip <tool_call> tags from content (model may output both native + text tags)
    original_len = len(draft)
    draft = _strip_tool_call_tags(draft)
    if len(draft) != original_len:
        logger.info("generate: stripped tool_call tags from content (%d -> %d chars)", original_len, len(draft))

    # Fallback: if content is empty and no tool calls, provide a message
    if not draft and not tool_calls_acc:
        if tools_disabled:
            draft = (
                "Tôi đã thu thập đủ thông tin từ các công cụ. "
                "Tuy nhiên, tôi cần thêm context để hoàn thành yêu cầu. "
                "Vui lòng cung cấp thêm chi tiết hoặc thử lại với câu hỏi cụ thể hơn."
            )
            logger.warning("generate: empty response with tools disabled, using fallback message")
        else:
            logger.warning("generate: empty response, no content or tool calls")

    logger.info(
        "generate: content_len=%d, raw_tool_calls=%d",
        len(draft), len(tool_calls_acc)
    )

    # Validate and map tool calls
    valid_tool_calls = _map_and_validate_tool_calls(tool_calls_acc)

    # Deduplicate
    already_called = _get_already_called_tools(state)
    final_tool_calls = _deduplicate_tool_calls(valid_tool_calls, already_called)

    if final_tool_calls:
        logger.info(
            "generate: final_tool_calls=%d (raw=%d, valid=%d)",
            len(final_tool_calls), len(tool_calls_acc), len(valid_tool_calls)
        )

    # Increment tool_turns_used if we're returning tool calls
    new_tool_turns = tool_turns_used + 1 if final_tool_calls else tool_turns_used

    return {
        "draft": draft,
        "pending_tool_calls": final_tool_calls,
        "tool_turns_used": new_tool_turns,
    }
