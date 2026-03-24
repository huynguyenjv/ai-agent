"""Node: tool_selector — Agent Tool Call Flow.

Deterministically selects MCP tools based on intent, gate outputs, and Qdrant cache state.
Agent logic decides tools — not the LLM.
"""

from __future__ import annotations

import json
import re
import uuid

from server.agent.state import AgentState

# Regex to extract a symbol name from the user's message for search intent
_SYMBOL_PATTERN = re.compile(r"\b([A-Z][a-zA-Z0-9]+|`([^`]+)`|\"([^\"]+)\")\b")


def _make_tool_call(name: str, arguments: dict, index: int = 0) -> dict:
    """Build an OpenAI-format tool call dict."""
    return {
        "index": index,
        "id": f"call_{uuid.uuid4().hex[:8]}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments),
        },
    }


def _extract_symbol_name(messages: list) -> str:
    """Extract a symbol name from the last user message for search_symbol."""
    if not messages:
        return "unknown"
    last = messages[-1]
    text = last.content if hasattr(last, "content") else str(last)
    match = _SYMBOL_PATTERN.search(text)
    if match:
        return match.group(2) or match.group(3) or match.group(1)
    return "unknown"


async def tool_selector(state: AgentState, qdrant=None) -> dict:
    """Select MCP tools based on intent + gate outputs + Qdrant cache state.

    Returns:
        pending_tool_calls: list of OpenAI tool call dicts (empty = no tools needed)
        tool_turns_used: incremented if tools are emitted
    """
    # Cap: never emit tools more than once per request
    if state.get("tool_turns_used", 0) >= 1 or state.get("is_tool_result_turn", False):
        return {
            "pending_tool_calls": [],
            "tool_turns_used": state.get("tool_turns_used", 0),
        }

    intent = state.get("intent", "code_gen")
    mentioned_files: list[str] = state.get("mentioned_files", [])
    active_file: str | None = state.get("active_file")
    freshness_signal: bool = state.get("freshness_signal", False)
    messages = state.get("messages", [])

    tool_calls: list[dict] = []

    if intent == "structural_analysis":
        tool_calls = [_make_tool_call("get_project_skeleton", {"include_methods": True})]

    elif intent == "search":
        name = _extract_symbol_name(messages)
        tool_calls = [_make_tool_call("search_symbol", {"name": name, "type_filter": "any"})]

    elif intent in ("code_gen", "refine", "unit_test"):
        if mentioned_files:
            file = mentioned_files[0]
            # Check Qdrant cache
            count = await qdrant.count_by_file(file) if qdrant else 0
            cache_hit = count > 0

            if not cache_hit or freshness_signal:
                # Miss or stale: index + read
                tool_calls = [
                    _make_tool_call("index_with_deps", {"file_path": file, "depth": 2}, index=0),
                    _make_tool_call("read_file", {"file_path": file}, index=1),
                ]
            else:
                # Hit, no freshness: trust cache, just read
                tool_calls = [_make_tool_call("read_file", {"file_path": file})]

        elif active_file:
            count = await qdrant.count_by_file(active_file) if qdrant else 0
            if count == 0:
                tool_calls = [_make_tool_call("index_with_deps", {"file_path": active_file, "depth": 2})]
            # else: hit + no freshness + no explicit file -> RAG sufficient, no tools

    elif intent == "explain":
        if mentioned_files:
            tool_calls = [_make_tool_call("read_file", {"file_path": mentioned_files[0]})]
        # else: RAG sufficient

    if not tool_calls:
        return {
            "pending_tool_calls": [],
            "tool_turns_used": state.get("tool_turns_used", 0),
        }

    return {
        "pending_tool_calls": tool_calls,
        "tool_turns_used": state.get("tool_turns_used", 0) + 1,
    }
