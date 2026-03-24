"""LangGraph Agent — Section 8, Graph Structure + Tool Call Flow.

Directed graph with conditional edges.

Turn 1 (normal):
  classify_intent -> route_context -> tool_selector
    -> [pending_tool_calls]: emit_tool_calls -> END
    -> [no tools, structural_analysis]: generate -> post_process -> END
    -> [no tools, other]: rag_search -> plan_steps/generate -> post_process -> END
    -> [volatile_rejected]: reject_volatile -> END

Turn 2 (tool results):
  classify_intent -> tool_selector (skip route_context) -> rag_search -> generate -> END
"""

from __future__ import annotations

from functools import partial

from langgraph.graph import END, StateGraph

from server.agent.state import AgentState
from server.agent.classify_intent import classify_intent
from server.agent.route_context import route_context
from server.agent.rag_search import rag_search
from server.agent.plan_steps import plan_steps
from server.agent.generate import generate
from server.agent.post_process import post_process
import server.agent.tool_selector as _tool_selector_mod
import server.agent.emit_tool_calls as _emit_tool_calls_mod

# Gate 3 canned response (Section 11)
_VOLATILE_RESPONSE = (
    "Xin lỗi, tính năng này chưa được hỗ trợ trong phiên bản hiện tại (V1). "
    "Hệ thống chưa thể truy cập dữ liệu real-time như git diff, runtime logs, "
    "live metrics, hoặc error stack traces từ process đang chạy. "
    "Vui lòng mô tả vấn đề cụ thể để tôi hỗ trợ dựa trên source code."
)


def _reject_volatile(state: AgentState) -> dict:
    """Return a canned 'not supported' message for Gate 3 volatile queries."""
    return {"draft": _VOLATILE_RESPONSE}


def _route_after_classify(state: AgentState) -> str:
    """Turn 2 bypass: skip route_context when tool results are present.

    On Turn 2, route_context would re-parse ToolMessage JSON content and
    set force_reindex/mentioned_files from the tool result text — causing
    unwanted side effects. Skip directly to tool_selector.
    """
    if state.get("is_tool_result_turn"):
        return "tool_selector"
    return "route_context"


def _route_after_intent(state: AgentState) -> str:
    """Conditional edge after route_context — volatile gate fires first."""
    if state.get("volatile_rejected"):
        return "reject_volatile"
    return "tool_selector"


def _route_after_tool_selector(state: AgentState) -> str:
    """Route based on whether tools need to be emitted."""
    if state.get("pending_tool_calls"):
        return "emit_tool_calls"
    if state.get("intent") == "structural_analysis":
        return "generate"
    return "rag_search"


def _route_after_rag(state: AgentState) -> str:
    """Conditional edge after rag_search."""
    if state.get("intent") == "code_gen":
        return "plan_steps"
    return "generate"


def build_agent_graph(vllm_client, model: str, qdrant, embedder, sse_callback=None):
    """Build and compile the LangGraph agent graph.

    All async nodes that need external services receive them via partial.
    """
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("route_context", route_context)
    graph.add_node("reject_volatile", _reject_volatile)
    _qdrant = qdrant

    async def _tool_selector_node(state):
        return await _tool_selector_mod.tool_selector(state, qdrant=_qdrant)

    _sse = sse_callback

    async def _emit_tool_calls_node(state):
        return await _emit_tool_calls_mod.emit_tool_calls(state, sse_callback=_sse)

    graph.add_node("tool_selector", _tool_selector_node)
    graph.add_node("emit_tool_calls", _emit_tool_calls_node)
    graph.add_node(
        "rag_search",
        partial(rag_search, qdrant=qdrant, embedder=embedder),
    )
    graph.add_node(
        "plan_steps",
        partial(plan_steps, vllm_client=vllm_client, model=model),
    )
    graph.add_node(
        "generate",
        partial(generate, vllm_client=vllm_client, model=model, sse_callback=sse_callback),
    )
    graph.add_node("post_process", post_process)

    # Set entry point
    graph.set_entry_point("classify_intent")

    # classify_intent -> conditional: Turn 2 bypass
    graph.add_conditional_edges(
        "classify_intent",
        _route_after_classify,
        {
            "route_context": "route_context",
            "tool_selector": "tool_selector",
        },
    )

    # route_context -> conditional: volatile gate or tool_selector
    graph.add_conditional_edges(
        "route_context",
        _route_after_intent,
        {
            "reject_volatile": "reject_volatile",
            "tool_selector": "tool_selector",
        },
    )

    # reject_volatile goes straight to END
    graph.add_edge("reject_volatile", END)

    # tool_selector -> conditional: emit tools, skip to generate, or rag_search
    graph.add_conditional_edges(
        "tool_selector",
        _route_after_tool_selector,
        {
            "emit_tool_calls": "emit_tool_calls",
            "rag_search": "rag_search",
            "generate": "generate",
        },
    )

    # emit_tool_calls -> END (Turn 1 ends here, Continue takes over)
    graph.add_edge("emit_tool_calls", END)

    # rag_search -> conditional
    graph.add_conditional_edges(
        "rag_search",
        _route_after_rag,
        {
            "plan_steps": "plan_steps",
            "generate": "generate",
        },
    )

    graph.add_edge("plan_steps", "generate")
    graph.add_edge("generate", "post_process")
    graph.add_edge("post_process", END)

    return graph.compile()
