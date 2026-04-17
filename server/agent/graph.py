"""LangGraph agent graph — native tool-call edition.

Flow:
  classify_intent
    ├─ is_tool_result_turn=True → generate → post_process → END
    └─ else → route_context
         ├─ volatile_rejected → reject_volatile → END
         └─ else → <intent router>
              ├─ code_review → review_analyze → review_format
              │                 ├─ auto_post → upsert_mr_comment → END
              │                 └─ else → post_process → END
              └─ else → generate → post_process → END

rag_search / plan_steps only wired when enable_rag=True (pending RAG redesign).
"""

from __future__ import annotations

from functools import partial

from langgraph.graph import END, StateGraph

from server.agent.state import AgentState
from server.agent.classify_intent import classify_intent
from server.agent.route_context import route_context
from server.agent.generate import generate
from server.agent.post_process import post_process
from server.agent.review_analyze import review_analyze
from server.agent.review_format import review_format
from server.agent.upsert_mr_comment import upsert_mr_comment

_VOLATILE_RESPONSE = (
    "Xin lỗi, tính năng này chưa được hỗ trợ trong phiên bản hiện tại (V1). "
    "Hệ thống chưa thể truy cập dữ liệu real-time như git diff, runtime logs, "
    "live metrics, hoặc error stack traces từ process đang chạy. "
    "Vui lòng mô tả vấn đề cụ thể để tôi hỗ trợ dựa trên source code."
)


def _reject_volatile(state: AgentState) -> dict:
    return {"draft": _VOLATILE_RESPONSE}


def _route_after_classify(state: AgentState) -> str:
    if state.get("is_tool_result_turn"):
        return "generate"
    return "route_context"


def _route_after_context(state: AgentState) -> str:
    if state.get("volatile_rejected"):
        return "reject_volatile"
    if state.get("intent") == "code_review":
        return "review_analyze"
    return "generate"


def _route_after_review_format(state: AgentState) -> str:
    return "upsert_mr_comment" if state.get("auto_post") else "post_process"


def build_agent_graph(
    vllm_client,
    model: str,
    qdrant,
    embedder,
    sse_callback=None,
    *,
    enable_rag: bool = False,
):
    # Re-read current module bindings each call so monkeypatch-based tests
    # see the intended fakes (partial binds at call time, not import time).
    import server.agent.graph as _this

    graph = StateGraph(AgentState)

    graph.add_node("classify_intent", _this.classify_intent)
    graph.add_node("route_context", _this.route_context)
    graph.add_node("reject_volatile", _reject_volatile)
    graph.add_node(
        "generate",
        partial(_this.generate, vllm_client=vllm_client, model=model, sse_callback=sse_callback),
    )
    graph.add_node("post_process", _this.post_process)

    graph.add_node(
        "review_analyze",
        partial(review_analyze, vllm_client=vllm_client, model=model),
    )
    graph.add_node("review_format", review_format)
    graph.add_node("upsert_mr_comment", upsert_mr_comment)

    graph.set_entry_point("classify_intent")

    graph.add_conditional_edges(
        "classify_intent",
        _route_after_classify,
        {"route_context": "route_context", "generate": "generate"},
    )

    graph.add_conditional_edges(
        "route_context",
        _route_after_context,
        {
            "reject_volatile": "reject_volatile",
            "review_analyze": "review_analyze",
            "generate": "generate",
        },
    )

    graph.add_edge("reject_volatile", END)

    graph.add_edge("review_analyze", "review_format")
    graph.add_conditional_edges(
        "review_format",
        _route_after_review_format,
        {"upsert_mr_comment": "upsert_mr_comment", "post_process": "post_process"},
    )
    graph.add_edge("upsert_mr_comment", END)

    graph.add_edge("generate", "post_process")
    graph.add_edge("post_process", END)

    if enable_rag:
        # Nodes kept resident for future wiring; edges intentionally not added.
        from server.agent.rag_search import rag_search
        from server.agent.plan_steps import plan_steps

        graph.add_node("rag_search", partial(rag_search, qdrant=qdrant, embedder=embedder))
        graph.add_node("plan_steps", partial(plan_steps, vllm_client=vllm_client, model=model))

    return graph.compile()
