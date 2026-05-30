"""LangGraph agent graph — native tool-call edition.

Flow:
  classify_intent
    ├─ is_tool_result_turn=True → generate → verify → post_process → END
    └─ else → route_context
         ├─ volatile_rejected → reject_volatile → END
         └─ else → <intent router>
              ├─ code_review → review_analyze → review_format → post_process → END
              └─ else → planner
                   ├─ simple → generate → verify → post_process → END
                   └─ complex → generate → verify → critic
                        ├─ passed → post_process → END
                        └─ failed → generate (retry with feedback)

Note: `code_review` here is only used by chat completions (Continue). The
GitLab MR runner calls /review/analyze which bypasses the graph entirely.
"""

from __future__ import annotations

import os
from functools import partial

from langgraph.graph import END, StateGraph

from server.agent.state import AgentState
from server.agent.classify_intent import classify_intent
from server.agent.route_context import route_context
from server.agent.generate import generate
from server.agent.post_process import post_process
from server.agent.review_analyze import review_analyze
from server.agent.review_format import review_format
from server.agent.verify_result import verify_result
from server.agent.planner import plan_task
from server.agent.critic import critique_output

_DEFAULT_VOLATILE_RESPONSE = (
    "Xin lỗi, tính năng này chưa được hỗ trợ trong phiên bản hiện tại (V1). "
    "Hệ thống chưa thể truy cập dữ liệu real-time như git diff, runtime logs, "
    "live metrics, hoặc error stack traces từ process đang chạy. "
    "Vui lòng mô tả vấn đề cụ thể để tôi hỗ trợ dựa trên source code."
)

_VOLATILE_RESPONSE = os.environ.get("VOLATILE_REJECTION_MESSAGE", _DEFAULT_VOLATILE_RESPONSE)


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
    return "planner"


def _route_after_planner(state: AgentState) -> str:
    """Route after planning: to RAG search if enabled, else generate."""
    # Check if RAG is enabled and beneficial
    if state.get("rag_enabled") and _should_use_rag(state):
        return "rag_search"
    return "generate"


def _should_use_rag(state: AgentState) -> bool:
    """Determine if RAG search would be beneficial for this query."""
    intent = state.get("intent", "")

    # Intents that benefit from RAG
    rag_intents = {"code_gen", "unit_test", "refactor", "explain", "search"}
    if intent not in rag_intents:
        return False

    # Skip RAG for very simple queries (already have file target)
    if state.get("file_target") and state.get("complexity") == "simple":
        return False

    return True


def _route_after_verify(state: AgentState) -> str:
    """Route after verification: retry, critic, or post_process."""
    if not state.get("verification_passed", True):
        # Verification failed - retry generate
        return "generate"

    # For complex tasks, go to critic for quality review
    if state.get("complexity") == "complex":
        return "critic"

    return "post_process"


def _route_after_critic(state: AgentState) -> str:
    """Route after critic: retry if quality issues, else post_process."""
    if state.get("critic_passed", True):
        return "post_process"

    # Critic found issues - check retry count
    critic_retries = state.get("critic_retries", 0)
    if critic_retries >= 2:
        # Max retries reached, proceed anyway
        return "post_process"

    # Retry with critic feedback
    return "generate"


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

    graph.add_node(
        "classify_intent",
        partial(_this.classify_intent, vllm_client=vllm_client, model=model),
    )
    graph.add_node("route_context", _this.route_context)
    graph.add_node("reject_volatile", _reject_volatile)
    graph.add_node(
        "planner",
        partial(_this.plan_task, vllm_client=vllm_client, model=model),
    )
    graph.add_node(
        "generate",
        partial(_this.generate, vllm_client=vllm_client, model=model, sse_callback=sse_callback),
    )
    graph.add_node("verify_result", _this.verify_result)
    graph.add_node(
        "critic",
        partial(_this.critique_output, vllm_client=vllm_client, model=model),
    )
    graph.add_node("post_process", _this.post_process)

    graph.add_node(
        "review_analyze",
        partial(review_analyze, vllm_client=vllm_client, model=model),
    )
    graph.add_node("review_format", review_format)

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
            "planner": "planner",
        },
    )

    graph.add_edge("reject_volatile", END)

    graph.add_edge("review_analyze", "review_format")
    graph.add_edge("review_format", "post_process")

    # Planner → RAG (if enabled) or Generate
    if enable_rag:
        graph.add_conditional_edges(
            "planner",
            _route_after_planner,
            {"rag_search": "rag_search", "generate": "generate"},
        )
    else:
        graph.add_conditional_edges(
            "planner",
            _route_after_planner,
            {"generate": "generate"},
        )

    # Agentic loop: generate → verify → (retry, critic, or post_process)
    graph.add_edge("generate", "verify_result")
    graph.add_conditional_edges(
        "verify_result",
        _route_after_verify,
        {"generate": "generate", "critic": "critic", "post_process": "post_process"},
    )

    # Critic → (retry or post_process)
    graph.add_conditional_edges(
        "critic",
        _route_after_critic,
        {"generate": "generate", "post_process": "post_process"},
    )

    graph.add_edge("post_process", END)

    if enable_rag:
        from server.agent.rag_search import rag_search

        graph.add_node("rag_search", partial(rag_search, qdrant=qdrant, embedder=embedder))

        # RAG → Generate (rag_search enriches context then proceeds to generate)
        graph.add_edge("rag_search", "generate")

    return graph.compile()
