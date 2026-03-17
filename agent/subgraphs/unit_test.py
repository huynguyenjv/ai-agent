"""
UnitTest SubGraph — orchestrates test generation via LangGraph.

Flow:
  retrieve → check_strategy
    → [single_pass] → build_prompt
    → [two_phase]   → analyze → build_prompt
  → call_llm → validate
    → [pass]          → human_review
    → [fail, retries] → repair → validate (loop)
    → [fail, max]     → human_review (with issues)
  → human_review
    → [approve] → save_result → END
    → [reject]  → call_llm (regenerate)
    → [auto]    → save_result → END
"""

from __future__ import annotations

from functools import partial

from langgraph.graph import StateGraph, END

from agent.state import UnitTestState


# ═══════════════════════════════════════════════════════════════════════
# Routing functions (conditional edges)
# ═══════════════════════════════════════════════════════════════════════

def route_after_retrieve(state: dict) -> str:
    """Check for errors after retrieve, short-circuit if needed."""
    if state.get("error"):
        return "error"
    return "continue"


def route_strategy(state: dict) -> str:
    """Route to single_pass or two_phase path after check_strategy."""
    strategy = state.get("strategy", "single_pass")
    if strategy == "two_phase":
        return "two_phase"
    return "single_pass"


def route_after_validate(state: dict) -> str:
    """Route after validation: pass → review, fail → repair or accept."""
    if state.get("validation_passed"):
        return "pass"

    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)

    if retry_count < max_retries:
        return "retry"

    # Max retries exhausted — show human review with issues
    return "max_retries"


def route_after_execution(state: dict) -> str:
    """Route after tool execution: pass → review, fail → repair (if retries left)."""
    if state.get("execution_passed"):
        return "pass"

    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)

    if retry_count < max_retries:
        return "retry"

    return "max_retries"


def route_after_review(state: dict) -> str:
    """Route after human review: approve → save, reject → regenerate."""
    approved = state.get("human_approved")

    if approved is None:
        # Auto mode (no review requested)
        return "approve"
    if approved:
        return "approve"

    return "reject"


# ═══════════════════════════════════════════════════════════════════════
# Graph builder
# ═══════════════════════════════════════════════════════════════════════

def build_unit_test_graph(
    *,
    rag_client,
    vllm_client,
    prompt_builder,
    validation_pipeline,
    repair_selector,
    context_builder=None,
    two_phase_strategy=None,
    domain_registry=None,
    memory_manager=None,
    cache_service=None,
):
    from agent.nodes.retrieve import retrieve_node
    from agent.nodes.check_strategy import check_strategy_node
    from agent.nodes.analyze import analyze_node
    from agent.nodes.build_prompt import build_prompt_node
    from agent.nodes.call_llm import call_llm_node
    from agent.nodes.validate import validate_node
    from agent.nodes.tool_node import tool_node as tool_node_func
    from agent.nodes.repair import repair_node
    from agent.nodes.human_review import human_review_node
    from agent.nodes.save_result import save_result_node

    # ── Bind dependencies ───────────────────────────────────────────
    retrieve = partial(
        retrieve_node,
        rag_client=rag_client,
        context_builder=context_builder,
        cache_service=cache_service,
    )
    check_strategy = check_strategy_node
    analyze = partial(
        analyze_node,
        two_phase_strategy=two_phase_strategy,
        domain_registry=domain_registry,
    )
    build_prompt = partial(
        build_prompt_node,
        prompt_builder=prompt_builder,
        two_phase_strategy=two_phase_strategy,
    )
    call_llm = partial(call_llm_node, vllm_client=vllm_client)
    validate = partial(validate_node, validation_pipeline=validation_pipeline)
    tool_executor = tool_node_func
    repair = partial(
        repair_node,
        repair_selector=repair_selector,
        vllm_client=vllm_client,
        prompt_builder=prompt_builder,
        validation_pipeline=validation_pipeline,
    )
    human_review = human_review_node
    save_result = partial(save_result_node, memory_manager=memory_manager)

    # ── Build StateGraph ────────────────────────────────────────────
    graph = StateGraph(UnitTestState)

    graph.add_node("retrieve", retrieve)
    graph.add_node("check_strategy", check_strategy)
    graph.add_node("analyze", analyze)
    graph.add_node("build_prompt", build_prompt)
    graph.add_node("call_llm", call_llm)
    graph.add_node("validate", validate)
    graph.add_node("tool_executor", tool_executor)
    graph.add_node("repair", repair)
    graph.add_node("human_review", human_review)
    graph.add_node("save_result", save_result)

    # ── Edges ───────────────────────────────────────────────────────
    graph.set_entry_point("retrieve")

    graph.add_conditional_edges(
        "retrieve",
        route_after_retrieve,
        {"error": "save_result", "continue": "check_strategy"},
    )

    graph.add_conditional_edges(
        "check_strategy",
        route_strategy,
        {"single_pass": "build_prompt", "two_phase": "analyze"},
    )

    graph.add_edge("analyze", "build_prompt")
    graph.add_edge("build_prompt", "call_llm")
    graph.add_edge("call_llm", "validate")

    graph.add_conditional_edges(
        "validate",
        route_after_validate,
        {
            "pass": "tool_executor",      # Successful rules validation leads to execution
            "retry": "repair",
            "max_retries": "human_review",
        },
    )

    graph.add_conditional_edges(
        "tool_executor",
        route_after_execution,
        {
            "pass": "human_review",
            "retry": "repair",
            "max_retries": "human_review"
        },
    )

    graph.add_edge("repair", "validate")

    graph.add_conditional_edges(
        "human_review",
        route_after_review,
        {"approve": "save_result", "reject": "call_llm"},
    )

    graph.add_edge("save_result", END)

    return graph.compile()
