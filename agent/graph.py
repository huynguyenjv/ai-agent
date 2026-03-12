"""
Main LangGraph — Supervisor + SubGraph routing + Checkpointer.

This is the top-level entry point for the refactored agent.
Replaces AgentOrchestrator as the primary orchestration layer.

Usage::

    from agent.graph import create_agent_graph

    graph = create_agent_graph(
        rag_client=rag_client,
        vllm_client=vllm_client,
    )

    # Invoke
    result = graph.invoke(
        {"user_input": "Generate tests for UserService", ...},
        config={"configurable": {"thread_id": session_id}},
    )

    # Resume after human review
    graph.update_state(
        config,
        {"human_approved": True, "human_feedback": ""},
    )
    result = graph.invoke(None, config=config)
"""

from __future__ import annotations

import os
from typing import Optional

from langgraph.graph import StateGraph, END

import structlog

from agent.state import AgentState
from agent.supervisor import supervisor_node

logger = structlog.get_logger()


# ═══════════════════════════════════════════════════════════════════════
# Routing
# ═══════════════════════════════════════════════════════════════════════

def route_to_subgraph(state: dict) -> str:
    """Route from supervisor to the appropriate subgraph."""
    intent = state.get("intent", "unit_test")

    # Currently only unit_test is implemented
    if intent in ("unit_test",):
        return "unit_test"

    # Future subgraphs
    # if intent == "code_review":
    #     return "code_review"

    # Default fallback
    return "unit_test"


# ═══════════════════════════════════════════════════════════════════════
# Graph factory
# ═══════════════════════════════════════════════════════════════════════

def create_agent_graph(
    *,
    rag_client,
    vllm_client,
    prompt_builder=None,
    validation_pipeline=None,
    repair_selector=None,
    context_builder=None,
    two_phase_strategy=None,
    domain_registry=None,
    memory_manager=None,
    cache_service=None,
    checkpoint_db: str = "checkpoints.db",
):
    """Create the main agent graph with all dependencies wired in.

    Args:
        rag_client: RAGClient instance.
        vllm_client: VLLMClient instance.
        prompt_builder: PromptBuilder (auto-created if None).
        validation_pipeline: ValidationPipeline (auto-created if None).
        repair_selector: RepairStrategySelector (auto-created if None).
        context_builder: Optional ContextBuilder.
        two_phase_strategy: Optional TwoPhaseStrategy.
        domain_registry: Optional DomainTypeRegistry.
        memory_manager: Optional MemoryManager.
        cache_service: Optional CacheService.
        checkpoint_db: SQLite database path for checkpointer.

    Returns:
        Compiled StateGraph with checkpointer.
    """
    from agent.prompt import PromptBuilder
    from agent.validation import ValidationPipeline
    from agent.repair import RepairStrategySelector
    from agent.subgraphs.unit_test import build_unit_test_graph

    # Auto-create defaults if not provided
    if prompt_builder is None:
        prompt_builder = PromptBuilder()
    if validation_pipeline is None:
        validation_pipeline = ValidationPipeline()
    if repair_selector is None:
        repair_selector = RepairStrategySelector()

    # Build UnitTest subgraph (compiled)
    unit_test_compiled = build_unit_test_graph(
        rag_client=rag_client,
        vllm_client=vllm_client,
        prompt_builder=prompt_builder,
        validation_pipeline=validation_pipeline,
        repair_selector=repair_selector,
        context_builder=context_builder,
        two_phase_strategy=two_phase_strategy,
        domain_registry=domain_registry,
        memory_manager=memory_manager,
        cache_service=cache_service,
    )

    # ── Build Supervisor graph ──────────────────────────────────────
    supervisor_graph = StateGraph(AgentState)

    # Supervisor node classifies intent
    supervisor_graph.add_node("supervisor", supervisor_node)

    # SubGraph nodes (each is a compiled subgraph)
    supervisor_graph.add_node("unit_test", unit_test_compiled)
    # Future: supervisor_graph.add_node("code_review", code_review_compiled)

    # Entry point
    supervisor_graph.set_entry_point("supervisor")

    # Route from supervisor to subgraph
    supervisor_graph.add_conditional_edges(
        "supervisor",
        route_to_subgraph,
        {
            "unit_test": "unit_test",
            # Future: "code_review": "code_review",
        },
    )

    # SubGraph → END
    supervisor_graph.add_edge("unit_test", END)

    # ── Compile with checkpointer ───────────────────────────────────
    checkpointer = _create_checkpointer(checkpoint_db)

    compiled = supervisor_graph.compile(checkpointer=checkpointer)

    logger.info(
        "Agent graph created",
        checkpointer=type(checkpointer).__name__,
        checkpoint_db=checkpoint_db,
    )

    return compiled


def _create_checkpointer(db_path: str):
    """Create SQLite checkpointer for persistent state.

    Falls back to in-memory if SQLite fails.
    """
    try:
        import sqlite3
        from langgraph.checkpoint.sqlite import SqliteSaver
        
        # Create a persistent connection and pass it to SqliteSaver
        conn = sqlite3.connect(db_path, check_same_thread=False)
        checkpointer = SqliteSaver(conn)
        logger.info("Checkpointer: SQLite", db=db_path)
        return checkpointer
    except Exception as e:
        logger.warning("SQLite checkpointer failed, using MemorySaver", error=str(e))
        from langgraph.checkpoint.memory import MemorySaver
        return MemorySaver()
