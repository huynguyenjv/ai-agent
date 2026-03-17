"""
LangGraph state schemas for the AI Agent.

Defines typed state dicts used by:
  - Supervisor graph (AgentState)
  - UnitTest subgraph (UnitTestState)
  - Future subgraphs (CodeReviewState, RefactorState, etc.)
"""

from __future__ import annotations

from typing import TypedDict, Literal, Annotated, Optional
from operator import add


# ═══════════════════════════════════════════════════════════════════════
# Supervisor State (top-level graph)
# ═══════════════════════════════════════════════════════════════════════

class AgentState(TypedDict, total=False):
    """Top-level state shared by the Supervisor graph.

    The supervisor classifies intent and routes to the appropriate subgraph.
    Subgraphs write their result into ``subgraph_result``.
    """

    # ── Input ──
    user_input: str
    session_id: str

    # ── Pass-through to subgraphs ──
    file_path: str
    class_name: str
    task_description: str
    collection_name: str
    source_code: str
    existing_test_code: str
    changed_methods: list[str]
    require_human_review: bool
    force_two_phase: bool
    force_single_pass: bool
    complexity_threshold: int
    retry_count: int
    max_retries: int
    repo_path: str

    # ── Routing ──
    intent: Literal[
        "unit_test", "code_review", "refactor", "doc_gen", "unknown"
    ]

    # ── Discovery (Option A) ──
    discovery_context: Optional[dict] = None  # Context gathered by supervisor discovery tools

    # ── Output ──
    subgraph_result: str  # JSON-serialized result from whichever subgraph ran


# ═══════════════════════════════════════════════════════════════════════
# UnitTest SubGraph State
# ═══════════════════════════════════════════════════════════════════════

class UnitTestState(TypedDict, total=False):
    """State for the UnitTest subgraph.

    Rich enough to preserve all existing orchestrator logic:
      - Single-pass and Two-Phase strategy routing
      - 7-pass validation pipeline (ValidationPipeline)
      - 3-level escalating repair (RepairStrategySelector + FailureMemory)
      - Human-in-the-loop via interrupt()
      - Domain Registry context
      - Context assembly (SnippetSelector + TokenOptimizer)
    """

    # ── Input (set by API / supervisor) ──
    user_input: str
    session_id: str
    file_path: str
    class_name: str
    task_description: str
    collection_name: str            # Qdrant collection to search
    source_code: str                # inline Java source (if provided by CI)
    existing_test_code: str         # for incremental mode
    changed_methods: list[str]      # for incremental mode
    require_human_review: bool      # whether to interrupt for human approval
    repo_path: str                 # resolved repository root path

    # ── Strategy Routing ──
    strategy: Literal["single_pass", "two_phase"]
    force_two_phase: bool
    force_single_pass: bool
    complexity_threshold: int
    complexity_score: int           # computed by check_strategy node

    # ── RAG / Context ──
    rag_chunks: list[dict]          # serialized CodeChunks (Pydantic → dict)
    context_result: dict            # ContextBuilder output (if available)
    tokens_used: int

    # ── Analysis (Two-Phase only) ──
    analysis_result: dict           # Phase 1 structured analysis JSON
    registry_context: str           # Domain type registry context string

    # ── Prompt ──
    system_prompt: str
    user_prompt: str

    # ── LLM Output ──
    llm_output: str                 # raw LLM response
    test_code: str                  # extracted Java code block

    # ── Validation ──
    validation_passed: bool
    validation_issues: list[str]    # error messages from ValidationPipeline
    validation_result: dict         # full ValidationResult.get_summary()

    # ── Repair (3-level escalating) ──
    repair_level: Literal["targeted", "reasoning", "regenerate"]
    retry_count: int                # current attempt number
    max_retries: int                # from config (default 3)
    failure_memory: list[dict]      # serialized FailureMemory records

    # ── Human Review ──
    human_approved: bool            # True = approve, False = reject
    human_feedback: str             # rejection reason / additional instructions

    # ── Tool Execution (Option C) ──
    tool_request: dict             # {"name": "...", "parameters": {...}}
    tool_result: dict
    execution_passed: bool
    execution_output: str

    # ── Output ──
    final_test_code: str            # accepted test code
    subgraph_result: str            # JSON-serialized GenerationResult for supervisor
    error: str                      # error message (if failed)


# ═══════════════════════════════════════════════════════════════════════
# Future SubGraph States (placeholders)
# ═══════════════════════════════════════════════════════════════════════

# class CodeReviewState(TypedDict, total=False):
#     """State for CodeReview subgraph (future)."""
#     user_input: str
#     session_id: str
#     file_path: str
#     review_result: str
#     subgraph_result: str

# class RefactorState(TypedDict, total=False):
#     """State for Refactor subgraph (future)."""
#     user_input: str
#     session_id: str
#     file_path: str
#     refactored_code: str
#     subgraph_result: str
