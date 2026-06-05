"""AgentState — Section 4.3.

Mutable state object passed between LangGraph nodes.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    """Section 4.3 — The mutable state object passed between LangGraph nodes."""

    messages: Annotated[list, add_messages]
    intent: str  # code_gen, unit_test, explain, structural_analysis, search, refine, code_review
    experiment_variant: str              # A/B prompt variant (Phase 16.2), default "default"
    active_file: str | None
    repo_path: str | None                # repository root (client-provided), used by context_builder
    mentioned_files: list[str]
    freshness_signal: bool
    force_reindex: bool
    rag_chunks: list[dict]
    rag_hit: bool
    rag_enabled: bool                    # Whether RAG is enabled for this request
    hash_verified: bool
    tool_results: list[dict]
    context_assembled: str
    draft: str
    degraded: bool                       # True when served via graceful-degradation fallback
    emitted_steps: list[str]
    volatile_rejected: bool  # Gate 3: query requests volatile data not supported in V1
    pending_tool_calls: list[dict]   # tools to emit, set by tool_selector
    is_tool_result_turn: bool        # True when request contains role:"tool" messages
    tool_turns_used: int             # capped at 1, prevents > 2 round-trips

    # --- Code review (Section: code_review spec) ---
    review_mode: str                 # "" | "pr" | "file"
    pr_context: dict | None          # {provider, repo, pr_id, commit_sha, base_sha, diff, files, previous_reviews}
    review_findings: list[dict]      # [{file, line, severity, category, title, description, suggestion}]
    output_format: str               # "" | "markdown" | "sse_stream"

    # --- Native tool-call (client-forwarded) ---
    client_tools: list[dict]             # raw tool schemas from ChatRequest.tools
    tool_choice: str | dict | None       # forwarded tool_choice

    # --- Validation (Phase 3) ---
    validation_warnings: list[str]       # warnings from post_process validation

    # --- Agentic Loop (Phase 2) ---
    verification_passed: bool            # True if verify_result passed
    retry_reason: str                    # Reason for retry if verification failed
    retry_count: int                     # Number of retries attempted

    # --- Planner (Phase 5) ---
    complexity: str                      # "simple" | "complex"
    task_plan: dict | None               # Full plan from planner
    planner_steps: list[dict]            # Steps to execute
    current_step: int                    # Current step index
    planner_reasoning: str               # Why this complexity

    # --- Critic (Phase 5) ---
    critic_passed: bool                  # True if critic approved
    critic_score: int                    # 0-10 score
    critic_issues: list[dict]            # Issues found
    critic_feedback: str                 # Feedback for retry
    critic_retries: int                  # Number of critic retries
