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
    active_file: str | None
    mentioned_files: list[str]
    freshness_signal: bool
    force_reindex: bool
    rag_chunks: list[dict]
    rag_hit: bool
    hash_verified: bool
    tool_results: list[dict]
    context_assembled: str
    draft: str
    emitted_steps: list[str]
    volatile_rejected: bool  # Gate 3: query requests volatile data not supported in V1
    pending_tool_calls: list[dict]   # tools to emit, set by tool_selector
    is_tool_result_turn: bool        # True when request contains role:"tool" messages
    tool_turns_used: int             # capped at 1, prevents > 2 round-trips

    # --- Code review (Section: code_review spec) ---
    review_mode: str                 # "" | "pr" | "file"
    pr_context: dict | None          # {provider, repo, pr_id, commit_sha, base_sha, diff, files, previous_reviews, note_id}
    review_findings: list[dict]      # [{file, line, severity, category, title, description, suggestion}]
    output_format: str               # "" | "markdown" | "sse_stream"
    auto_post: bool                  # True khi gọi từ /review/pr

    # --- Native tool-call (client-forwarded) ---
    client_tools: list[dict]             # raw tool schemas from ChatRequest.tools
    tool_choice: str | dict | None       # forwarded tool_choice
