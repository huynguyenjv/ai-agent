"""Smart context selection for optimal token usage.

Builds context within token budget, prioritizing relevance.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from server.agent.state import AgentState

logger = logging.getLogger("server.agent.context_builder")

# Default token budget
DEFAULT_TOKEN_BUDGET = 8000

# Approximate tokens per character (conservative estimate)
CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Estimate token count from text.

    Uses character-based estimation. For more accurate counting,
    use tiktoken with the specific model's tokenizer.

    Args:
        text: Text to estimate

    Returns:
        Estimated token count
    """
    if not text:
        return 0
    return len(text) // CHARS_PER_TOKEN


def read_file_content(file_path: str, repo_path: str, max_lines: int = 500) -> str | None:
    """Read file content with size limit.

    Args:
        file_path: Relative file path
        repo_path: Repository root
        max_lines: Maximum lines to read

    Returns:
        File content or None
    """
    full_path = os.path.join(repo_path, file_path)

    if not os.path.isfile(full_path):
        return None

    try:
        with open(full_path, "r", encoding="utf-8", errors="replace") as f:
            lines = []
            for i, line in enumerate(f):
                if i >= max_lines:
                    lines.append(f"\n... [truncated at {max_lines} lines]")
                    break
                lines.append(line)
            return "".join(lines)
    except OSError:
        return None


def build_optimal_context(
    state: AgentState,
    repo_path: str,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
) -> dict[str, Any]:
    """Build context within token budget, prioritizing relevance.

    Priority order:
    1. Active file (always include if fits)
    2. RAG chunks (sorted by relevance score)
    3. Mentioned files
    4. Tool results

    Args:
        state: Agent state
        repo_path: Repository root path
        token_budget: Maximum tokens for context

    Returns:
        {context: str, parts: list, tokens_used: int, parts_included: int}
    """
    context_parts: list[tuple[str, str, float, int]] = []  # (name, content, priority, tokens)

    # Priority 1: Active file (priority 10)
    active_file = state.get("active_file")
    if active_file:
        content = read_file_content(active_file, repo_path)
        if content:
            tokens = estimate_tokens(content)
            context_parts.append((
                f"Active File: {active_file}",
                content,
                10.0,
                tokens,
            ))

    # Priority 2: RAG chunks (use rrf_score or score as priority)
    for chunk in state.get("rag_chunks", []):
        body = chunk.get("body", chunk.get("content", ""))
        if body:
            file_path = chunk.get("file_path", "unknown")
            start = chunk.get("start_line", "?")
            end = chunk.get("end_line", "?")
            score = chunk.get("rrf_score", chunk.get("score", 0.5))

            tokens = estimate_tokens(body)
            context_parts.append((
                f"RAG: {file_path} (lines {start}-{end})",
                body,
                5.0 + score,  # Base 5 + score
                tokens,
            ))

    # Priority 3: Mentioned files (priority 3)
    for file_path in state.get("mentioned_files", []):
        if file_path == active_file:
            continue  # Already included

        content = read_file_content(file_path, repo_path, max_lines=200)
        if content:
            tokens = estimate_tokens(content)
            context_parts.append((
                f"Mentioned: {file_path}",
                content,
                3.0,
                tokens,
            ))

    # Priority 4: Previous context (priority 2)
    prev_context = state.get("context_assembled", "")
    if prev_context and len(prev_context) > 100:
        tokens = estimate_tokens(prev_context)
        context_parts.append((
            "Previous Context",
            prev_context[:4000],  # Limit previous context
            2.0,
            min(tokens, 1000),
        ))

    # Sort by priority (descending)
    context_parts.sort(key=lambda x: x[2], reverse=True)

    # Build context within budget
    final_parts = []
    used_tokens = 0
    included_count = 0

    for name, content, priority, tokens in context_parts:
        if used_tokens + tokens <= token_budget:
            final_parts.append(f"### {name}\n```\n{content}\n```")
            used_tokens += tokens
            included_count += 1
        else:
            # Try to fit partial content
            remaining = token_budget - used_tokens
            if remaining > 200:
                partial_chars = remaining * CHARS_PER_TOKEN
                partial_content = content[:partial_chars]
                final_parts.append(f"### {name} (truncated)\n```\n{partial_content}\n```")
                used_tokens += remaining
                included_count += 1
            break

    context_str = "\n\n".join(final_parts)

    logger.info("context_builder: %d parts, %d tokens used (budget: %d)",
                included_count, used_tokens, token_budget)

    return {
        "context": context_str,
        "parts": [p[0] for p in context_parts[:included_count]],
        "tokens_used": used_tokens,
        "token_budget": token_budget,
        "parts_included": included_count,
        "parts_total": len(context_parts),
    }


def get_context_summary(state: AgentState) -> str:
    """Get a brief summary of available context.

    Args:
        state: Agent state

    Returns:
        Summary string
    """
    parts = []

    if state.get("active_file"):
        parts.append(f"Active: {state['active_file']}")

    rag_count = len(state.get("rag_chunks", []))
    if rag_count:
        parts.append(f"RAG: {rag_count} chunks")

    mentioned = state.get("mentioned_files", [])
    if mentioned:
        parts.append(f"Mentioned: {len(mentioned)} files")

    return " | ".join(parts) if parts else "No context"
