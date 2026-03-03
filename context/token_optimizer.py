"""
Token Optimizer — budget-aware context truncation.

Ensures the assembled context fits within a specified token budget.
Uses a priority-based approach: high-priority snippets are preserved
while lower-priority ones are truncated or dropped.

Token estimation uses a fast char-based heuristic (~4 chars/token for code).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from .snippet_selector import Snippet

logger = structlog.get_logger()

# ── Constants ────────────────────────────────────────────────────────

CHARS_PER_TOKEN = 4  # Conservative estimate for code
DEFAULT_TOKEN_BUDGET = 6000  # ~24K chars


# ── Token Optimizer ──────────────────────────────────────────────────

@dataclass
class BudgetReport:
    """Summary of token optimization."""

    original_tokens: int
    final_tokens: int
    snippets_kept: int
    snippets_dropped: int
    snippets_truncated: int
    over_budget: bool


class TokenOptimizer:
    """Enforces a token budget on a list of snippets.

    Strategy (in order):
      1. Keep all snippets that fit within budget (by priority).
      2. If over budget, truncate lowest-priority snippets first.
      3. If still over, drop lowest-priority snippets entirely.
    """

    def __init__(self, token_budget: int = DEFAULT_TOKEN_BUDGET) -> None:
        self.token_budget = token_budget

    # ── Public API ───────────────────────────────────────────────────

    def optimize(self, snippets: list[Snippet]) -> tuple[list[Snippet], BudgetReport]:
        """Trim snippets to fit within the token budget.

        Snippets should already be sorted by priority (highest first).
        Returns (optimized_snippets, report).
        """
        if not snippets:
            return [], BudgetReport(0, 0, 0, 0, 0, False)

        original_tokens = sum(self.estimate_tokens(s.content) for s in snippets)

        if original_tokens <= self.token_budget:
            return list(snippets), BudgetReport(
                original_tokens=original_tokens,
                final_tokens=original_tokens,
                snippets_kept=len(snippets),
                snippets_dropped=0,
                snippets_truncated=0,
                over_budget=False,
            )

        # Phase 1: Greedily add snippets in priority order
        kept: list[Snippet] = []
        remaining_budget = self.token_budget
        truncated_count = 0
        dropped_count = 0

        for snippet in snippets:
            token_cost = self.estimate_tokens(snippet.content)

            if token_cost <= remaining_budget:
                kept.append(snippet)
                remaining_budget -= token_cost
            elif remaining_budget >= 100:
                # Truncate this snippet to fit the remaining budget
                truncated = self._truncate_snippet(snippet, remaining_budget)
                if truncated:
                    kept.append(truncated)
                    remaining_budget -= self.estimate_tokens(truncated.content)
                    truncated_count += 1
                else:
                    dropped_count += 1
            else:
                dropped_count += 1

        final_tokens = sum(self.estimate_tokens(s.content) for s in kept)

        report = BudgetReport(
            original_tokens=original_tokens,
            final_tokens=final_tokens,
            snippets_kept=len(kept),
            snippets_dropped=dropped_count,
            snippets_truncated=truncated_count,
            over_budget=final_tokens > self.token_budget,
        )

        logger.info(
            "Token optimization",
            original=original_tokens,
            final=final_tokens,
            budget=self.token_budget,
            kept=len(kept),
            dropped=dropped_count,
            truncated=truncated_count,
        )

        return kept, report

    # ── Utilities ────────────────────────────────────────────────────

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Fast token estimation based on character count."""
        return max(1, len(text) // CHARS_PER_TOKEN)

    def _truncate_snippet(self, snippet: Snippet, max_tokens: int) -> Snippet | None:
        """Truncate a snippet to fit within max_tokens.

        For code snippets, truncates at method boundaries if possible.
        For summaries, truncates at sentence boundaries.
        Returns a new Snippet with truncated content, or None if too small.
        """
        max_chars = max_tokens * CHARS_PER_TOKEN
        if max_chars < 50:
            return None

        content = snippet.content
        if len(content) <= max_chars:
            return snippet

        # Try to truncate at a meaningful boundary
        truncated = content[:max_chars]

        # For code: find the last complete method (look for closing brace at line start)
        if snippet.role.value in ("source", "dependency", "related"):
            last_brace = truncated.rfind("\n}")
            if last_brace > max_chars * 0.5:  # At least 50% of content
                truncated = truncated[: last_brace + 2] + "\n// ... (truncated)"
            else:
                # Fall back to last complete line
                last_newline = truncated.rfind("\n")
                if last_newline > 0:
                    truncated = truncated[:last_newline] + "\n// ... (truncated)"

        # For summaries: truncate at sentence boundary
        elif snippet.role.value == "summary":
            last_period = truncated.rfind(". ")
            if last_period > max_chars * 0.5:
                truncated = truncated[: last_period + 1] + " ..."
            else:
                last_newline = truncated.rfind("\n")
                if last_newline > 0:
                    truncated = truncated[:last_newline] + " ..."

        # Create new snippet with truncated content
        from .snippet_selector import Snippet
        return Snippet(
            content=truncated,
            role=snippet.role,
            priority=snippet.priority,
            class_name=snippet.class_name,
            file_path=snippet.file_path,
            metadata={**snippet.metadata, "truncated": True},
        )
