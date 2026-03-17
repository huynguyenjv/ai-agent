"""
Token Optimizer — budget-aware context truncation.

Ensures the assembled context fits within a specified token budget.
Uses a priority-based approach: high-priority snippets are preserved
while lower-priority ones are truncated or dropped.

Token counting uses tiktoken (cl100k_base) for accuracy, with a
character-based heuristic fallback when tiktoken is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from .snippet_selector import Snippet

logger = structlog.get_logger()

# ── Constants ────────────────────────────────────────────────────────

DEFAULT_TOKEN_BUDGET = 6000


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

        # Pre-compute token costs once (avoid double-counting)
        token_costs = [self.estimate_tokens(s.content) for s in snippets]
        original_tokens = sum(token_costs)

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

        for snippet, token_cost in zip(snippets, token_costs):

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
        """Token estimation using tiktoken (with heuristic fallback)."""
        from utils.tokenizer import count_tokens
        return count_tokens(text)

    def _truncate_snippet(self, snippet: Snippet, max_tokens: int) -> Snippet | None:
        """Truncate a snippet to fit within max_tokens.

        For code snippets, truncates at method boundaries if possible.
        For summaries, truncates at sentence boundaries.
        Returns a new Snippet with truncated content, or None if too small.

        Uses iterative binary-search to find the best char boundary whose
        token count is ≤ *max_tokens*, then snaps to a meaningful boundary
        (closing brace or newline for code, period or newline for summaries).
        """
        if max_tokens < 10:
            return None

        content = snippet.content
        # Fast path — already fits
        if self.estimate_tokens(content) <= max_tokens:
            return snippet

        # ── Binary search for the best char length ≤ max_tokens ──────
        lo, hi = 0, len(content)
        best = 0
        while lo <= hi:
            mid = (lo + hi) // 2
            if self.estimate_tokens(content[:mid]) <= max_tokens:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1

        if best < 50:
            return None

        truncated = content[:best]

        # ── Snap to a meaningful boundary ────────────────────────────
        if snippet.role.value in ("source", "dependency", "related"):
            # Code: prefer closing brace at start of line
            last_brace = truncated.rfind("\n}")
            if last_brace > best * 0.5:
                truncated = truncated[: last_brace + 2] + "\n// ... (truncated)"
            else:
                last_newline = truncated.rfind("\n")
                if last_newline > 0:
                    truncated = truncated[:last_newline] + "\n// ... (truncated)"

        elif snippet.role.value == "summary":
            last_period = truncated.rfind(". ")
            if last_period > best * 0.5:
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
