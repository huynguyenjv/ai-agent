"""TokenBudget — Section 5, index_with_deps Token budget enforcement.

Depth 0 (focal file): 2000 tokens
Depth 1 (direct deps): 3000 tokens
Depth 2 (transitive deps): 3000 tokens
Total maximum: 8000 tokens
"""

from __future__ import annotations

from mcp_server.models import ExtractionMode

# Approximate tokens per character ratio (conservative)
CHARS_PER_TOKEN = 4

DEPTH_BUDGETS = {
    0: 2000,
    1: 3000,
    2: 3000,
}

DEPTH_MODES = {
    0: ExtractionMode.full_body,
    1: ExtractionMode.signatures,
    2: ExtractionMode.names_only,
}


class TokenBudget:
    """Enforces depth-based mode selection and total token limits.

    Section 5: If adding chunks from a given depth would exceed the budget,
    those chunks are truncated or omitted.
    """

    def __init__(self, total_budget: int = 8000) -> None:
        self._total_budget = total_budget
        self._used: dict[int, int] = {}

    def get_mode(self, depth: int) -> ExtractionMode:
        """Return extraction mode for a given BFS depth."""
        if depth in DEPTH_MODES:
            return DEPTH_MODES[depth]
        return ExtractionMode.names_only

    def get_budget(self, depth: int) -> int:
        """Return token budget for a given depth."""
        return DEPTH_BUDGETS.get(depth, 0)

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count from text length."""
        return max(1, len(text) // CHARS_PER_TOKEN)

    def can_add(self, depth: int, text: str) -> bool:
        """Check if adding text at this depth stays within budget."""
        used = self._used.get(depth, 0)
        budget = self.get_budget(depth)
        estimated = self.estimate_tokens(text)
        return used + estimated <= budget

    def add(self, depth: int, text: str) -> None:
        """Record token usage at a depth."""
        self._used[depth] = self._used.get(depth, 0) + self.estimate_tokens(text)

    def total_used(self) -> int:
        return sum(self._used.values())
