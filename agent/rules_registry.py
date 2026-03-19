"""
P4: Decoupled rule registry — avoids circular imports between
validation.py and validation_rules.py.

Rules are registered at import time by validation_rules.py.
ValidationPipeline reads them from the registry without importing validation_rules directly.
"""

from __future__ import annotations

from typing import Any


class RulesRegistry:
    """Central store for validation rules.

    Avoids tight coupling between the validation pipeline and rule
    definitions.  Any module can register rules by category at import
    time, and the pipeline fetches them lazily.
    """

    _forbidden: list[tuple[str, str]] = []
    _required: list[tuple[str, Any, Any]] = []
    _anti_patterns: list[dict[str, Any]] = []
    _static_utils: list[dict[str, Any]] = []

    # ── Registration API ─────────────────────────────────────────────

    @classmethod
    def register_forbidden(cls, pattern: str, suggestion: str) -> None:
        cls._forbidden.append((pattern, suggestion))

    @classmethod
    def register_required(cls, pattern: str, severity: Any, category: Any) -> None:
        cls._required.append((pattern, severity, category))

    @classmethod
    def register_anti_pattern(cls, rule: dict[str, Any]) -> None:
        cls._anti_patterns.append(rule)

    @classmethod
    def register_static_util(cls, util: dict[str, Any]) -> None:
        cls._static_utils.append(util)

    # ── Query API ────────────────────────────────────────────────────

    @classmethod
    def get_forbidden(cls) -> list[tuple[str, str]]:
        return cls._forbidden

    @classmethod
    def get_required(cls) -> list[tuple[str, Any, Any]]:
        return cls._required

    @classmethod
    def get_anti_patterns(cls) -> list[dict[str, Any]]:
        return cls._anti_patterns

    @classmethod
    def get_static_utils(cls) -> list[dict[str, Any]]:
        return cls._static_utils

    @classmethod
    def clear(cls) -> None:
        """Reset all registries (useful for testing)."""
        cls._forbidden.clear()
        cls._required.clear()
        cls._anti_patterns.clear()
        cls._static_utils.clear()
