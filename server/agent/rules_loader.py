"""Hot-reloadable rules loader for intent classification.

Loads rules from config/rules.yaml and checks for changes periodically.
Changes apply without server restart.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger("server.rules_loader")

# Default rules if file is missing or invalid
DEFAULT_RULES: dict[str, Any] = {
    "intents": [
        {
            "name": "code_gen",
            "description": "Default fallback intent",
            "keywords_vi": [],
            "keywords_en": [],
            "tools": [],
            "requires_file": False,
            "priority": 99,
        }
    ],
    "freshness_keywords": [],
    "file_mention_suffixes": ["Service", "Controller", "Repository"],
    "file_extensions": [".java", ".py", ".go", ".ts"],
    "confidence_threshold": 0.7,
    "rules_reload_interval_seconds": 30,
}


class RulesLoader:
    """Thread-safe hot-reloadable rules loader.

    Loads rules from YAML file and checks for changes periodically.
    On reload failure, keeps the last valid rules.
    """

    def __init__(self, rules_path: Path | None = None):
        """Initialize the rules loader.

        Args:
            rules_path: Path to rules.yaml. Defaults to config/rules.yaml
                       relative to project root.
        """
        if rules_path is None:
            # Resolve relative to project root
            rules_path = Path(__file__).parent.parent.parent / "config" / "rules.yaml"

        self._rules_path = rules_path
        self._rules: dict[str, Any] = DEFAULT_RULES.copy()
        self._last_mtime: float = 0.0
        self._last_check: float = 0.0
        self._lock = threading.Lock()
        self._cached_sorted_intents: list[dict] | None = None

        # Initial load
        self._load_rules()

    def _load_rules(self) -> None:
        """Load rules from YAML file. Keep old rules on failure."""
        try:
            if not self._rules_path.exists():
                logger.warning("Rules file not found: %s, using defaults", self._rules_path)
                return

            mtime = self._rules_path.stat().st_mtime

            with self._lock:
                if mtime == self._last_mtime:
                    return  # No change

                with open(self._rules_path, "r", encoding="utf-8") as f:
                    new_rules = yaml.safe_load(f)

                if not isinstance(new_rules, dict):
                    logger.warning("Invalid rules format (not a dict), keeping old rules")
                    return

                # Validate required keys
                if "intents" not in new_rules or not isinstance(new_rules["intents"], list):
                    logger.warning("Rules missing 'intents' list, keeping old rules")
                    return

                self._rules = new_rules
                self._last_mtime = mtime
                self._cached_sorted_intents = None  # Invalidate cache
                logger.info("Reloaded rules from %s", self._rules_path)

        except yaml.YAMLError as e:
            logger.warning("YAML parse error in rules file: %s, keeping old rules", e)
        except Exception as e:
            logger.warning("Failed to load rules: %s, keeping old rules", e)

    def _maybe_reload(self) -> None:
        """Check if reload is needed based on time interval."""
        now = time.time()
        interval = self._rules.get("rules_reload_interval_seconds", 30)

        if now - self._last_check < interval:
            return

        self._last_check = now
        self._load_rules()

    def get_rules(self) -> dict[str, Any]:
        """Get the full rules dict (hot-reloaded)."""
        self._maybe_reload()
        return self._rules

    def get_intents(self) -> list[dict]:
        """Get intents list sorted by priority ascending (cached)."""
        self._maybe_reload()
        if self._cached_sorted_intents is None:
            with self._lock:
                if self._cached_sorted_intents is None:
                    intents = self._rules.get("intents", [])
                    self._cached_sorted_intents = sorted(
                        intents, key=lambda x: x.get("priority", 99)
                    )
        return self._cached_sorted_intents

    def get_intent_names(self) -> set[str]:
        """Get set of valid intent names."""
        return {intent["name"] for intent in self.get_intents()}

    def get_freshness_keywords(self) -> list[str]:
        """Get freshness keywords list."""
        return self.get_rules().get("freshness_keywords", [])

    def get_file_suffixes(self) -> list[str]:
        """Get file mention suffixes list."""
        return self.get_rules().get("file_mention_suffixes", [])

    def get_file_extensions(self) -> list[str]:
        """Get file extensions list."""
        return self.get_rules().get("file_extensions", [".java", ".py", ".go", ".ts"])

    def get_confidence_threshold(self) -> float:
        """Get confidence threshold value."""
        return float(self.get_rules().get("confidence_threshold", 0.7))

    def get_reload_interval(self) -> int:
        """Get reload interval in seconds."""
        return int(self.get_rules().get("rules_reload_interval_seconds", 30))


# Module-level singleton
_rules_loader: RulesLoader | None = None
_rules_loader_lock = threading.Lock()


def get_rules_loader() -> RulesLoader:
    """Get the singleton RulesLoader instance."""
    global _rules_loader

    if _rules_loader is None:
        with _rules_loader_lock:
            if _rules_loader is None:
                _rules_loader = RulesLoader()

    return _rules_loader


def reset_rules_loader(rules_path: Path | None = None) -> RulesLoader:
    """Reset the singleton (for testing). Returns new instance."""
    global _rules_loader

    with _rules_loader_lock:
        _rules_loader = RulesLoader(rules_path)
        return _rules_loader
