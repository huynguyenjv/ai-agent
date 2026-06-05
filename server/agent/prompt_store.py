"""Prompt Versioning — Phase 16.4.

Loads versioned intent system prompts from YAML (config/prompts/intents.yaml)
with hot-reload, named variants (for A/B testing), version tracking, and
rollback to the previously-loaded version. Intents absent from the YAML return
None so callers can fall back to the hardcoded defaults.
"""

from __future__ import annotations

import logging
import os
import threading

import yaml

logger = logging.getLogger("server.agent.prompt_store")

DEFAULT_DIR = os.environ.get("PROMPT_CONFIG_DIR", "config/prompts")
_INTENTS_FILE = "intents.yaml"


class PromptStore:
    """Versioned prompt loader with hot-reload + variants + rollback."""

    def __init__(self, config_dir: str = DEFAULT_DIR):
        self._path = os.path.join(config_dir, _INTENTS_FILE)
        self._lock = threading.Lock()
        self._mtime: float = 0.0
        self._data: dict = {}
        self._prev_data: dict | None = None  # for rollback
        self._load()

    # -- loading / hot-reload -------------------------------------------------
    def _load(self) -> None:
        try:
            mtime = os.path.getmtime(self._path)
        except OSError:
            self._data = {}
            return

        if mtime == self._mtime and self._data:
            return

        try:
            with open(self._path, "r", encoding="utf-8") as f:
                new_data = yaml.safe_load(f) or {}
            # keep previous for rollback
            if self._data:
                self._prev_data = self._data
            self._data = new_data
            self._mtime = mtime
            logger.info("PromptStore loaded %s (version=%s)",
                        self._path, new_data.get("version"))
        except Exception as e:
            logger.error("PromptStore failed to load %s: %s", self._path, e)

    def _reload_if_changed(self) -> None:
        try:
            if os.path.getmtime(self._path) != self._mtime:
                self._load()
        except OSError:
            pass

    # -- public API -----------------------------------------------------------
    def get_intent(self, intent: str, variant: str = "default") -> str | None:
        """Return the system prompt for an intent (+variant), or None if absent."""
        with self._lock:
            self._reload_if_changed()
            spec = (self._data.get("intents") or {}).get(intent)
            if not spec:
                return None
            if variant and variant != "default":
                vspec = (spec.get("variants") or {}).get(variant)
                if vspec and vspec.get("system"):
                    return vspec["system"].strip()
            system = spec.get("system")
            return system.strip() if system else None

    def variants_for(self, intent: str) -> list[str]:
        with self._lock:
            self._reload_if_changed()
            spec = (self._data.get("intents") or {}).get(intent) or {}
            return ["default", *list((spec.get("variants") or {}).keys())]

    def version(self) -> str:
        with self._lock:
            self._reload_if_changed()
            return str(self._data.get("version", "unknown"))

    def rollback(self) -> bool:
        """Revert to the previously-loaded version (in-memory). True if rolled back."""
        with self._lock:
            if self._prev_data is None:
                return False
            self._data, self._prev_data = self._prev_data, None
            self._mtime = 0.0  # force reconcile on next change
            logger.warning("PromptStore rolled back to version=%s",
                           self._data.get("version"))
            return True


_store: PromptStore | None = None
_store_lock = threading.Lock()


def get_prompt_store() -> PromptStore:
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = PromptStore()
    return _store


def reset_prompt_store(config_dir: str = DEFAULT_DIR) -> PromptStore:
    """Reset singleton (tests)."""
    global _store
    with _store_lock:
        _store = PromptStore(config_dir)
        return _store
