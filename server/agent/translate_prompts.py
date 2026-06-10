"""Translate prompt loader.

Loads externalized translate prompts from config/prompts/translate.yaml with
hot-reload. Keys absent from the file (or a missing file) return None so callers
fall back to the hardcoded defaults in server/agent/translate.py. Mirrors the
prompt_store.py pattern but for the stateless translate endpoint.
"""

from __future__ import annotations

import logging
import os
import threading

import yaml

logger = logging.getLogger("server.agent.translate_prompts")

DEFAULT_DIR = os.environ.get("PROMPT_CONFIG_DIR", "config/prompts")
_FILE = "translate.yaml"


class TranslatePrompts:
    def __init__(self, config_dir: str = DEFAULT_DIR):
        self._path = os.path.join(config_dir, _FILE)
        self._lock = threading.Lock()
        self._mtime: float = 0.0
        self._data: dict = {}
        self._load()

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
                self._data = yaml.safe_load(f) or {}
            self._mtime = mtime
            logger.info("TranslatePrompts loaded %s", self._path)
        except Exception as e:
            logger.error("TranslatePrompts failed to load %s: %s", self._path, e)

    def _reload_if_changed(self) -> None:
        try:
            if os.path.getmtime(self._path) != self._mtime:
                self._load()
        except OSError:
            pass

    def get(self, key: str) -> str | None:
        """Return the prompt for `key`, or None if file/key absent or blank."""
        with self._lock:
            self._reload_if_changed()
            val = (self._data.get("prompts") or {}).get(key)
            return val.strip() if isinstance(val, str) and val.strip() else None


_store: TranslatePrompts | None = None
_store_lock = threading.Lock()


def get_translate_prompts() -> TranslatePrompts:
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = TranslatePrompts()
    return _store


def reset_translate_prompts(config_dir: str = DEFAULT_DIR) -> TranslatePrompts:
    """Reset singleton (tests)."""
    global _store
    with _store_lock:
        _store = TranslatePrompts(config_dir)
        return _store
