"""Plugin Registry — Section 6, Plugin Registry subsection.

Maintains a dictionary mapping file extensions to plugin instances.
"""

from __future__ import annotations

from pathlib import Path

from mcp_server.plugins.base import LanguagePlugin
from mcp_server.plugins.fallback import FallbackPlugin


class PluginRegistry:
    """Maps file extensions to language plugin instances.

    Lookup order:
    1. File extension (lowercase)
    2. Shebang line inspection (python, node, deno)
    3. FallbackPlugin
    """

    def __init__(self) -> None:
        self._ext_map: dict[str, LanguagePlugin] = {}
        self._fallback = FallbackPlugin()
        self._plugins: list[LanguagePlugin] = []

    def register(self, plugin: LanguagePlugin) -> None:
        """Register a plugin for its declared extensions."""
        self._plugins.append(plugin)
        for ext in plugin.extensions():
            self._ext_map[ext.lower()] = plugin

    def get_plugin(self, path: str) -> LanguagePlugin:
        """Return the appropriate plugin for a file path.

        1. Look up by file extension.
        2. If not found, inspect shebang line.
        3. If still not found, return FallbackPlugin.
        """
        ext = Path(path).suffix.lower()
        plugin = self._ext_map.get(ext)
        if plugin is not None:
            return plugin

        # Try shebang detection
        plugin = self._detect_by_shebang(path)
        if plugin is not None:
            return plugin

        return self._fallback

    def _detect_by_shebang(self, path: str) -> LanguagePlugin | None:
        """Inspect the first line of a file for a shebang hint."""
        try:
            with open(path, "rb") as f:
                first_line = f.readline(256).decode("utf-8", errors="replace").strip()
        except (OSError, UnicodeDecodeError):
            return None

        if not first_line.startswith("#!"):
            return None

        shebang = first_line.lower()
        shebang_map = {
            "python": ".py",
            "node": ".js",
            "deno": ".ts",
        }
        for keyword, ext in shebang_map.items():
            if keyword in shebang:
                return self._ext_map.get(ext)

        return None

    @property
    def fallback(self) -> FallbackPlugin:
        return self._fallback
