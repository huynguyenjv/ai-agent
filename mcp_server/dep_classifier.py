"""DepClassifier — Section 6, DepClassifier subsection.

Two-step classification logic for each import string:
1. Attempt resolution via plugin.resolve_dep_path
2. Check stdlib list

Only project deps are fed back into the BFS indexing queue.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from mcp_server.plugins.base import LanguagePlugin

# Section 6: Stdlib Prefix Lists (authoritative, must not be reduced)
STDLIB_PREFIXES: dict[str, list[str]] = {
    "java": ["java.", "javax.", "sun.", "com.sun."],
    "csharp": ["System.", "Microsoft.", "Windows."],
    "go": [
        "fmt", "os", "io", "net", "strings", "strconv", "sync", "context",
        "errors", "math", "sort", "time", "encoding", "crypto", "path",
        "bufio", "bytes", "runtime", "reflect", "log",
    ],
    "python": [
        "os", "sys", "re", "json", "pathlib", "typing", "collections",
        "itertools", "functools", "abc", "datetime", "math", "random",
        "hashlib", "logging", "asyncio", "dataclasses", "enum", "copy", "io",
    ],
    "typescript": [
        "node:fs", "node:path", "node:http", "node:os",
        "node:crypto", "node:events", "node:stream",
    ],
}


class DepType(str, Enum):
    project = "project"
    stdlib = "stdlib"
    third_party = "third_party"


class DepClassifier:
    """Classifies import strings as project, stdlib, or third_party."""

    def classify(
        self,
        import_str: str,
        lang: str,
        plugin: LanguagePlugin,
        from_file: str,
        repo_root: str,
    ) -> tuple[DepType, Path | None]:
        """Classify an import and return (type, resolved_path_or_None).

        Step 1: Attempt resolution via plugin.
        Step 2: Check stdlib list.
        Default: third_party.
        """
        # Step 1 — Attempt resolution
        resolved = plugin.resolve_dep_path(import_str, from_file, repo_root)
        if resolved is not None:
            full_path = Path(repo_root) / resolved
            if full_path.is_file():
                return DepType.project, resolved

        # Step 2 — Check stdlib list
        clean = self._clean_import(import_str, lang)
        prefixes = STDLIB_PREFIXES.get(lang, [])
        for prefix in prefixes:
            if lang == "go":
                # Go stdlib: exact match or sub-package
                if clean == prefix or clean.startswith(prefix + "/"):
                    return DepType.stdlib, None
            else:
                if clean.startswith(prefix):
                    return DepType.stdlib, None

        # TypeScript: non-relative imports without node: prefix are third_party
        if lang == "typescript" and not import_str.startswith("."):
            if import_str.startswith("node:"):
                return DepType.stdlib, None
            return DepType.third_party, None

        return DepType.third_party, None

    @staticmethod
    def _clean_import(import_str: str, lang: str) -> str:
        """Clean import string for stdlib comparison."""
        clean = import_str.strip()

        if lang == "python":
            # Handle "from os import path" -> "os"
            for prefix in ("from ", "import "):
                if clean.startswith(prefix):
                    clean = clean[len(prefix):]
            if " import " in clean:
                clean = clean.split(" import ")[0].strip()

        elif lang == "java":
            clean = clean.replace("import ", "").replace("static ", "").rstrip(";").strip()

        elif lang == "csharp":
            clean = clean.replace("using ", "").replace("static ", "").rstrip(";").strip()

        elif lang == "go":
            clean = clean.strip('"')

        return clean
