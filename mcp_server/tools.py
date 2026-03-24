"""MCP Tools — Section 5: read_file and search_symbol.

Phase 1 implements read_file and search_symbol.
get_project_skeleton and index_with_deps are added in Phase 3.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from mcp_server.models import ExtractionMode, SKIP_DIRS, SKIP_EXTENSIONS
from mcp_server.plugins.registry import PluginRegistry

logger = logging.getLogger("mcp_server.tools")


def read_file(
    repo_path: str,
    file_path: str,
    start_line: int = 1,
    end_line: int = 150,
) -> dict:
    """Section 5, Tool: read_file.

    Read a contiguous range of lines from a file.
    Critical constraint: result must NEVER be uploaded to Qdrant.
    """
    abs_path = os.path.join(repo_path, file_path)

    if not os.path.isfile(abs_path):
        return {"error": f"File not found: {file_path}"}

    # Reject paths outside REPO_PATH
    real_repo = os.path.realpath(repo_path)
    real_file = os.path.realpath(abs_path)
    if not real_file.startswith(real_repo):
        return {"error": f"File not found: {file_path}"}

    try:
        with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
    except OSError as e:
        return {"error": f"Cannot read file: {e}"}

    total_lines = len(all_lines)
    start = max(1, start_line)
    end = min(total_lines, end_line)

    selected = all_lines[start - 1 : end]
    content = "".join(selected)

    return {
        "content": content,
        "start_line": start,
        "end_line": end,
        "total_lines": total_lines,
        "file_path": file_path,
    }


def search_symbol(
    repo_path: str,
    registry: PluginRegistry,
    name: str,
    type_filter: str = "any",
) -> list[dict]:
    """Section 5, Tool: search_symbol.

    Locate a class, function, or method by name anywhere in the repository.
    Uses the same skip rules as get_project_skeleton.
    """
    matches: list[dict] = []

    for root, dirs, files in os.walk(repo_path):
        # Prune blocked directories
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]

        for fname in files:
            # Skip blocked extensions
            ext = Path(fname).suffix.lower()
            if ext in SKIP_EXTENSIONS:
                continue

            full_path = os.path.join(root, fname)
            rel_path = os.path.relpath(full_path, repo_path).replace("\\", "/")

            plugin = registry.get_plugin(full_path)

            try:
                with open(full_path, "rb") as f:
                    source = f.read()
            except OSError:
                continue

            try:
                chunks = plugin.extract_chunks(rel_path, source, ExtractionMode.names_only)
            except Exception as e:
                logger.debug("Failed to parse %s: %s", rel_path, e)
                continue

            for chunk in chunks:
                if name.lower() not in chunk.symbol_name.lower():
                    continue

                # Apply type_filter
                if type_filter != "any":
                    type_map = {
                        "class": "grouping",
                        "function": "callable",
                        "method": "callable",
                    }
                    expected = type_map.get(type_filter)
                    if expected and chunk.chunk_type.value != expected:
                        continue

                matches.append({
                    "symbol_name": chunk.symbol_name,
                    "chunk_type": chunk.chunk_type.value,
                    "file_path": chunk.file_path,
                    "start_line": chunk.start_line,
                    "lang": chunk.lang,
                })

    return matches
