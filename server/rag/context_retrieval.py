"""Parent-Child context retrieval for RAG.

Retrieves surrounding context for chunks to provide better
understanding of code structure.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any

logger = logging.getLogger("server.rag.context_retrieval")

# Default context lines before/after chunk
DEFAULT_CONTEXT_LINES = int(os.environ.get("RAG_CONTEXT_LINES", "30"))

# Cache for file contents
_file_cache: dict[str, list[str]] = {}
MAX_CACHE_SIZE = 100


def get_file_lines(file_path: str, repo_path: str) -> list[str] | None:
    """Get file content as lines with caching.

    Args:
        file_path: Relative file path
        repo_path: Repository root path

    Returns:
        List of lines, or None if file not found
    """
    cache_key = f"{repo_path}:{file_path}"

    if cache_key in _file_cache:
        return _file_cache[cache_key]

    # Evict oldest if cache full
    if len(_file_cache) >= MAX_CACHE_SIZE:
        oldest = next(iter(_file_cache))
        del _file_cache[oldest]

    full_path = os.path.join(repo_path, file_path)

    try:
        with open(full_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
            _file_cache[cache_key] = lines
            return lines
    except OSError as e:
        logger.warning("Failed to read file %s: %s", file_path, e)
        return None


def retrieve_with_context(
    chunks: list[dict],
    repo_path: str,
    context_lines: int = DEFAULT_CONTEXT_LINES,
) -> list[dict]:
    """Retrieve chunks with surrounding context.

    Args:
        chunks: RAG chunks to enrich
        repo_path: Repository root path
        context_lines: Lines of context before/after

    Returns:
        Chunks enriched with context
    """
    if not chunks:
        return []

    enriched = []

    for chunk in chunks:
        file_path = chunk.get("file_path", "")
        start_line = chunk.get("start_line", 1)
        end_line = chunk.get("end_line", start_line)

        # Get file content
        lines = get_file_lines(file_path, repo_path)

        if lines is None:
            # Can't get context, keep original chunk
            enriched.append(chunk)
            continue

        # Calculate context window
        context_start = max(0, start_line - 1 - context_lines)
        context_end = min(len(lines), end_line + context_lines)

        # Extract context
        context_before = "".join(lines[context_start:start_line - 1])
        context_after = "".join(lines[end_line:context_end])

        # Build enriched chunk
        enriched_chunk = {
            **chunk,
            "context_before": context_before,
            "context_after": context_after,
            "context_start_line": context_start + 1,
            "context_end_line": context_end,
            "has_context": bool(context_before or context_after),
        }

        enriched.append(enriched_chunk)

    logger.info("context_retrieval: enriched %d chunks with context", len(enriched))

    return enriched


def get_parent_context(
    chunk: dict,
    repo_path: str,
) -> dict | None:
    """Get parent (file-level) context for a chunk.

    Useful for understanding imports, class definitions, etc.

    Args:
        chunk: RAG chunk
        repo_path: Repository root

    Returns:
        Parent context dict or None
    """
    file_path = chunk.get("file_path", "")
    lines = get_file_lines(file_path, repo_path)

    if lines is None:
        return None

    # Get file header (imports, class def, etc.)
    # Usually first 30 lines contain important context
    header_lines = min(30, len(lines))
    header = "".join(lines[:header_lines])

    # Get class/function containing this chunk
    start_line = chunk.get("start_line", 1)
    container = _find_container(lines, start_line)

    return {
        "file_path": file_path,
        "header": header,
        "header_lines": header_lines,
        "container": container,
    }


def _find_container(lines: list[str], target_line: int) -> dict | None:
    """Find the class or function containing a line.

    Args:
        lines: File lines
        target_line: Target line number (1-based)

    Returns:
        Container info or None
    """
    # Simple heuristic: look backwards for class/def
    for i in range(target_line - 1, -1, -1):
        line = lines[i].strip()

        if line.startswith("class ") or line.startswith("def "):
            # Found container
            name = line.split("(")[0].split(":")[0]
            name = name.replace("class ", "").replace("def ", "").strip()

            return {
                "type": "class" if line.startswith("class") else "function",
                "name": name,
                "line": i + 1,
            }

        # Java/TS style
        if "class " in line or "interface " in line:
            parts = line.split()
            for j, part in enumerate(parts):
                if part in ("class", "interface") and j + 1 < len(parts):
                    return {
                        "type": part,
                        "name": parts[j + 1].rstrip("{:"),
                        "line": i + 1,
                    }

    return None


def build_context_prompt(
    chunks: list[dict],
    repo_path: str,
    max_context_chars: int = 8000,
) -> str:
    """Build a context string from chunks with surrounding context.

    Args:
        chunks: RAG chunks
        repo_path: Repository root
        max_context_chars: Maximum characters for context

    Returns:
        Formatted context string
    """
    enriched = retrieve_with_context(chunks, repo_path)

    parts = []
    total_chars = 0

    for chunk in enriched:
        file_path = chunk.get("file_path", "unknown")
        start = chunk.get("start_line", "?")
        end = chunk.get("end_line", "?")
        body = chunk.get("body", chunk.get("content", ""))

        # Build chunk text
        chunk_text = f"# {file_path} (lines {start}-{end})\n"

        if chunk.get("context_before"):
            chunk_text += f"# ... context before ...\n{chunk['context_before']}"

        chunk_text += body

        if chunk.get("context_after"):
            chunk_text += f"\n{chunk['context_after']}# ... context after ..."

        chunk_text += "\n\n"

        # Check size limit
        if total_chars + len(chunk_text) > max_context_chars:
            # Truncate this chunk
            remaining = max_context_chars - total_chars
            if remaining > 200:
                chunk_text = chunk_text[:remaining] + "\n[truncated]\n"
                parts.append(chunk_text)
            break

        parts.append(chunk_text)
        total_chars += len(chunk_text)

    return "".join(parts)


def clear_cache():
    """Clear the file content cache."""
    _file_cache.clear()
