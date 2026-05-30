"""Chunking strategies for RAG indexing.

Provides overlapping chunk strategies to improve retrieval
by ensuring context is preserved across chunk boundaries.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("server.rag.chunking")

# Configuration via environment
DEFAULT_CHUNK_SIZE = int(os.environ.get("RAG_CHUNK_SIZE", "100"))
DEFAULT_OVERLAP = int(os.environ.get("RAG_CHUNK_OVERLAP", "20"))


@dataclass
class Chunk:
    """A text chunk with metadata."""
    content: str
    start_line: int
    end_line: int
    has_overlap: bool = False
    overlap_start: int = 0
    overlap_end: int = 0


def chunk_by_lines(
    content: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Split content into overlapping chunks by lines.

    Args:
        content: File content
        chunk_size: Target lines per chunk
        overlap: Lines of overlap between chunks

    Returns:
        List of Chunk objects
    """
    lines = content.split("\n")
    total_lines = len(lines)

    if total_lines <= chunk_size:
        return [Chunk(
            content=content,
            start_line=1,
            end_line=total_lines,
            has_overlap=False,
        )]

    chunks = []
    start = 0

    while start < total_lines:
        end = min(start + chunk_size, total_lines)

        chunk_lines = lines[start:end]
        chunk_content = "\n".join(chunk_lines)

        chunk = Chunk(
            content=chunk_content,
            start_line=start + 1,
            end_line=end,
            has_overlap=start > 0,
            overlap_start=start + 1 if start > 0 else 0,
            overlap_end=start + overlap if start > 0 else 0,
        )
        chunks.append(chunk)

        # Move start with overlap
        next_start = end - overlap
        if next_start <= start:
            # Prevent infinite loop
            next_start = end

        if next_start >= total_lines:
            break

        start = next_start

    logger.debug("chunking: split %d lines into %d chunks (size=%d, overlap=%d)",
                 total_lines, len(chunks), chunk_size, overlap)

    return chunks


def chunk_by_tokens(
    content: str,
    max_tokens: int = 500,
    overlap_tokens: int = 50,
) -> list[Chunk]:
    """Split content into overlapping chunks by approximate tokens.

    Uses simple whitespace tokenization as approximation.

    Args:
        content: File content
        max_tokens: Target tokens per chunk
        overlap_tokens: Tokens of overlap

    Returns:
        List of Chunk objects
    """
    # Simple tokenization by whitespace
    words = content.split()
    total_tokens = len(words)

    if total_tokens <= max_tokens:
        return [Chunk(
            content=content,
            start_line=1,
            end_line=content.count("\n") + 1,
            has_overlap=False,
        )]

    chunks = []
    start = 0

    while start < total_tokens:
        end = min(start + max_tokens, total_tokens)

        chunk_words = words[start:end]
        chunk_content = " ".join(chunk_words)

        # Approximate line numbers
        start_line = content[:content.find(chunk_words[0])].count("\n") + 1 if chunk_words else 1
        end_line = start_line + chunk_content.count("\n")

        chunk = Chunk(
            content=chunk_content,
            start_line=start_line,
            end_line=end_line,
            has_overlap=start > 0,
        )
        chunks.append(chunk)

        next_start = end - overlap_tokens
        if next_start <= start:
            next_start = end

        if next_start >= total_tokens:
            break

        start = next_start

    return chunks


def chunk_by_syntax(
    content: str,
    lang: str,
    max_lines: int = 150,
) -> list[Chunk]:
    """Split content by syntax boundaries (class, function, etc.).

    Tries to keep logical units together.

    Args:
        content: File content
        lang: Language (python, java, etc.)
        max_lines: Maximum lines per chunk

    Returns:
        List of Chunk objects
    """
    lines = content.split("\n")

    # Find syntax boundaries
    boundaries = _find_syntax_boundaries(lines, lang)

    if not boundaries:
        # Fall back to line-based chunking
        return chunk_by_lines(content, max_lines, max_lines // 5)

    chunks = []
    current_start = 0

    for boundary in boundaries:
        # Check if adding to current chunk would exceed limit
        if boundary - current_start > max_lines and current_start < boundary:
            # Create chunk up to this boundary
            chunk_lines = lines[current_start:boundary]
            chunks.append(Chunk(
                content="\n".join(chunk_lines),
                start_line=current_start + 1,
                end_line=boundary,
                has_overlap=False,
            ))
            current_start = boundary

    # Final chunk
    if current_start < len(lines):
        chunk_lines = lines[current_start:]
        chunks.append(Chunk(
            content="\n".join(chunk_lines),
            start_line=current_start + 1,
            end_line=len(lines),
            has_overlap=False,
        ))

    return chunks


def _find_syntax_boundaries(lines: list[str], lang: str) -> list[int]:
    """Find syntax boundaries in code.

    Args:
        lines: Code lines
        lang: Language

    Returns:
        List of line numbers that are good split points
    """
    boundaries = []

    patterns = {
        "python": ["def ", "class ", "async def "],
        "java": ["public class ", "private class ", "class ", "public void ", "private void ", "public static "],
        "typescript": ["export class ", "export function ", "export const ", "function ", "class "],
        "javascript": ["function ", "class ", "const ", "export "],
        "go": ["func ", "type "],
    }

    lang_patterns = patterns.get(lang, patterns.get("python", []))

    for i, line in enumerate(lines):
        stripped = line.strip()
        for pattern in lang_patterns:
            if stripped.startswith(pattern):
                boundaries.append(i)
                break

    return boundaries


def dedup_overlapping_results(
    results: list[dict],
    overlap_threshold: float = 0.5,
) -> list[dict]:
    """Remove duplicate results from overlapping chunks.

    Args:
        results: Search results
        overlap_threshold: Similarity threshold to consider duplicate

    Returns:
        Deduplicated results
    """
    if len(results) <= 1:
        return results

    deduped = []
    seen_ranges = []

    for result in results:
        file_path = result.get("file_path", "")
        start = result.get("start_line", 0)
        end = result.get("end_line", start)

        # Check if this overlaps significantly with seen results
        is_dup = False
        for seen_file, seen_start, seen_end in seen_ranges:
            if seen_file != file_path:
                continue

            # Calculate overlap
            overlap_start = max(start, seen_start)
            overlap_end = min(end, seen_end)

            if overlap_start < overlap_end:
                overlap_size = overlap_end - overlap_start
                result_size = end - start
                if result_size > 0 and overlap_size / result_size > overlap_threshold:
                    is_dup = True
                    break

        if not is_dup:
            deduped.append(result)
            seen_ranges.append((file_path, start, end))

    if len(results) != len(deduped):
        logger.info("chunking: deduped %d → %d results", len(results), len(deduped))

    return deduped
