"""FallbackPlugin — Section 6, Fallback subsection.

Handles any file type via regex-based extraction.
No dep resolution. Chunks extracted via regex patterns matching
common function and class declaration syntax.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from mcp_server.models import ChunkType, CodeChunk, ExtractionMode
from mcp_server.plugins.base import LanguagePlugin

# Common patterns for function/class declarations across languages
_CLASS_PATTERN = re.compile(
    r"^(?:(?:public|private|protected|abstract|static|final|export|default)\s+)*"
    r"(?:class|struct|interface|enum|type)\s+"
    r"(\w+)",
    re.MULTILINE,
)

_FUNCTION_PATTERN = re.compile(
    r"^(?:(?:public|private|protected|static|async|export|default)\s+)*"
    r"(?:def|func|function|fn)\s+"
    r"(\w+)\s*\(",
    re.MULTILINE,
)

# Method pattern: indented function-like declarations
_METHOD_PATTERN = re.compile(
    r"^\s+(?:(?:public|private|protected|static|async|override|virtual|abstract)\s+)*"
    r"(?:\w+\s+)?(\w+)\s*\([^)]*\)\s*[{:]",
    re.MULTILINE,
)


class FallbackPlugin(LanguagePlugin):
    """Regex-based fallback plugin for any file type."""

    def extensions(self) -> list[str]:
        return []  # Handles files not matched by any other plugin

    def extract_chunks(
        self, path: str, source_bytes: bytes, mode: ExtractionMode
    ) -> list[CodeChunk]:
        source = source_bytes.decode("utf-8", errors="replace")
        lines = source.split("\n")
        file_hash = hashlib.md5(source_bytes).hexdigest()
        chunks: list[CodeChunk] = []

        # Extract classes/structs
        for match in _CLASS_PATTERN.finditer(source):
            name = match.group(1)
            start_line = source[: match.start()].count("\n") + 1
            end_line = self._find_block_end(lines, start_line - 1)

            chunk = CodeChunk(
                chunk_id=CodeChunk.make_chunk_id("", path, name),
                chunk_type=ChunkType.grouping,
                symbol_name=name,
                embed_text=name if mode == ExtractionMode.names_only else match.group(0).strip(),
                body="\n".join(lines[start_line - 1 : end_line]) if mode == ExtractionMode.full_body else "",
                file_path=path,
                lang="unknown",
                start_line=start_line,
                end_line=end_line,
                file_hash=file_hash,
            )
            chunks.append(chunk)

        # Extract functions
        for match in _FUNCTION_PATTERN.finditer(source):
            name = match.group(1)
            start_line = source[: match.start()].count("\n") + 1
            end_line = self._find_block_end(lines, start_line - 1)

            chunk = CodeChunk(
                chunk_id=CodeChunk.make_chunk_id("", path, name),
                chunk_type=ChunkType.callable,
                symbol_name=name,
                embed_text=name if mode == ExtractionMode.names_only else match.group(0).strip(),
                body="\n".join(lines[start_line - 1 : end_line]) if mode == ExtractionMode.full_body else "",
                file_path=path,
                lang="unknown",
                start_line=start_line,
                end_line=end_line,
                file_hash=file_hash,
            )
            chunks.append(chunk)

        # If no structured extraction, chunk by paragraphs
        if not chunks:
            chunks = self._chunk_by_paragraphs(path, source, lines, file_hash, mode)

        return chunks

    def resolve_dep_path(
        self, import_str: str, from_file: str, repo_root: str
    ) -> Path | None:
        """FallbackPlugin does not resolve dependencies."""
        return None

    @staticmethod
    def _find_block_end(lines: list[str], start_idx: int) -> int:
        """Find the end of a code block by brace matching or indentation."""
        if start_idx >= len(lines):
            return start_idx + 1

        # Try brace matching
        brace_count = 0
        found_open = False
        for i in range(start_idx, min(start_idx + 500, len(lines))):
            for ch in lines[i]:
                if ch == "{":
                    brace_count += 1
                    found_open = True
                elif ch == "}":
                    brace_count -= 1
            if found_open and brace_count <= 0:
                return i + 1

        # Fallback: use indentation for Python-style blocks
        if not found_open and start_idx + 1 < len(lines):
            base_indent = len(lines[start_idx]) - len(lines[start_idx].lstrip())
            for i in range(start_idx + 1, min(start_idx + 500, len(lines))):
                line = lines[i]
                if line.strip() == "":
                    continue
                current_indent = len(line) - len(line.lstrip())
                if current_indent <= base_indent:
                    return i
            return min(start_idx + 500, len(lines))

        return min(start_idx + 50, len(lines))

    @staticmethod
    def _chunk_by_paragraphs(
        path: str,
        source: str,
        lines: list[str],
        file_hash: str,
        mode: ExtractionMode,
    ) -> list[CodeChunk]:
        """Chunk file by paragraphs when structured extraction fails."""
        chunks: list[CodeChunk] = []
        paragraphs: list[tuple[int, int]] = []
        start = 0
        blank_count = 0

        for i, line in enumerate(lines):
            if line.strip() == "":
                blank_count += 1
                if blank_count >= 2 and i > start:
                    paragraphs.append((start, i))
                    start = i + 1
                    blank_count = 0
            else:
                blank_count = 0

        if start < len(lines):
            paragraphs.append((start, len(lines)))

        for idx, (s, e) in enumerate(paragraphs[:20]):  # Max 20 chunks
            text = "\n".join(lines[s:e]).strip()
            if not text:
                continue
            name = f"block_{idx}"
            chunks.append(
                CodeChunk(
                    chunk_id=CodeChunk.make_chunk_id("", path, name),
                    chunk_type=ChunkType.callable,
                    symbol_name=name,
                    embed_text=name if mode == ExtractionMode.names_only else text[:200],
                    body=text if mode == ExtractionMode.full_body else "",
                    file_path=path,
                    lang="unknown",
                    start_line=s + 1,
                    end_line=e,
                    file_hash=file_hash,
                )
            )

        return chunks
