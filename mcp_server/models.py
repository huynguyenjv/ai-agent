"""Data models — Section 4 of the implementation plan."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

# Section 5: Hardcoded blocklists shared across tools
SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".terraform",
    "vendor", "target", "build", "dist", ".venv", "venv",
    ".idea", ".vscode",
}

SKIP_EXTENSIONS = {".min.js", ".lock", ".jar", ".class", ".pyc", ".map"}


class ChunkType(str, Enum):
    """Section 4.2 — chunk_type semantics."""

    callable = "callable"
    grouping = "grouping"
    dependency = "dependency"
    config_block = "config_block"


class ExtractionMode(str, Enum):
    """Extraction modes used by index_with_deps depth control."""

    full_body = "full_body"
    signatures = "signatures"
    names_only = "names_only"


@dataclass
class CodeChunk:
    """Section 4.1 — The universal unit of indexed knowledge.

    Every language plugin produces CodeChunk objects.
    No downstream component sees any language-specific structure.
    """

    chunk_id: str
    chunk_type: ChunkType
    symbol_name: str
    embed_text: str
    body: str
    file_path: str
    lang: str
    start_line: int
    end_line: int
    deps: list[str] = field(default_factory=list)
    file_hash: str = ""
    raw_imports: list[str] = field(default_factory=list)

    @staticmethod
    def make_chunk_id(repo_root: str, relative_path: str, symbol_name: str) -> str:
        """Deterministic hash of repo_root + relative_path + symbol_name."""
        key = f"{repo_root}:{relative_path}:{symbol_name}"
        return hashlib.md5(key.encode()).hexdigest()

    def to_payload(self) -> dict:
        """Convert to payload dict for upload (excludes raw_imports per Section 4.4)."""
        return {
            "chunk_id": self.chunk_id,
            "chunk_type": self.chunk_type.value,
            "symbol_name": self.symbol_name,
            "embed_text": self.embed_text,
            "body": self.body,
            "file_path": self.file_path,
            "lang": self.lang,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "deps": self.deps,
            "file_hash": self.file_hash,
        }
