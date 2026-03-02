"""
Repo Scanner — scans a Java repository and produces structured metadata.

Wraps the existing ``indexer.parse_java.JavaParser`` and augments the
output with additional in-memory indexes (name → ClassInfo, FQN → ClassInfo,
file → ClassInfo list).

Usage::

    scanner = RepoScanner()
    snapshot = scanner.scan("/path/to/repo")
    cls = snapshot.get_class("OrderService")
"""

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import structlog

from indexer.parse_java import JavaParser, ClassInfo

logger = structlog.get_logger()


@dataclass
class RepoSnapshot:
    """Immutable snapshot of a repository's parsed structure.

    Provides O(1) lookups by class name, FQN, and file path.
    """

    repo_path: str
    classes: list[ClassInfo]
    scanned_at: float = field(default_factory=time.time)

    # Lookup indexes (built lazily on first access)
    _by_name: dict[str, list[ClassInfo]] = field(default_factory=dict, repr=False)
    _by_fqn: dict[str, ClassInfo] = field(default_factory=dict, repr=False)
    _by_file: dict[str, list[ClassInfo]] = field(default_factory=dict, repr=False)
    _indexed: bool = field(default=False, repr=False)

    def _ensure_indexed(self) -> None:
        if self._indexed:
            return
        for cls in self.classes:
            self._by_fqn[cls.fully_qualified_name] = cls
            self._by_name.setdefault(cls.name, []).append(cls)
            self._by_file.setdefault(cls.file_path, []).append(cls)
        self._indexed = True

    # ── Lookups ──────────────────────────────────────────────────────

    def get_class(self, name: str) -> Optional[ClassInfo]:
        """Get a class by simple name.  Returns first match if ambiguous."""
        self._ensure_indexed()
        matches = self._by_name.get(name)
        return matches[0] if matches else None

    def get_classes_by_name(self, name: str) -> list[ClassInfo]:
        """Get all classes with a given simple name (handles duplicates)."""
        self._ensure_indexed()
        return self._by_name.get(name, [])

    def get_by_fqn(self, fqn: str) -> Optional[ClassInfo]:
        """Get a class by fully-qualified name."""
        self._ensure_indexed()
        return self._by_fqn.get(fqn)

    def get_by_file(self, file_path: str) -> list[ClassInfo]:
        """Get all classes defined in a file."""
        self._ensure_indexed()
        normalized = file_path.replace("\\", "/")
        return self._by_file.get(normalized, [])

    def list_files(self) -> list[str]:
        """List all Java files that were parsed."""
        self._ensure_indexed()
        return sorted(self._by_file.keys())

    def list_packages(self) -> list[str]:
        """List all unique package names."""
        return sorted({cls.package for cls in self.classes if cls.package})

    @property
    def class_count(self) -> int:
        return len(self.classes)

    @property
    def file_count(self) -> int:
        self._ensure_indexed()
        return len(self._by_file)

    def get_summary(self) -> dict:
        """Summary statistics."""
        self._ensure_indexed()
        type_counts: dict[str, int] = {}
        for cls in self.classes:
            type_counts[cls.class_type] = type_counts.get(cls.class_type, 0) + 1
        return {
            "repo_path": self.repo_path,
            "total_classes": self.class_count,
            "total_files": self.file_count,
            "packages": len(self.list_packages()),
            "type_distribution": type_counts,
            "scanned_at": self.scanned_at,
        }


class RepoScanner:
    """Scans a Java repo and produces a ``RepoSnapshot``."""

    def __init__(self):
        self._parser = JavaParser()

    def scan(self, repo_path: str) -> RepoSnapshot:
        """Parse all Java files in ``repo_path`` and return a snapshot."""
        start = time.time()
        logger.info("Scanning repository", repo_path=repo_path)

        classes = self._parser.parse_directory(repo_path)

        elapsed = (time.time() - start) * 1000
        logger.info(
            "Repository scan complete",
            repo_path=repo_path,
            classes=len(classes),
            elapsed_ms=round(elapsed, 1),
        )

        return RepoSnapshot(repo_path=repo_path, classes=classes)

    def scan_file(self, file_path: str) -> list[ClassInfo]:
        """Parse a single Java file."""
        return self._parser.parse_file(file_path)
