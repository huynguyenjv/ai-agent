"""
File Graph — file-level dependency graph for the repository.

Each node is a Java source file.  Edges represent import relationships:
file A imports a class defined in file B  ⟹  edge A → B.

This enables:
  • Finding which files are impacted by a change
  • Determining the minimal set of files to include as context
  • Detecting circular dependencies

Usage::

    from intelligence.repo_scanner import RepoScanner
    from intelligence.file_graph import FileGraph

    snapshot = RepoScanner().scan("/path/to/repo")
    graph = FileGraph.build(snapshot)

    deps = graph.dependencies_of("src/main/java/.../OrderService.java")
    impacted = graph.dependents_of("src/main/java/.../OrderEntity.java")
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import structlog

from .repo_scanner import RepoSnapshot
from indexer.parse_java import ClassInfo

logger = structlog.get_logger()


@dataclass
class FileNode:
    """A node in the file graph."""

    file_path: str
    package: str
    class_names: list[str] = field(default_factory=list)
    # Edges
    imports_files: set[str] = field(default_factory=set)     # outgoing: files I depend on
    imported_by_files: set[str] = field(default_factory=set)  # incoming: files that depend on me


class FileGraph:
    """Directed graph over source files, built from import relationships.

    Edges:
        A.java → B.java  means A imports a class defined in B.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, FileNode] = {}

        # Lookup: class FQN → file_path
        self._fqn_to_file: dict[str, str] = {}
        # Lookup: simple name → file_path(s)
        self._name_to_files: dict[str, list[str]] = defaultdict(list)

    # ── Construction ─────────────────────────────────────────────────

    @classmethod
    def build(cls, snapshot: RepoSnapshot) -> FileGraph:
        """Build a file graph from a ``RepoSnapshot``."""
        start = time.time()
        graph = cls()

        # Pass 1: register all files and their classes
        for ci in snapshot.classes:
            fp = ci.file_path
            if fp not in graph._nodes:
                graph._nodes[fp] = FileNode(
                    file_path=fp,
                    package=ci.package,
                )
            graph._nodes[fp].class_names.append(ci.name)
            graph._fqn_to_file[ci.fully_qualified_name] = fp
            graph._name_to_files[ci.name].append(fp)

        # Pass 2: resolve imports → edges
        for ci in snapshot.classes:
            source_file = ci.file_path
            for imp in ci.imports:
                target_file = graph._fqn_to_file.get(imp)
                if target_file and target_file != source_file:
                    graph._nodes[source_file].imports_files.add(target_file)
                    graph._nodes[target_file].imported_by_files.add(source_file)

            # Also resolve referenced type simple names
            for ref in ci.referenced_types:
                candidates = graph._name_to_files.get(ref.type_name, [])
                for target_file in candidates:
                    if target_file != source_file:
                        graph._nodes[source_file].imports_files.add(target_file)
                        graph._nodes[target_file].imported_by_files.add(source_file)

        elapsed = (time.time() - start) * 1000
        logger.info(
            "File graph built",
            nodes=len(graph._nodes),
            edges=sum(len(n.imports_files) for n in graph._nodes.values()),
            elapsed_ms=round(elapsed, 1),
        )
        return graph

    # ── Queries ──────────────────────────────────────────────────────

    def get_node(self, file_path: str) -> Optional[FileNode]:
        """Get the node for a file."""
        return self._nodes.get(file_path)

    def dependencies_of(self, file_path: str) -> list[str]:
        """Files that ``file_path`` directly depends on (imports from)."""
        node = self._nodes.get(file_path)
        return sorted(node.imports_files) if node else []

    def dependents_of(self, file_path: str) -> list[str]:
        """Files that directly import from ``file_path``."""
        node = self._nodes.get(file_path)
        return sorted(node.imported_by_files) if node else []

    def transitive_dependencies(
        self, file_path: str, max_depth: int = 3
    ) -> set[str]:
        """All files reachable by following imports up to ``max_depth``."""
        visited: set[str] = set()
        frontier = {file_path}
        for _ in range(max_depth):
            next_frontier: set[str] = set()
            for fp in frontier:
                if fp in visited:
                    continue
                visited.add(fp)
                node = self._nodes.get(fp)
                if node:
                    next_frontier |= node.imports_files
            frontier = next_frontier - visited
            if not frontier:
                break
        visited.discard(file_path)
        return visited

    def transitive_dependents(
        self, file_path: str, max_depth: int = 3
    ) -> set[str]:
        """All files that transitively depend on ``file_path``."""
        visited: set[str] = set()
        frontier = {file_path}
        for _ in range(max_depth):
            next_frontier: set[str] = set()
            for fp in frontier:
                if fp in visited:
                    continue
                visited.add(fp)
                node = self._nodes.get(fp)
                if node:
                    next_frontier |= node.imported_by_files
            frontier = next_frontier - visited
            if not frontier:
                break
        visited.discard(file_path)
        return visited

    def file_for_class(self, class_name: str) -> Optional[str]:
        """Find the file that defines a class (by simple name)."""
        candidates = self._name_to_files.get(class_name, [])
        return candidates[0] if candidates else None

    def file_for_fqn(self, fqn: str) -> Optional[str]:
        """Find the file that defines a class (by FQN)."""
        return self._fqn_to_file.get(fqn)

    # ── Statistics ───────────────────────────────────────────────────

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        return sum(len(n.imports_files) for n in self._nodes.values())

    def most_depended_on(self, top_n: int = 10) -> list[tuple[str, int]]:
        """Files with the most incoming edges (most imported)."""
        ranked = [
            (fp, len(node.imported_by_files))
            for fp, node in self._nodes.items()
        ]
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked[:top_n]

    def most_dependencies(self, top_n: int = 10) -> list[tuple[str, int]]:
        """Files with the most outgoing edges (most imports)."""
        ranked = [
            (fp, len(node.imports_files))
            for fp, node in self._nodes.items()
        ]
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked[:top_n]

    def get_summary(self) -> dict:
        return {
            "total_files": self.node_count,
            "total_edges": self.edge_count,
            "most_depended_on": self.most_depended_on(5),
            "most_dependencies": self.most_dependencies(5),
        }
