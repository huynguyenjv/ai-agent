"""
Repo Intelligence Layer – structural understanding of Java codebases.

This package provides graph-based and symbol-level intelligence that
goes beyond vector search.  It builds:

  • RepoScanner  — scans a repo, produces ClassInfo list
  • FileGraph    — file-level dependency graph (who imports whom)
  • SymbolMap    — global symbol table (class → methods, fields, FQN)
  • DependencyAnalyzer — transitive dependency resolution, impact analysis

All structures are built from the existing ``indexer.parse_java.JavaParser``
and are kept in-memory for fast retrieval during agent execution.
"""

from .repo_scanner import RepoScanner
from .file_graph import FileGraph, FileNode
from .symbol_map import SymbolMap, SymbolEntry
from .dependency_analyzer import DependencyAnalyzer

__all__ = [
    "RepoScanner",
    "FileGraph",
    "FileNode",
    "SymbolMap",
    "SymbolEntry",
    "DependencyAnalyzer",
]
