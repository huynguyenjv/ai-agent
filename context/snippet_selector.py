"""
Snippet Selector — picks the minimal set of code snippets needed for a prompt.

Given a target class and the intelligence layer's analysis, selects and
prioritises snippets:

    Priority 1 (MUST)   — target class source code
    Priority 2 (MUST)   — injectable dependencies (will become @Mock)
    Priority 3 (SHOULD) — domain types used in method signatures (DTOs, entities)
    Priority 4 (NICE)   — related interfaces / abstract bases
    Priority 5 (EXTRA)  — transitive dependencies (only if budget allows)

Each snippet carries a role and a priority score; the TokenOptimizer
can then drop lower-priority snippets if the budget is tight.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import structlog

from intelligence.dependency_analyzer import DependencyAnalyzer, TestContext
from intelligence.symbol_map import SymbolEntry, SymbolMap
from rag.schema import CodeChunk

logger = structlog.get_logger()


# ── Data types ───────────────────────────────────────────────────────

class SnippetRole(Enum):
    """Role of a code snippet in the prompt."""

    SOURCE = "source"           # Target class source code
    DEPENDENCY = "dependency"   # Injectable dependency (to mock)
    DOMAIN_TYPE = "domain_type" # DTO / entity / record in method signatures
    INTERFACE = "interface"     # Interface / abstract class
    RELATED = "related"         # Other related code
    SUMMARY = "summary"         # RAG summary (when no parsed source available)


# Priority tiers (lower = higher priority)
PRIORITY_MAP = {
    SnippetRole.SOURCE: 10,
    SnippetRole.DEPENDENCY: 20,
    SnippetRole.DOMAIN_TYPE: 30,
    SnippetRole.INTERFACE: 40,
    SnippetRole.RELATED: 50,
    SnippetRole.SUMMARY: 60,
}


@dataclass
class Snippet:
    """A code snippet selected for the prompt context."""

    content: str
    role: SnippetRole
    priority: int               # Lower = more important
    class_name: str = ""
    file_path: str = ""
    metadata: dict = field(default_factory=dict)

    def __lt__(self, other: Snippet) -> bool:
        return self.priority < other.priority


# ── Snippet Selector ─────────────────────────────────────────────────

class SnippetSelector:
    """Selects the minimal set of snippets for a prompt.

    Combines intelligence-layer analysis with RAG results:
      • Uses TestContext to know what to mock and what domain types to include
      • Falls back to RAG chunks when intelligence data is unavailable
    """

    def __init__(
        self,
        analyzer: Optional[DependencyAnalyzer] = None,
        symbols: Optional[SymbolMap] = None,
    ) -> None:
        self.analyzer = analyzer
        self.symbols = symbols

    def select(
        self,
        class_name: str,
        rag_chunks: list[CodeChunk],
        test_context: Optional[TestContext] = None,
        inline_source: Optional[str] = None,
    ) -> list[Snippet]:
        """Select and prioritise snippets for the prompt.

        Args:
            class_name: Target class to generate tests for.
            rag_chunks: RAG search results.
            test_context: (Optional) Pre-computed TestContext from DependencyAnalyzer.
            inline_source: (Optional) Source code pasted inline by the user.

        Returns:
            List of Snippet objects, sorted by priority (highest first).
        """
        snippets: list[Snippet] = []

        # 1. Target class source (highest priority)
        if inline_source:
            snippets.append(Snippet(
                content=inline_source,
                role=SnippetRole.SOURCE,
                priority=PRIORITY_MAP[SnippetRole.SOURCE],
                class_name=class_name,
                metadata={"origin": "inline"},
            ))

        # 2. Use intelligence layer if available
        if test_context:
            snippets.extend(self._from_test_context(test_context, rag_chunks))
        else:
            # Fallback to RAG-only selection
            snippets.extend(self._from_rag_chunks(class_name, rag_chunks))

        # Sort by priority
        snippets.sort()

        # Deduplicate by class name (keep highest priority)
        seen: set[str] = set()
        deduped: list[Snippet] = []
        for s in snippets:
            key = s.class_name or s.content[:50]
            if key not in seen:
                seen.add(key)
                deduped.append(s)

        logger.info(
            "Snippets selected",
            class_name=class_name,
            total=len(deduped),
            roles={r.value: sum(1 for s in deduped if s.role == r) for r in SnippetRole},
        )

        return deduped

    # ── Intelligence-based selection ─────────────────────────────────

    def _from_test_context(
        self, ctx: TestContext, rag_chunks: list[CodeChunk]
    ) -> list[Snippet]:
        """Select snippets using intelligence layer analysis."""
        snippets: list[Snippet] = []

        # Build a lookup from RAG chunks by class name for enrichment
        rag_by_class: dict[str, CodeChunk] = {
            c.class_name: c for c in rag_chunks
        }

        # Mock candidates → DEPENDENCY snippets
        for mock in ctx.mock_candidates:
            # 1. Try to get exact match from SymbolMap first (Highest Priority & Accuracy)
            entry = self.symbols.lookup(mock.field_type) if self.symbols else None
            if entry:
                summary = self._entry_to_summary(entry)
                snippets.append(Snippet(
                    content=summary,
                    role=SnippetRole.DEPENDENCY,
                    priority=PRIORITY_MAP[SnippetRole.DEPENDENCY],
                    class_name=mock.field_type,
                    file_path=entry.file_path,
                    metadata={
                        "field_name": mock.field_name,
                        "is_interface": mock.is_interface,
                        "origin": "symbol_map",
                    },
                ))
            else:
                # 2. Fallback to RAG chunk if AST parsing didn't find it (e.g., external library)
                chunk = rag_by_class.get(mock.field_type)
                if chunk:
                    snippets.append(Snippet(
                        content=chunk.summary,
                        role=SnippetRole.DEPENDENCY,
                        priority=PRIORITY_MAP[SnippetRole.DEPENDENCY],
                        class_name=mock.field_type,
                        file_path=chunk.file_path,
                        metadata={
                            "field_name": mock.field_name,
                            "is_interface": mock.is_interface,
                            "fqn": mock.fqn,
                            "origin": "rag",
                        },
                    ))

        # Related types → DOMAIN_TYPE snippets
        for related in ctx.related_types:
            # Prioritize SymbolMap over RAG for accurate domain types
            content = self._entry_to_summary(related)
            snippets.append(Snippet(
                content=content,
                role=SnippetRole.DOMAIN_TYPE,
                priority=PRIORITY_MAP[SnippetRole.DOMAIN_TYPE],
                class_name=related.name,
                file_path=related.file_path,
                metadata={
                    "java_type": related.java_type,
                    "has_builder": related.has_builder,
                    "has_data": related.has_data,
                    "origin": "symbol_map",
                },
            ))

        # Target class summary from RAG (if no inline source)
        # Often the target class source is injected in line 111, but if we need a summary:
        entry = self.symbols.lookup(ctx.target_class) if self.symbols else None
        if entry:
            snippets.append(Snippet(
                content=self._entry_to_summary(entry),
                role=SnippetRole.SUMMARY,
                priority=PRIORITY_MAP[SnippetRole.SOURCE] + 5,
                class_name=ctx.target_class,
                file_path=entry.file_path,
                metadata={"origin": "symbol_map"},
            ))
        else:
            target_chunk = rag_by_class.get(ctx.target_class)
            if target_chunk:
                snippets.append(Snippet(
                    content=target_chunk.summary,
                    role=SnippetRole.SUMMARY,
                    priority=PRIORITY_MAP[SnippetRole.SOURCE] + 5,  # Just below source
                    class_name=ctx.target_class,
                    file_path=target_chunk.file_path,
                    metadata={"origin": "rag"},
                ))

        return snippets

    # ── RAG-only fallback ────────────────────────────────────────────

    def _from_rag_chunks(
        self, class_name: str, rag_chunks: list[CodeChunk]
    ) -> list[Snippet]:
        """Fallback: build snippets from RAG chunks only (no intelligence layer)."""
        snippets: list[Snippet] = []

        _SERVICE_SUFFIXES = ("Service", "Repository", "Client", "Gateway", "Handler")
        _DOMAIN_TYPES = {"entity", "dto", "model", "record", "domain", "vo", "request", "response"}

        # Find the main class chunk
        main_chunk = None
        dep_names: set[str] = set()
        used_types: set[str] = set()

        for chunk in rag_chunks:
            if chunk.class_name == class_name:
                main_chunk = chunk
                dep_names = set(chunk.dependencies or [])
                used_types = set(chunk.used_types or [])
                break

        # Main class summary
        if main_chunk:
            snippets.append(Snippet(
                content=main_chunk.summary,
                role=SnippetRole.SUMMARY,
                priority=PRIORITY_MAP[SnippetRole.SOURCE] + 5,
                class_name=class_name,
                file_path=main_chunk.file_path,
                metadata={"origin": "rag_main"},
            ))

        for chunk in rag_chunks:
            if chunk.class_name == class_name:
                continue

            # Classify
            if chunk.class_name in dep_names:
                role = SnippetRole.DEPENDENCY
            elif chunk.class_name in used_types:
                role = SnippetRole.DOMAIN_TYPE
            elif any(chunk.class_name.endswith(s) for s in _SERVICE_SUFFIXES):
                role = SnippetRole.DEPENDENCY
            elif (
                (chunk.java_type in ("record", "class", "enum"))
                and (
                    (chunk.type or "").lower() in _DOMAIN_TYPES
                    or (chunk.layer or "").lower() in ("domain", "model", "dto")
                )
            ):
                role = SnippetRole.DOMAIN_TYPE
            elif chunk.java_type in ("interface",):
                role = SnippetRole.INTERFACE
            else:
                role = SnippetRole.RELATED

            snippets.append(Snippet(
                content=chunk.summary,
                role=role,
                priority=PRIORITY_MAP[role],
                class_name=chunk.class_name,
                file_path=chunk.file_path,
                metadata={
                    "origin": "rag",
                    "java_type": chunk.java_type or chunk.type,
                    "has_builder": getattr(chunk, "has_builder", False),
                    "has_data": getattr(chunk, "has_data", False),
                },
            ))

        return snippets

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _entry_to_summary(entry: SymbolEntry) -> str:
        """Generate a compact summary from a SymbolEntry (when no RAG chunk)."""
        lines = [f"{entry.java_type} {entry.name}"]
        if entry.annotations:
            lines[0] = " ".join(entry.annotations) + " " + lines[0]
        if entry.extends:
            lines[0] += f" extends {entry.extends}"
        if entry.implements:
            lines[0] += f" implements {', '.join(entry.implements)}"

        # Fields
        for f in entry.fields:
            ann = " ".join(f.annotations) + " " if f.annotations else ""
            lines.append(f"  {ann}{f.type} {f.name}")

        # Methods (signatures only)
        for m in entry.methods:
            params = ", ".join(f"{t} {n}" for t, n in m.parameters)
            ann = " ".join(m.annotations) + " " if m.annotations else ""
            lines.append(f"  {ann}{m.return_type} {m.name}({params})")

        return "\n".join(lines)
