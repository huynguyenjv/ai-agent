"""
Context Builder — orchestrates context assembly for LLM prompts.

This is the main entry point for Phase 2 context logic.  It coordinates:

    1. Intelligence layer (RepoScanner → FileGraph → SymbolMap → DependencyAnalyzer)
    2. Snippet selection (SnippetSelector)
    3. Token optimization (TokenOptimizer)

The result is a ``ContextResult`` that can be consumed by the PromptBuilder
or wired into the Orchestrator pipeline.

Usage::

    builder = ContextBuilder(rag_client=rag, repo_path="/path/to/repo")
    result = builder.build_context("OrderService", file_path="...", inline_source="...")
    # result.snippets   — prioritised & budget-trimmed
    # result.rag_chunks — original RAG chunks (for backward compat)
    # result.test_context — intelligence analysis (if available)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import structlog

from rag.client import RAGClient
from rag.schema import CodeChunk

from .snippet_selector import SnippetSelector, Snippet
from .token_optimizer import TokenOptimizer, BudgetReport

logger = structlog.get_logger()

# Optional intelligence imports — graceful degradation
try:
    from intelligence.repo_scanner import RepoScanner, RepoSnapshot
    from intelligence.file_graph import FileGraph
    from intelligence.symbol_map import SymbolMap
    from intelligence.dependency_analyzer import DependencyAnalyzer, TestContext

    _INTELLIGENCE_AVAILABLE = True
except ImportError:
    _INTELLIGENCE_AVAILABLE = False
    RepoScanner = RepoSnapshot = FileGraph = SymbolMap = DependencyAnalyzer = TestContext = None  # type: ignore


# ── Result ───────────────────────────────────────────────────────────

@dataclass
class ContextResult:
    """Output of context building — everything the prompt builder needs."""

    # Core context
    snippets: list[Snippet]
    rag_chunks: list[CodeChunk]

    # Intelligence analysis (may be None if repo not scanned)
    test_context: Optional[object] = None  # TestContext when available

    # Token budget report
    budget_report: Optional[BudgetReport] = None

    # Timing
    elapsed_ms: float = 0.0

    # Metadata
    intelligence_available: bool = False
    class_name: str = ""

    @property
    def mock_types(self) -> list[str]:
        """Quick access to types that need @Mock."""
        if self.test_context and hasattr(self.test_context, "mock_types"):
            return self.test_context.mock_types
        return []

    @property
    def related_types(self) -> list[str]:
        """Quick access to related domain type names."""
        if self.test_context and hasattr(self.test_context, "related_type_names"):
            return self.test_context.related_type_names
        return []


# ── Context Builder ──────────────────────────────────────────────────

class ContextBuilder:
    """Orchestrates context assembly for LLM prompts.

    Combines RAG search with (optional) intelligence layer analysis
    to produce a minimal, high-quality context for test generation.
    """

    def __init__(
        self,
        rag_client: RAGClient,
        repo_path: Optional[str] = None,
        token_budget: int = 6000,
    ) -> None:
        self.rag = rag_client
        self.repo_path = repo_path
        self.token_budget = token_budget

        # Intelligence layer (lazy-initialized)
        self._snapshot: Optional[object] = None
        self._graph: Optional[object] = None
        self._symbols: Optional[object] = None
        self._analyzer: Optional[object] = None
        self._intelligence_ready = False

        # Reusable components (avoid re-creation per call)
        self._selector = SnippetSelector()
        self._optimizer = TokenOptimizer(token_budget=token_budget)

    # ── Intelligence initialization ──────────────────────────────────

    def init_intelligence(self, repo_path: Optional[str] = None) -> bool:
        """Initialize the intelligence layer by scanning the repo.

        This is expensive (~seconds) so it should be called once at startup
        or lazily on first use.  Returns True if successful.
        """
        if not _INTELLIGENCE_AVAILABLE:
            logger.warning("Intelligence layer not available (import failed)")
            return False

        path = repo_path or self.repo_path
        if not path:
            logger.warning("No repo path provided for intelligence layer")
            return False

        # Update instance path if a new one was provided
        if repo_path:
            self.repo_path = repo_path

        try:
            start = time.time()
            scanner = RepoScanner()
            self._snapshot = scanner.scan(path)
            self._graph = FileGraph.build(self._snapshot)
            self._symbols = SymbolMap.build(self._snapshot)
            self._analyzer = DependencyAnalyzer(self._graph, self._symbols, self._snapshot)
            self._intelligence_ready = True

            # Update selector with intelligence components
            self._selector = SnippetSelector(
                analyzer=self._analyzer,
                symbols=self._symbols,
            )

            elapsed = (time.time() - start) * 1000
            summary = self._snapshot.get_summary()
            logger.info(
                "Intelligence layer initialized",
                classes=summary.get("total_classes", 0),
                files=summary.get("total_files", 0),
                elapsed_ms=round(elapsed, 1),
            )
            return True
        except Exception as e:
            logger.error("Failed to initialize intelligence layer", error=str(e))
            self._intelligence_ready = False
            return False

    # ── Main API ─────────────────────────────────────────────────────

    async def abuild_context(
        self,
        class_name: str,
        file_path: str = "",
        inline_source: Optional[str] = None,
        session=None,
        top_k: int = 10,
        collection_name: Optional[str] = None,
    ) -> ContextResult:
        """Async version of build_context()."""
        start = time.time()

        # Step 1: RAG search (always needed for summaries)
        rag_chunks = await self._arag_search(class_name, file_path, session, inline_source, top_k, collection_name=collection_name)

        # Step 2: Intelligence analysis (if available)
        test_context = None
        if self._intelligence_ready and self._analyzer:
            # Intelligence layer is currently sync, but fast enough as it's in-memory
            test_context = self._analyzer.test_context_for(class_name)

        # Step 3: Snippet selection (reuse instance)
        snippets = self._selector.select(
            class_name=class_name,
            rag_chunks=rag_chunks,
            test_context=test_context,
            inline_source=inline_source,
        )

        # Step 4: Token optimization (reuse instance)
        optimized_snippets, budget_report = self._optimizer.optimize(snippets)

        elapsed = (time.time() - start) * 1000

        logger.info(
            "Async context built",
            class_name=class_name,
            rag_chunks=len(rag_chunks),
            snippets=len(optimized_snippets),
            intelligence=self._intelligence_ready,
            elapsed_ms=round(elapsed, 1),
        )

        return ContextResult(
            snippets=optimized_snippets,
            rag_chunks=rag_chunks,
            test_context=test_context,
            budget_report=budget_report,
            elapsed_ms=elapsed,
            intelligence_available=self._intelligence_ready,
            class_name=class_name,
        )

    def build_context(
        self,
        class_name: str,
        file_path: str = "",
        inline_source: Optional[str] = None,
        session=None,
        top_k: int = 10,
        collection_name: Optional[str] = None,
    ) -> ContextResult:
        """Build optimized context for test generation.

        Args:
            class_name: Name of the target class (e.g. "OrderService").
            file_path: Source file path.
            inline_source: Source code pasted inline.
            session: Optional SessionMemory for conversation context.
            top_k: RAG search depth.

        Returns:
            ContextResult with assembled snippets + metadata.
        """
        start = time.time()

        # Step 1: RAG search (always needed for summaries)
        rag_chunks = self._rag_search(class_name, file_path, session, inline_source, top_k, collection_name=collection_name)

        # Step 2: Intelligence analysis (if available)
        test_context = None
        if self._intelligence_ready and self._analyzer:
            test_context = self._analyzer.test_context_for(class_name)

        # Step 3: Snippet selection (reuse instance)
        snippets = self._selector.select(
            class_name=class_name,
            rag_chunks=rag_chunks,
            test_context=test_context,
            inline_source=inline_source,
        )

        # Step 4: Token optimization (reuse instance)
        optimized_snippets, budget_report = self._optimizer.optimize(snippets)

        elapsed = (time.time() - start) * 1000

        logger.info(
            "Context built",
            class_name=class_name,
            rag_chunks=len(rag_chunks),
            snippets=len(optimized_snippets),
            intelligence=self._intelligence_ready,
            elapsed_ms=round(elapsed, 1),
        )

        return ContextResult(
            snippets=optimized_snippets,
            rag_chunks=rag_chunks,
            test_context=test_context,
            budget_report=budget_report,
            elapsed_ms=elapsed,
            intelligence_available=self._intelligence_ready,
            class_name=class_name,
        )

    def build_refinement_context(
        self,
        class_name: str,
        rag_chunks: list[CodeChunk],
    ) -> ContextResult:
        """Build context for refinement (uses cached RAG chunks)."""
        start = time.time()

        test_context = None
        if self._intelligence_ready and self._analyzer:
            test_context = self._analyzer.test_context_for(class_name)

        snippets = self._selector.select(
            class_name=class_name,
            rag_chunks=rag_chunks,
            test_context=test_context,
        )

        optimized_snippets, budget_report = self._optimizer.optimize(snippets)

        elapsed = (time.time() - start) * 1000

        return ContextResult(
            snippets=optimized_snippets,
            rag_chunks=rag_chunks,
            test_context=test_context,
            budget_report=budget_report,
            elapsed_ms=elapsed,
            intelligence_available=self._intelligence_ready,
            class_name=class_name,
        )

    # ── Properties ───────────────────────────────────────────────────

    @property
    def intelligence_ready(self) -> bool:
        return self._intelligence_ready

    @property
    def snapshot(self) -> Optional[object]:
        return self._snapshot

    @property
    def symbols(self) -> Optional[object]:
        return self._symbols

    # ── Internal ─────────────────────────────────────────────────────

    async def _arag_search(
        self,
        class_name: str,
        file_path: str,
        session,
        inline_source: Optional[str],
        top_k: int,
        collection_name: Optional[str] = None,
    ) -> list[CodeChunk]:
        """Perform RAG search for a class and its dependencies."""
        try:
            result = await self.rag.asearch_by_class(
                class_name=class_name,
                top_k=top_k,
                include_dependencies=True,
                collection_name=collection_name,
            )
            return result.chunks
        except Exception as e:
            logger.error("Async RAG search failed", class_name=class_name, error=str(e))
            return []

    def _rag_search(
        self,
        class_name: str,
        file_path: str,
        session,
        inline_source: Optional[str],
        top_k: int,
        collection_name: Optional[str] = None,
    ) -> list[CodeChunk]:
        """Perform RAG search for a class and its dependencies."""
        try:
            result = self.rag.search_by_class(
                class_name=class_name,
                top_k=top_k,
                include_dependencies=True,
                collection_name=collection_name,
            )
            return result.chunks
        except Exception as e:
            logger.error("RAG search failed", class_name=class_name, error=str(e))
            return []
