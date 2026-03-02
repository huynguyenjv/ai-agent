"""
Dependency Analyzer — merges FileGraph + SymbolMap for rich structural queries.

Provides high-level analysis capabilities:
  • What are all the dependencies needed to test class X?
  • Which classes will be affected if I change class Y?
  • What mocks does class X require for unit testing?
  • What's the full type context for building a test?

Usage::

    from intelligence.repo_scanner import RepoScanner
    from intelligence.file_graph import FileGraph
    from intelligence.symbol_map import SymbolMap
    from intelligence.dependency_analyzer import DependencyAnalyzer

    snapshot = RepoScanner().scan("/path/to/repo")
    graph = FileGraph.build(snapshot)
    symbols = SymbolMap.build(snapshot)
    analyzer = DependencyAnalyzer(graph, symbols, snapshot)

    ctx = analyzer.test_context_for("OrderService")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import structlog

from .file_graph import FileGraph
from .symbol_map import SymbolMap, SymbolEntry, FieldSymbol
from .repo_scanner import RepoSnapshot

logger = structlog.get_logger()


# ── Result data structures ───────────────────────────────────────────

@dataclass
class MockCandidate:
    """A dependency that should be mocked in a unit test."""

    field_name: str
    field_type: str
    fqn: Optional[str]          # Fully-qualified name (if resolved)
    is_interface: bool
    has_concrete_impl: bool     # True if we found a concrete implementation
    concrete_impls: list[str]   # FQNs of concrete implementations


@dataclass
class TestContext:
    """Everything needed to generate a unit test for a class."""

    target_class: str
    target_fqn: str
    target_entry: SymbolEntry

    # Dependencies → what to mock
    mock_candidates: list[MockCandidate]

    # Related types (DTOs, entities referenced in method signatures)
    related_types: list[SymbolEntry]

    # Method-level detail
    public_methods: list[str]  # method signatures

    # Structural metadata
    dependency_depth: int      # How many transitive deps
    total_related_files: int   # Files in the dependency cone

    @property
    def mock_types(self) -> list[str]:
        """Simple list of types to mock."""
        return [m.field_type for m in self.mock_candidates]

    @property
    def related_type_names(self) -> list[str]:
        return [e.name for e in self.related_types]


@dataclass
class ImpactReport:
    """Report on the impact of changing a class."""

    changed_class: str
    directly_affected: list[str]   # Classes that directly depend on it
    transitively_affected: list[str]  # All classes in the dependency cone
    test_files_to_update: list[str]   # Test files that may need updating


# ── Dependency Analyzer ──────────────────────────────────────────────

class DependencyAnalyzer:
    """High-level dependency analysis combining FileGraph + SymbolMap."""

    def __init__(
        self,
        graph: FileGraph,
        symbols: SymbolMap,
        snapshot: RepoSnapshot,
    ) -> None:
        self.graph = graph
        self.symbols = symbols
        self.snapshot = snapshot
        # Build FQN → simple-name reverse map for quick resolution
        self._interface_impls: dict[str, list[str]] = self._build_interface_impl_map()

    # ── Public API ───────────────────────────────────────────────────

    def test_context_for(self, class_name: str) -> Optional[TestContext]:
        """Build complete test context for a given class.

        Returns everything the prompt builder needs:
          - What to mock
          - What DTOs/entities are referenced
          - Method signatures
        """
        entry = self.symbols.lookup(class_name)
        if not entry:
            logger.warning("Class not found in symbol map", class_name=class_name)
            return None

        # 1. Find mock candidates (injectable dependencies)
        mocks = self._find_mock_candidates(entry)

        # 2. Find related types (DTOs, entities in method signatures)
        related = self._find_related_types(entry)

        # 3. Method signatures
        public_methods = self.symbols.get_method_signatures(class_name)

        # 4. Dependency depth
        file_path = entry.file_path
        transitive = self.graph.transitive_dependencies(file_path, max_depth=3)
        dep_depth = len(transitive)

        # 5. Count total related files
        dependents = self.graph.transitive_dependents(file_path, max_depth=2)
        total_related = len(transitive) + len(dependents)

        return TestContext(
            target_class=entry.name,
            target_fqn=entry.fqn,
            target_entry=entry,
            mock_candidates=mocks,
            related_types=related,
            public_methods=public_methods,
            dependency_depth=dep_depth,
            total_related_files=total_related,
        )

    def impact_of(self, class_name: str) -> Optional[ImpactReport]:
        """Analyze the impact of changing a class."""
        entry = self.symbols.lookup(class_name)
        if not entry:
            return None

        file_path = entry.file_path

        # Direct dependents
        direct_files = self.graph.dependents_of(file_path)
        direct_classes = self._files_to_class_names(direct_files)

        # Transitive dependents
        transitive_files = self.graph.transitive_dependents(file_path, max_depth=3)
        transitive_classes = self._files_to_class_names(transitive_files)

        # Test files (heuristic: files containing "Test" in their name)
        test_files = [f for f in transitive_files if "Test" in f or "test" in f]

        return ImpactReport(
            changed_class=class_name,
            directly_affected=direct_classes,
            transitively_affected=transitive_classes,
            test_files_to_update=test_files,
        )

    def dependencies_of(self, class_name: str) -> list[SymbolEntry]:
        """Return all resolved SymbolEntry objects that a class depends on."""
        entry = self.symbols.lookup(class_name)
        if not entry:
            return []

        dep_files = self.graph.dependencies_of(entry.file_path)
        entries = []
        for f in dep_files:
            for ci in self.snapshot.get_by_file(f):
                dep_entry = self.symbols.lookup_fqn(ci.fully_qualified_name)
                if dep_entry:
                    entries.append(dep_entry)
        return entries

    def find_interface_implementations(self, interface_name: str) -> list[SymbolEntry]:
        """Find all concrete implementations of an interface."""
        impls = self._interface_impls.get(interface_name, [])
        entries = []
        for fqn in impls:
            entry = self.symbols.lookup_fqn(fqn)
            if entry:
                entries.append(entry)
        return entries

    def get_mock_setup_hints(self, class_name: str) -> list[dict]:
        """Generate hints for setting up mocks in a test.

        Returns a list of dicts with:
          - field_name: the field to mock
          - type: the type to mock
          - methods: methods available on the mocked type
          - is_interface: whether it's an interface
        """
        entry = self.symbols.lookup(class_name)
        if not entry:
            return []

        hints = []
        for f in entry.fields:
            if not f.is_injectable:
                continue

            dep_entry = self.symbols.lookup(f.type)
            methods = []
            if dep_entry:
                methods = self.symbols.get_method_signatures(f.type)

            hints.append({
                "field_name": f.name,
                "type": f.type,
                "methods": methods,
                "is_interface": dep_entry.is_interface if dep_entry else False,
            })
        return hints

    # ── Internal helpers ─────────────────────────────────────────────

    def _find_mock_candidates(self, entry: SymbolEntry) -> list[MockCandidate]:
        """Find all dependencies that need to be mocked for testing."""
        candidates = []

        for f in entry.fields:
            if not f.is_injectable:
                continue

            dep_entry = self.symbols.lookup(f.type)
            is_interface = dep_entry.is_interface if dep_entry else False
            fqn = dep_entry.fqn if dep_entry else None

            # Find concrete implementations
            concrete_impls = []
            has_concrete = False
            if is_interface:
                impls = self._interface_impls.get(f.type, [])
                concrete_impls = impls
                has_concrete = len(impls) > 0
            else:
                has_concrete = dep_entry is not None

            candidates.append(
                MockCandidate(
                    field_name=f.name,
                    field_type=f.type,
                    fqn=fqn,
                    is_interface=is_interface,
                    has_concrete_impl=has_concrete,
                    concrete_impls=concrete_impls,
                )
            )
        return candidates

    def _find_related_types(self, entry: SymbolEntry) -> list[SymbolEntry]:
        """Find DTOs, entities, and other types referenced in method signatures."""
        seen: set[str] = set()
        related: list[SymbolEntry] = []

        # Collect all type names from method signatures
        type_names: set[str] = set()
        for m in entry.methods:
            # Return type
            if m.return_type and m.return_type[0].isupper():
                type_names.add(self._strip_generics(m.return_type))
            # Parameters
            for ptype, _ in m.parameters:
                if ptype[0].isupper():
                    type_names.add(self._strip_generics(ptype))

        # Also check extends/implements
        if entry.extends:
            type_names.add(entry.extends)
        type_names.update(entry.implements)

        # Resolve to SymbolEntry
        for tn in type_names:
            if tn == entry.name or tn in seen:
                continue
            # Skip java.lang built-ins
            if tn in ("String", "Integer", "Long", "Double", "Float", "Boolean",
                       "Object", "Void", "List", "Set", "Map", "Optional",
                       "Collection", "Iterable", "Stream", "Page", "Pageable",
                       "Sort", "Specification", "Predicate", "ResponseEntity",
                       "HttpStatus", "UUID", "BigDecimal", "LocalDate",
                       "LocalDateTime", "Instant"):
                continue
            seen.add(tn)
            dep_entry = self.symbols.lookup(tn)
            if dep_entry:
                related.append(dep_entry)

        return related

    def _files_to_class_names(self, files: list[str]) -> list[str]:
        """Convert file paths to class names."""
        names = []
        for f in files:
            classes = self.snapshot.get_by_file(f)
            for ci in classes:
                names.append(ci.name)
        return names

    def _build_interface_impl_map(self) -> dict[str, list[str]]:
        """Build a map from interface name → list of implementing class FQNs."""
        impl_map: dict[str, list[str]] = {}

        for entry_list in self.symbols._by_name.values():
            for entry in entry_list:
                if entry.is_interface:
                    continue
                for iface in entry.implements:
                    if iface not in impl_map:
                        impl_map[iface] = []
                    impl_map[iface].append(entry.fqn)
                # Also check extends (abstract class)
                if entry.extends:
                    if entry.extends not in impl_map:
                        impl_map[entry.extends] = []
                    impl_map[entry.extends].append(entry.fqn)

        return impl_map

    @staticmethod
    def _strip_generics(type_name: str) -> str:
        """Strip generic type parameters: List<Order> → List, Optional<OrderDto> → OrderDto."""
        if "<" not in type_name:
            return type_name
        # For wrapper types, return the inner type
        wrapper_types = {"List", "Set", "Optional", "Collection", "Iterable",
                         "Page", "Slice", "Stream", "CompletableFuture",
                         "ResponseEntity", "Mono", "Flux"}
        outer = type_name[:type_name.index("<")]
        if outer in wrapper_types:
            inner = type_name[type_name.index("<") + 1:type_name.rindex(">")]
            # Handle nested generics
            if "<" in inner:
                return DependencyAnalyzer._strip_generics(inner)
            return inner.strip()
        return outer
