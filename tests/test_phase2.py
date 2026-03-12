"""
Phase 2 Tests — Intelligence Layer + Context Builder

Tests the new components WITHOUT starting Qdrant/vLLM.
Uses isolated module loading to avoid heavy imports (torch, sentence_transformers).
"""

import sys
import types
import importlib.util
from pathlib import Path
from unittest.mock import MagicMock

# ── Bootstrap: avoid heavy imports ──────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Stub out heavy modules
for mod_name in [
    "structlog",
    "tree_sitter_java",
    "tree_sitter",
    "sentence_transformers",
    "qdrant_client",
    "qdrant_client.http",
    "qdrant_client.http.models",
    "torch",
]:
    if mod_name not in sys.modules:
        stub = types.ModuleType(mod_name)
        if mod_name == "structlog":
            stub.get_logger = lambda: MagicMock()
        if mod_name == "tree_sitter":
            stub.Language = MagicMock()
            stub.Parser = MagicMock()
        if mod_name == "tree_sitter_java":
            stub.language = MagicMock
        sys.modules[mod_name] = stub


# ── Helpers to load modules by file path ────────────────────────────

def _load(mod_name: str, file_path: Path):
    """Load a single .py file as a module."""
    spec = importlib.util.spec_from_file_location(mod_name, str(file_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def _load_pkg(pkg_name: str, init_path: Path):
    """Register a package (its __init__.py) — imports may fail, so we do submodules first."""
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [str(init_path.parent)]  # type: ignore
    pkg.__file__ = str(init_path)
    sys.modules[pkg_name] = pkg
    return pkg


# ── Load indexer (tree-sitter will be stubbed) ──────────────────────

_load_pkg("indexer", ROOT / "indexer" / "__init__.py")
# We need the dataclass definitions from parse_java but NOT the tree-sitter parse logic
# Load just the module — it may fail at Language(...) but we only need dataclasses

_parse_java_path = ROOT / "indexer" / "parse_java.py"
_parse_java_src = _parse_java_path.read_text(encoding="utf-8")

# Extract just the dataclass definitions (before the Parser class)
# We'll exec them into a module
_parse_java_mod = types.ModuleType("indexer.parse_java")
_parse_java_mod.__file__ = str(_parse_java_path)
sys.modules["indexer.parse_java"] = _parse_java_mod

# Inject needed imports into the module namespace
import os, re, dataclasses
from dataclasses import dataclass, field
from typing import Optional

_parse_java_mod.os = os
_parse_java_mod.re = re
_parse_java_mod.dataclass = dataclass
_parse_java_mod.field = field
_parse_java_mod.Optional = Optional
_parse_java_mod.structlog = sys.modules["structlog"]
_parse_java_mod.Path = Path

# Execute just the source to get dataclass definitions
try:
    exec(compile(_parse_java_src, str(_parse_java_path), "exec"), _parse_java_mod.__dict__)
except Exception:
    # If tree-sitter init fails, that's OK — we only need the dataclasses
    pass

# Verify we got the dataclasses we need
assert hasattr(_parse_java_mod, "ClassInfo"), "ClassInfo not loaded from parse_java"
assert hasattr(_parse_java_mod, "MethodInfo"), "MethodInfo not loaded from parse_java"
assert hasattr(_parse_java_mod, "FieldInfo"), "FieldInfo not loaded from parse_java"

ClassInfo = _parse_java_mod.ClassInfo
MethodInfo = _parse_java_mod.MethodInfo
FieldInfo = _parse_java_mod.FieldInfo

# ── Load RAG schema (lightweight) ───────────────────────────────────

_load_pkg("rag", ROOT / "rag" / "__init__.py")
_rag_schema = _load("rag.schema", ROOT / "rag" / "schema.py")

# Stub rag.client (needs sentence_transformers)
_rag_client_mod = types.ModuleType("rag.client")
_rag_client_mod.RAGClient = MagicMock
sys.modules["rag.client"] = _rag_client_mod

# ── Load intelligence package ───────────────────────────────────────

_load_pkg("intelligence", ROOT / "intelligence" / "__init__.py")
_repo_scanner = _load("intelligence.repo_scanner", ROOT / "intelligence" / "repo_scanner.py")
_file_graph = _load("intelligence.file_graph", ROOT / "intelligence" / "file_graph.py")
_symbol_map = _load("intelligence.symbol_map", ROOT / "intelligence" / "symbol_map.py")
_dep_analyzer = _load("intelligence.dependency_analyzer", ROOT / "intelligence" / "dependency_analyzer.py")

# Re-execute the __init__ to populate exports
_load("intelligence", ROOT / "intelligence" / "__init__.py")

RepoScanner = _repo_scanner.RepoScanner
RepoSnapshot = _repo_scanner.RepoSnapshot
FileGraph = _file_graph.FileGraph
FileNode = _file_graph.FileNode
SymbolMap = _symbol_map.SymbolMap
SymbolEntry = _symbol_map.SymbolEntry
DependencyAnalyzer = _dep_analyzer.DependencyAnalyzer
TestContext = _dep_analyzer.TestContext

# ── Load context package ────────────────────────────────────────────

_load_pkg("context", ROOT / "context" / "__init__.py")
_token_opt = _load("context.token_optimizer", ROOT / "context" / "token_optimizer.py")
_snippet_sel = _load("context.snippet_selector", ROOT / "context" / "snippet_selector.py")
_ctx_builder = _load("context.context_builder", ROOT / "context" / "context_builder.py")

# Re-execute the __init__
_load("context", ROOT / "context" / "__init__.py")

TokenOptimizer = _token_opt.TokenOptimizer
SnippetSelector = _snippet_sel.SnippetSelector
Snippet = _snippet_sel.Snippet
SnippetRole = _snippet_sel.SnippetRole
ContextBuilder = _ctx_builder.ContextBuilder
ContextResult = _ctx_builder.ContextResult


# ═══════════════════════════════════════════════════════════════════
# Test data factory
# ═══════════════════════════════════════════════════════════════════

def _make_class_info(
    name: str,
    package: str = "com.example",
    class_type: str = "class",
    annotations: list = None,
    imports: list = None,
    methods: list = None,
    detailed_fields: list = None,
    implements: list = None,
    extends: str = None,
    has_required_args_constructor: bool = False,
    has_builder: bool = False,
    has_data: bool = False,
    has_value: bool = False,
    file_path: str = "",
) -> ClassInfo:
    return ClassInfo(
        name=name,
        package=package,
        file_path=file_path or f"src/main/java/com/example/{name}.java",
        class_type=class_type,
        annotations=annotations or [],
        implements=implements or [],
        extends=extends,
        fields=[],
        methods=methods or [],
        imports=imports or [],
        detailed_fields=detailed_fields or [],
        has_required_args_constructor=has_required_args_constructor,
        has_builder=has_builder,
        has_data=has_data,
        has_value=has_value,
    )


def _make_method(name: str, return_type: str = "void", params: list = None) -> MethodInfo:
    return MethodInfo(
        name=name,
        return_type=return_type,
        parameters=params or [],
        body="",
        annotations=[],
    )


def _make_field(name: str, ftype: str, anns: list = None, mods: list = None) -> FieldInfo:
    return FieldInfo(
        name=name,
        type=ftype,
        annotations=anns or [],
        modifiers=mods or [],
    )


# ═══════════════════════════════════════════════════════════════════
# TESTS
# ═══════════════════════════════════════════════════════════════════

passed = 0
failed = 0


def _run(name, fn):
    global passed, failed
    try:
        fn()
        print(f"  ✓ {name}")
        passed += 1
    except Exception as e:
        print(f"  ✗ {name}: {e}")
        failed += 1


# ── 1. RepoSnapshot ─────────────────────────────────────────────────

print("\n=== RepoSnapshot ===")


def test_snapshot_index():
    classes = [
        _make_class_info("OrderService", annotations=["@Service"]),
        _make_class_info("OrderRepository", class_type="interface"),
        _make_class_info("OrderDto", class_type="record", has_builder=True),
    ]
    snap = RepoSnapshot(repo_path="/test", classes=classes)
    assert snap.get_class("OrderService").name == "OrderService"
    assert snap.get_by_fqn("com.example.OrderRepository").class_type == "interface"
    assert len(snap.list_packages()) == 1
    assert "com.example" in snap.list_packages()
    summary = snap.get_summary()
    assert summary["total_classes"] == 3


_run("snapshot indexes and queries", test_snapshot_index)


# ── 2. FileGraph ────────────────────────────────────────────────────

print("\n=== FileGraph ===")


def test_file_graph_build():
    classes = [
        _make_class_info(
            "OrderService",
            imports=["com.example.OrderRepository", "com.example.OrderDto"],
            file_path="OrderService.java",
        ),
        _make_class_info(
            "OrderRepository",
            class_type="interface",
            file_path="OrderRepository.java",
        ),
        _make_class_info(
            "OrderDto",
            class_type="record",
            file_path="OrderDto.java",
        ),
    ]
    snap = RepoSnapshot(repo_path="/test", classes=classes)
    graph = FileGraph.build(snap)

    # OrderService depends on OrderRepository and OrderDto
    deps = graph.dependencies_of("OrderService.java")
    assert "OrderRepository.java" in deps
    assert "OrderDto.java" in deps

    # OrderRepository is depended on by OrderService
    dependents = graph.dependents_of("OrderRepository.java")
    assert "OrderService.java" in dependents


_run("file graph build + queries", test_file_graph_build)


def test_file_graph_transitive():
    classes = [
        _make_class_info("A", imports=["com.example.B"], file_path="A.java"),
        _make_class_info("B", imports=["com.example.C"], file_path="B.java"),
        _make_class_info("C", file_path="C.java"),
    ]
    snap = RepoSnapshot(repo_path="/test", classes=classes)
    graph = FileGraph.build(snap)

    trans_deps = graph.transitive_dependencies("A.java", max_depth=3)
    assert "B.java" in trans_deps
    assert "C.java" in trans_deps


_run("transitive dependencies", test_file_graph_transitive)


# ── 3. SymbolMap ────────────────────────────────────────────────────

print("\n=== SymbolMap ===")


def test_symbol_map_build():
    classes = [
        _make_class_info(
            "OrderService",
            annotations=["@Service", "@RequiredArgsConstructor"],
            methods=[
                _make_method("createOrder", "OrderDto", [("OrderRequest", "request")]),
                _make_method("getOrder", "OrderDto", [("Long", "id")]),
            ],
            detailed_fields=[
                _make_field("orderRepo", "OrderRepository", mods=["private", "final"]),
                _make_field("mapper", "OrderMapper", mods=["private", "final"]),
            ],
            has_required_args_constructor=True,
        ),
        _make_class_info(
            "OrderRepository",
            class_type="interface",
            annotations=["@Repository"],
            methods=[_make_method("findById", "Optional<Order>", [("Long", "id")])],
        ),
        _make_class_info(
            "OrderDto",
            class_type="record",
            has_builder=True,
        ),
    ]
    snap = RepoSnapshot(repo_path="/test", classes=classes)
    symbols = SymbolMap.build(snap)

    # Lookup by name
    entry = symbols.lookup("OrderService")
    assert entry is not None
    assert entry.is_service
    assert len(entry.methods) == 2
    assert len(entry.fields) == 2

    # Injectable fields
    injectable = symbols.get_injectable_dependencies("OrderService")
    assert len(injectable) == 2
    assert injectable[0].type == "OrderRepository"
    assert injectable[1].type == "OrderMapper"

    # Reverse lookups
    assert "com.example.OrderService" in symbols.classes_with_method("createOrder")
    assert "com.example.OrderService" in symbols.classes_injecting("OrderRepository")

    # Classification
    assert len(symbols.services) == 1
    assert len(symbols.repositories) == 1

    # Method signatures
    sigs = symbols.get_method_signatures("OrderService")
    assert len(sigs) == 2
    assert "OrderDto createOrder(OrderRequest request)" in sigs


_run("symbol map build + queries", test_symbol_map_build)


# ── 4. DependencyAnalyzer ──────────────────────────────────────────

print("\n=== DependencyAnalyzer ===")


def test_dependency_analyzer_test_context():
    classes = [
        _make_class_info(
            "OrderService",
            annotations=["@Service", "@RequiredArgsConstructor"],
            imports=["com.example.OrderRepository", "com.example.OrderMapper"],
            methods=[
                _make_method("createOrder", "OrderDto", [("OrderRequest", "request")]),
            ],
            detailed_fields=[
                _make_field("orderRepo", "OrderRepository", mods=["private", "final"]),
                _make_field("mapper", "OrderMapper", mods=["private", "final"]),
            ],
            has_required_args_constructor=True,
            file_path="OrderService.java",
        ),
        _make_class_info(
            "OrderRepository",
            class_type="interface",
            annotations=["@Repository"],
            methods=[_make_method("save", "Order", [("Order", "entity")])],
            file_path="OrderRepository.java",
        ),
        _make_class_info(
            "OrderMapper",
            annotations=["@Component"],
            methods=[_make_method("toDto", "OrderDto", [("Order", "order")])],
            file_path="OrderMapper.java",
        ),
        _make_class_info(
            "OrderDto",
            class_type="record",
            has_builder=True,
            file_path="OrderDto.java",
        ),
        _make_class_info(
            "OrderRequest",
            class_type="record",
            file_path="OrderRequest.java",
        ),
    ]
    snap = RepoSnapshot(repo_path="/test", classes=classes)
    graph = FileGraph.build(snap)
    symbols = SymbolMap.build(snap)
    analyzer = DependencyAnalyzer(graph, symbols, snap)

    ctx = analyzer.test_context_for("OrderService")
    assert ctx is not None
    assert ctx.target_class == "OrderService"
    assert len(ctx.mock_candidates) == 2
    assert set(ctx.mock_types) == {"OrderRepository", "OrderMapper"}
    assert len(ctx.public_methods) == 1

    # Related types (DTOs/entities in method signatures)
    related_names = ctx.related_type_names
    # OrderDto and OrderRequest should be in the related types
    assert "OrderDto" in related_names or "OrderRequest" in related_names


_run("test context for OrderService", test_dependency_analyzer_test_context)


def test_dependency_analyzer_impact():
    classes = [
        _make_class_info("A", imports=["com.example.B"], file_path="A.java"),
        _make_class_info("B", imports=["com.example.C"], file_path="B.java"),
        _make_class_info("C", file_path="C.java"),
    ]
    snap = RepoSnapshot(repo_path="/test", classes=classes)
    graph = FileGraph.build(snap)
    symbols = SymbolMap.build(snap)
    analyzer = DependencyAnalyzer(graph, symbols, snap)

    impact = analyzer.impact_of("C")
    assert impact is not None
    assert "B" in impact.directly_affected
    assert "A" in impact.transitively_affected


_run("impact analysis", test_dependency_analyzer_impact)


def test_interface_implementations():
    classes = [
        _make_class_info("PaymentGateway", class_type="interface", file_path="PG.java"),
        _make_class_info("StripeGateway", implements=["PaymentGateway"], file_path="SG.java"),
        _make_class_info("PayPalGateway", implements=["PaymentGateway"], file_path="PPG.java"),
    ]
    snap = RepoSnapshot(repo_path="/test", classes=classes)
    graph = FileGraph.build(snap)
    symbols = SymbolMap.build(snap)
    analyzer = DependencyAnalyzer(graph, symbols, snap)

    impls = analyzer.find_interface_implementations("PaymentGateway")
    impl_names = [e.name for e in impls]
    assert "StripeGateway" in impl_names
    assert "PayPalGateway" in impl_names


_run("interface implementations", test_interface_implementations)


# ── 5. TokenOptimizer ──────────────────────────────────────────────

print("\n=== TokenOptimizer ===")


def test_token_optimizer_under_budget():
    snippets = [
        Snippet(content="short content", role=SnippetRole.SOURCE, priority=10),
        Snippet(content="another snippet", role=SnippetRole.DEPENDENCY, priority=20),
    ]
    optimizer = TokenOptimizer(token_budget=1000)
    result, report = optimizer.optimize(snippets)
    assert len(result) == 2
    assert report.snippets_dropped == 0
    assert not report.over_budget


_run("under budget — keep all", test_token_optimizer_under_budget)


def test_token_optimizer_over_budget():
    # Use estimate_tokens so assertions match the real tokenizer backend
    est = TokenOptimizer.estimate_tokens

    big_content = "public void process(Order o) { repo.save(o); }\n" * 30
    small_content = "private final OrderRepository repo;\n" * 5
    low_content = "// related helper\n" * 60

    big_snippet = Snippet(
        content=big_content,
        role=SnippetRole.SOURCE,
        priority=10,
        class_name="Main",
    )
    small_snippet = Snippet(
        content=small_content,
        role=SnippetRole.DEPENDENCY,
        priority=20,
        class_name="Dep",
    )
    low_prio = Snippet(
        content=low_content,
        role=SnippetRole.RELATED,
        priority=50,
        class_name="Related",
    )

    total = est(big_content) + est(small_content) + est(low_content)
    budget = est(big_content) + est(small_content) + 20  # leave only 20 tokens for third
    assert total > budget, f"Test data must exceed budget ({total} vs {budget})"

    optimizer = TokenOptimizer(token_budget=budget)
    result, report = optimizer.optimize([big_snippet, small_snippet, low_prio])

    # Should keep the first two and drop/truncate the third
    assert len(result) <= 3
    assert report.snippets_dropped + report.snippets_truncated > 0
    assert report.final_tokens <= budget


_run("over budget — drop/truncate", test_token_optimizer_over_budget)


# ── 6. SnippetSelector ─────────────────────────────────────────────

print("\n=== SnippetSelector ===")


def test_snippet_selector_from_rag():
    """Test RAG-only fallback (no intelligence layer)."""
    CodeChunk = _rag_schema.CodeChunk

    rag_chunks = [
        CodeChunk(
            id="1", summary="OrderService handles orders", score=0.9,
            type="service", layer="service", class_name="OrderService",
            package="com.example", file_path="OrderService.java",
            fully_qualified_name="com.example.OrderService",
            dependencies=["OrderRepository"], used_types=["OrderDto"],
        ),
        CodeChunk(
            id="2", summary="OrderRepository data access", score=0.8,
            type="repository", layer="repository", class_name="OrderRepository",
            package="com.example", file_path="OrderRepository.java",
            fully_qualified_name="com.example.OrderRepository",
        ),
        CodeChunk(
            id="3", summary="OrderDto record", score=0.7,
            type="dto", layer="domain", class_name="OrderDto",
            package="com.example", file_path="OrderDto.java",
            fully_qualified_name="com.example.OrderDto",
            java_type="record",
        ),
    ]

    selector = SnippetSelector()
    snippets = selector.select("OrderService", rag_chunks)

    assert len(snippets) >= 3
    # First should be the OrderService summary (highest priority)
    roles = [s.role for s in snippets]
    assert SnippetRole.DEPENDENCY in roles
    assert SnippetRole.DOMAIN_TYPE in roles


_run("snippet selector RAG-only", test_snippet_selector_from_rag)


def test_snippet_selector_with_intelligence():
    """Test intelligence-enhanced selection."""
    CodeChunk = _rag_schema.CodeChunk

    # Build intelligence layer
    classes = [
        _make_class_info(
            "OrderService", annotations=["@Service", "@RequiredArgsConstructor"],
            methods=[_make_method("create", "OrderDto", [("OrderRequest", "req")])],
            detailed_fields=[_make_field("repo", "OrderRepository", mods=["private", "final"])],
            has_required_args_constructor=True, file_path="OrderService.java",
            imports=["com.example.OrderRepository"],
        ),
        _make_class_info("OrderRepository", class_type="interface", file_path="OrderRepository.java"),
        _make_class_info("OrderDto", class_type="record", file_path="OrderDto.java"),
        _make_class_info("OrderRequest", class_type="record", file_path="OrderRequest.java"),
    ]
    snap = RepoSnapshot(repo_path="/test", classes=classes)
    graph = FileGraph.build(snap)
    symbols = SymbolMap.build(snap)
    analyzer = DependencyAnalyzer(graph, symbols, snap)

    test_context = analyzer.test_context_for("OrderService")

    rag_chunks = [
        CodeChunk(
            id="1", summary="OrderRepository interface", score=0.8,
            type="repository", layer="repository", class_name="OrderRepository",
            package="com.example", file_path="OrderRepository.java",
            fully_qualified_name="com.example.OrderRepository",
        ),
    ]

    selector = SnippetSelector(analyzer=analyzer, symbols=symbols)
    snippets = selector.select("OrderService", rag_chunks, test_context=test_context)

    # Should have at least dependency snippet (OrderRepository)
    dep_snippets = [s for s in snippets if s.role == SnippetRole.DEPENDENCY]
    assert len(dep_snippets) >= 1


_run("snippet selector with intelligence", test_snippet_selector_with_intelligence)


# ── 7. ContextBuilder (mocked RAG) ─────────────────────────────────

print("\n=== ContextBuilder ===")


def test_context_builder_rag_only():
    """Test ContextBuilder without intelligence (RAG-only mode)."""
    CodeChunk = _rag_schema.CodeChunk

    mock_rag = MagicMock()
    mock_search_result = MagicMock()
    mock_search_result.chunks = [
        CodeChunk(
            id="1", summary="OrderService handles orders", score=0.9,
            type="service", layer="service", class_name="OrderService",
            package="com.example", file_path="OrderService.java",
            fully_qualified_name="com.example.OrderService",
            dependencies=["OrderRepo"], used_types=["OrderDto"],
        ),
    ]
    mock_rag.search_by_class.return_value = mock_search_result

    builder = ContextBuilder(rag_client=mock_rag, token_budget=5000)
    result = builder.build_context("OrderService", file_path="OrderService.java")

    assert isinstance(result, ContextResult)
    assert result.class_name == "OrderService"
    assert len(result.rag_chunks) >= 1
    assert not result.intelligence_available


_run("context builder RAG-only", test_context_builder_rag_only)


# ═══════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════

print(f"\n{'='*50}")
print(f"Phase 2 tests: {passed} passed, {failed} failed")
print(f"{'='*50}")

if failed > 0:
    sys.exit(1)
