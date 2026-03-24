"""Comprehensive tests for the new architecture (Phases 1-7)."""

import hashlib
import os
import shutil
import tempfile

# ============ Phase 1: Models, HashStore, Tools ============


def test_codechunk_model():
    from mcp_server.models import CodeChunk, ChunkType

    cid = CodeChunk.make_chunk_id("repo", "src/Main.java", "Main")
    assert len(cid) == 32
    assert cid == CodeChunk.make_chunk_id("repo", "src/Main.java", "Main")
    assert cid != CodeChunk.make_chunk_id("repo", "src/Main.java", "Other")

    chunk = CodeChunk(
        chunk_id="abc", chunk_type=ChunkType.callable, symbol_name="foo",
        embed_text="sig", body="body", file_path="a.py", lang="python",
        start_line=1, end_line=10, file_hash="h", raw_imports=["import os"],
    )
    payload = chunk.to_payload()
    assert "raw_imports" not in payload  # Section 4.4
    assert payload["chunk_type"] == "callable"


def test_hash_store():
    from mcp_server.hash_store import HashStore

    db = os.path.join(tempfile.gettempdir(), "test_hs.db")
    try:
        hs = HashStore(db)
        assert hs.get_hash("test.py") is None
        hs.set_hash("test.py", "abc123")
        assert hs.get_hash("test.py") == "abc123"
        hs.set_hashes_batch([("a.py", "h1"), ("b.py", "h2")])
        assert hs.get_hash("a.py") == "h1"
        hs.remove("test.py")
        assert hs.get_hash("test.py") is None
        hs.close()
    finally:
        if os.path.exists(db):
            os.unlink(db)


def test_read_file():
    from mcp_server.tools import read_file

    repo = tempfile.mkdtemp()
    try:
        with open(os.path.join(repo, "test.py"), "w") as f:
            f.write("line1\nline2\nline3\nline4\nline5\n")

        result = read_file(repo, "test.py", 2, 4)
        assert result["start_line"] == 2
        assert result["end_line"] == 4
        assert "line2" in result["content"]
        assert "line1" not in result["content"]

        result = read_file(repo, "nonexistent.py")
        assert "error" in result

        result = read_file(repo, "../../../etc/passwd")
        assert "error" in result
    finally:
        shutil.rmtree(repo)


def test_search_symbol():
    from mcp_server.tools import search_symbol
    from mcp_server.plugins.registry import PluginRegistry
    from mcp_server.plugins.fallback import FallbackPlugin

    registry = PluginRegistry()
    registry.register(FallbackPlugin())

    repo = tempfile.mkdtemp()
    try:
        with open(os.path.join(repo, "service.py"), "w") as f:
            f.write("class UserService:\n    def get_user(self):\n        pass\n")

        matches = search_symbol(repo, registry, "UserService")
        assert len(matches) >= 1
        assert matches[0]["symbol_name"] == "UserService"
    finally:
        shutil.rmtree(repo)


# ============ Phase 2: Language Plugins ============


def test_fallback_plugin():
    from mcp_server.plugins.fallback import FallbackPlugin
    from mcp_server.models import ExtractionMode

    fp = FallbackPlugin()
    assert fp.extensions() == []
    chunks = fp.extract_chunks(
        "t.py", b"class Foo:\n  pass\ndef bar():\n  pass\n", ExtractionMode.full_body
    )
    assert len(chunks) >= 2


def test_python_plugin():
    from mcp_server.plugins.python_plugin import PythonPlugin
    from mcp_server.models import ExtractionMode

    plugin = PythonPlugin()
    source = b"import os\nclass A:\n  def m(self): pass\ndef f(): pass\n"
    chunks = plugin.extract_chunks("s.py", source, ExtractionMode.full_body)
    names = [c.symbol_name for c in chunks]
    assert "A" in names
    assert "A.m" in names
    assert "f" in names

    # Method should have raw_imports
    method = [c for c in chunks if c.symbol_name == "A.m"][0]
    assert len(method.raw_imports) > 0

    # Signatures mode: body should be empty
    sig_chunks = plugin.extract_chunks("s.py", source, ExtractionMode.signatures)
    for c in sig_chunks:
        assert c.body == ""


def test_java_plugin():
    from mcp_server.plugins.java_plugin import JavaPlugin
    from mcp_server.models import ExtractionMode

    plugin = JavaPlugin()
    source = b"import java.util.List;\npublic class M { public void run() {} }"
    chunks = plugin.extract_chunks("M.java", source, ExtractionMode.full_body)
    names = [c.symbol_name for c in chunks]
    assert "M" in names
    assert any("run" in n for n in names)

    cls = [c for c in chunks if c.symbol_name == "M"][0]
    assert "java.util.List" in cls.raw_imports


def test_go_plugin():
    from mcp_server.plugins.go_plugin import GoPlugin
    from mcp_server.models import ExtractionMode

    plugin = GoPlugin()
    source = b'package main\nimport "fmt"\ntype S struct{}\nfunc F() {}\nfunc (s *S) M() {}'
    chunks = plugin.extract_chunks("m.go", source, ExtractionMode.full_body)
    names = [c.symbol_name for c in chunks]
    assert "S" in names
    assert "F" in names
    assert "S.M" in names


def test_typescript_plugin():
    from mcp_server.plugins.typescript_plugin import TypeScriptPlugin
    from mcp_server.models import ExtractionMode

    plugin = TypeScriptPlugin()
    source = b"export class A { m() {} }\nexport function f() {}\nconst h = () => {};\n"
    chunks = plugin.extract_chunks("a.ts", source, ExtractionMode.full_body)
    names = [c.symbol_name for c in chunks]
    assert "A" in names
    assert "A.m" in names
    assert "f" in names
    assert "h" in names


def test_csharp_plugin():
    from mcp_server.plugins.csharp_plugin import CSharpPlugin
    from mcp_server.models import ExtractionMode

    plugin = CSharpPlugin()
    source = b"namespace N { public class A { public void M() {} } }"
    chunks = plugin.extract_chunks("A.cs", source, ExtractionMode.full_body)
    names = [c.symbol_name for c in chunks]
    assert any("A" in n for n in names)
    assert any("M" in n for n in names)


def test_hcl_plugin():
    from mcp_server.plugins.hcl_plugin import HCLPlugin
    from mcp_server.models import ExtractionMode

    plugin = HCLPlugin()
    source = b'resource "aws_s3_bucket" "main" {\n  bucket = "b"\n}\nvariable "r" {\n  default = "x"\n}\n'
    chunks = plugin.extract_chunks("m.tf", source, ExtractionMode.full_body)
    names = [c.symbol_name for c in chunks]
    assert "aws_s3_bucket.main" in names
    assert "var.r" in names


def test_plugin_registry():
    from mcp_server.plugins.registry import PluginRegistry
    from mcp_server.plugins.python_plugin import PythonPlugin
    from mcp_server.plugins.java_plugin import JavaPlugin
    from mcp_server.plugins.go_plugin import GoPlugin
    from mcp_server.plugins.typescript_plugin import TypeScriptPlugin
    from mcp_server.plugins.csharp_plugin import CSharpPlugin
    from mcp_server.plugins.hcl_plugin import HCLPlugin
    from mcp_server.plugins.fallback import FallbackPlugin

    reg = PluginRegistry()
    reg.register(PythonPlugin())
    reg.register(JavaPlugin())
    reg.register(GoPlugin())
    reg.register(TypeScriptPlugin())
    reg.register(CSharpPlugin())
    reg.register(HCLPlugin())
    reg.register(FallbackPlugin())

    assert isinstance(reg.get_plugin("a.py"), PythonPlugin)
    assert isinstance(reg.get_plugin("a.java"), JavaPlugin)
    assert isinstance(reg.get_plugin("a.go"), GoPlugin)
    assert isinstance(reg.get_plugin("a.ts"), TypeScriptPlugin)
    assert isinstance(reg.get_plugin("a.cs"), CSharpPlugin)
    assert isinstance(reg.get_plugin("a.tf"), HCLPlugin)
    assert isinstance(reg.get_plugin("a.xyz"), FallbackPlugin)


def test_dep_classifier():
    from mcp_server.dep_classifier import DepClassifier, DepType
    from mcp_server.plugins.fallback import FallbackPlugin

    dc = DepClassifier()
    fb = FallbackPlugin()

    # Java stdlib
    t, _ = dc.classify("java.util.List", "java", fb, "F.java", "/r")
    assert t == DepType.stdlib

    # Go stdlib
    t, _ = dc.classify("fmt", "go", fb, "main.go", "/r")
    assert t == DepType.stdlib

    # Python stdlib
    t, _ = dc.classify("import os", "python", fb, "main.py", "/r")
    assert t == DepType.stdlib

    # TypeScript stdlib
    t, _ = dc.classify("node:fs", "typescript", fb, "app.ts", "/r")
    assert t == DepType.stdlib

    # TypeScript third_party
    t, _ = dc.classify("express", "typescript", fb, "app.ts", "/r")
    assert t == DepType.third_party

    # C# stdlib
    t, _ = dc.classify("using System.Collections.Generic;", "csharp", fb, "F.cs", "/r")
    assert t == DepType.stdlib


# ============ Phase 3: TokenBudget, Skeleton ============


def test_token_budget():
    from mcp_server.token_budget import TokenBudget
    from mcp_server.models import ExtractionMode

    tb = TokenBudget(8000)
    assert tb.get_mode(0) == ExtractionMode.full_body
    assert tb.get_mode(1) == ExtractionMode.signatures
    assert tb.get_mode(2) == ExtractionMode.names_only
    assert tb.get_budget(0) == 2000
    assert tb.get_budget(1) == 3000


def test_get_project_skeleton():
    from mcp_server.tools_indexer import get_project_skeleton
    from mcp_server.plugins.registry import PluginRegistry
    from mcp_server.plugins.python_plugin import PythonPlugin
    from mcp_server.plugins.fallback import FallbackPlugin

    registry = PluginRegistry()
    registry.register(PythonPlugin())
    registry.register(FallbackPlugin())

    repo = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(repo, "src"))
        with open(os.path.join(repo, "src", "app.py"), "w") as f:
            f.write("class App:\n    def run(self): pass\n")
        with open(os.path.join(repo, "src", "util.py"), "w") as f:
            f.write("def helper(): pass\n")

        result = get_project_skeleton(repo, registry, include_methods=True)
        assert result["stats"]["total_classes"] >= 1
        assert result["stats"]["total_packages"] >= 1
    finally:
        shutil.rmtree(repo)


# ============ Phase 5: Agent Nodes ============


def test_classify_intent():
    from server.agent.classify_intent import classify_intent

    cases = [
        ("viết unit test cho UserService", "unit_test"),
        ("write test for OrderService", "unit_test"),
        ("phân tích cấu trúc project này", "structural_analysis"),
        ("analyze architecture", "structural_analysis"),
        ("giải thích function này", "explain"),
        ("explain how does this work", "explain"),
        ("tìm class UserService", "search"),
        ("sửa lại function này", "refine"),
        ("refactor this code", "refine"),
        ("create API endpoint", "code_gen"),
    ]
    for content, expected in cases:
        state = {"messages": [type("M", (), {"content": content})()]}
        result = classify_intent(state)
        assert result["intent"] == expected, f"{content!r}: expected {expected}, got {result['intent']}"


def test_route_context():
    from server.agent.route_context import route_context

    # Gate 1: file mention
    state = {"messages": [type("M", (), {"content": "explain UserService.java"})()], "active_file": None}
    r = route_context(state)
    assert r["force_reindex"] is True
    assert len(r["mentioned_files"]) >= 1

    # Gate 1: deictic
    state = {"messages": [type("M", (), {"content": "explain this file"})()], "active_file": "Main.java"}
    r = route_context(state)
    assert r["force_reindex"] is True

    # Gate 2: freshness (use query without file name pattern to avoid Gate 1)
    state = {"messages": [type("M", (), {"content": "vừa sửa code xong"})()], "active_file": None}
    r = route_context(state)
    assert r["force_reindex"] is True
    assert r["freshness_signal"] is True

    # Gate 3: volatile data
    state = {"messages": [type("M", (), {"content": "show me the git diff"})()], "active_file": None}
    r = route_context(state)
    assert r["volatile_rejected"] is True
    assert r["force_reindex"] is False

    # No gate
    state = {"messages": [type("M", (), {"content": "how does DI work"})()], "active_file": None}
    r = route_context(state)
    assert r["force_reindex"] is False


def test_post_process():
    from server.agent.post_process import post_process

    r = post_process({"intent": "unit_test", "draft": "@Test void test() {}"})
    assert "@Test" in r["draft"]

    r = post_process({"intent": "code_gen", "draft": "def foo():\n    pass"})
    assert "def foo" in r["draft"]


# ============ Phase 6: SSE Events ============


def test_sse_events():
    import json
    from server.streaming.sse import (
        thinking_event, content_delta_event, done_event,
        heartbeat_comment, tool_start_event, tool_done_event,
        tool_error_event,
    )

    # Thinking events are SSE comments (invisible to Continue)
    assert thinking_event("test").startswith(": thinking:")

    # Content deltas are OpenAI format (Continue renders these)
    d = json.loads(content_delta_event("hi").split("data: ")[1])
    assert d["choices"][0]["delta"]["content"] == "hi"

    # Tool events are SSE comments
    assert tool_start_event("idx", "Indexing").startswith(": tool_start:")
    assert tool_done_event("idx", "done").startswith(": tool_done:")

    # Tool errors are content deltas (user must see errors)
    err = tool_error_event("gen", "timeout")
    d = json.loads(err.split("data: ")[1])
    assert "timeout" in d["choices"][0]["delta"]["content"]

    assert done_event() == "data: [DONE]\n\n"
    assert heartbeat_comment() == ": keep-alive\n\n"


def test_continue_compat():
    from server.continue_compat import extract_active_file

    # Extract from code fence
    msg = type("M", (), {"content": "Explain this\n\n```java src/main/java/UserService.java\npublic class UserService {}\n```"})()
    assert extract_active_file([msg]) is not None

    # Extract from plain code fence with file path
    msg2 = type("M", (), {"content": "Fix this\n\n```src/OrderService.java\nclass OrderService {}\n```"})()
    result = extract_active_file([msg2])
    assert result is not None
    assert "OrderService.java" in result

    # Explicit active_file takes priority
    assert extract_active_file([msg], "explicit.py") == "explicit.py"

    # No file info
    msg3 = type("M", (), {"content": "Hello world"})()
    assert extract_active_file([msg3]) is None


def test_agent_state_new_fields():
    """AgentState must declare tool call flow fields."""
    from server.agent.state import AgentState
    import typing

    hints = typing.get_type_hints(AgentState)
    assert "pending_tool_calls" in hints
    assert "is_tool_result_turn" in hints
    assert "tool_turns_used" in hints


import pytest

@pytest.mark.asyncio
async def test_count_by_file_returns_zero_when_empty():
    """count_by_file returns 0 when no chunks exist for a path."""
    from unittest.mock import AsyncMock, MagicMock
    from server.rag.qdrant_client import QdrantService

    svc = QdrantService.__new__(QdrantService)
    mock_client = AsyncMock()
    mock_result = MagicMock()
    mock_result.count = 0
    mock_client.count = AsyncMock(return_value=mock_result)
    svc._client = mock_client
    svc._collection = "codebase"

    result = await svc.count_by_file("src/UserService.java")
    assert result == 0
    mock_client.count.assert_called_once()


@pytest.mark.asyncio
async def test_count_by_file_returns_count_when_exists():
    """count_by_file returns > 0 when chunks exist."""
    from unittest.mock import AsyncMock, MagicMock
    from server.rag.qdrant_client import QdrantService

    svc = QdrantService.__new__(QdrantService)
    mock_client = AsyncMock()
    mock_result = MagicMock()
    mock_result.count = 5
    mock_client.count = AsyncMock(return_value=mock_result)
    svc._client = mock_client
    svc._collection = "codebase"

    result = await svc.count_by_file("src/UserService.java")
    assert result == 5


@pytest.mark.asyncio
async def test_count_by_file_returns_zero_on_exception():
    """count_by_file returns 0 (miss) on any Qdrant exception."""
    from unittest.mock import AsyncMock
    from server.rag.qdrant_client import QdrantService

    svc = QdrantService.__new__(QdrantService)
    mock_client = AsyncMock()
    mock_client.count = AsyncMock(side_effect=Exception("connection refused"))
    svc._client = mock_client
    svc._collection = "codebase"

    result = await svc.count_by_file("src/UserService.java")
    assert result == 0
