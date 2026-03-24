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


def test_tool_calls_event_format():
    """tool_calls_event must emit two SSE data lines per OpenAI spec."""
    import json
    from server.streaming.sse import tool_calls_event

    tool_calls = [
        {"index": 0, "id": "call_1", "type": "function",
         "function": {"name": "index_with_deps", "arguments": '{"file_path": "src/A.java"}'}},
    ]
    result = tool_calls_event(tool_calls)

    # Must have exactly two data lines
    data_lines = [l for l in result.split("\n") if l.startswith("data: ")]
    assert len(data_lines) == 2, f"Expected 2 data lines, got {len(data_lines)}"

    # Chunk 1: has tool_calls, finish_reason null
    chunk1 = json.loads(data_lines[0][6:])
    assert chunk1["choices"][0]["delta"]["tool_calls"] == tool_calls
    assert chunk1["choices"][0]["finish_reason"] is None

    # Chunk 2: empty delta, finish_reason "tool_calls"
    chunk2 = json.loads(data_lines[1][6:])
    assert chunk2["choices"][0]["delta"] == {}
    assert chunk2["choices"][0]["finish_reason"] == "tool_calls"


def test_tool_result_turn_detection():
    """classify_intent sets is_tool_result_turn=True when ToolMessage present."""
    from langchain_core.messages import ToolMessage, HumanMessage
    from server.agent.classify_intent import classify_intent

    # Turn 2: messages contain a ToolMessage
    messages = [
        HumanMessage(content="generate tests for UserService.java"),
        ToolMessage(content='{"files": ["UserService.java"]}', tool_call_id="call_1"),
    ]
    state = {"messages": messages, "intent": "unit_test"}
    result = classify_intent(state)

    assert result["is_tool_result_turn"] is True
    # Must preserve prior intent, not re-classify from ToolMessage JSON content
    assert result["intent"] == "unit_test"


def test_classify_intent_not_tool_result_turn_on_normal_message():
    """classify_intent sets is_tool_result_turn=False for normal user messages."""
    from langchain_core.messages import HumanMessage
    from server.agent.classify_intent import classify_intent

    messages = [HumanMessage(content="viết test cho UserService")]
    state = {"messages": messages, "intent": ""}
    result = classify_intent(state)

    assert result.get("is_tool_result_turn", False) is False
    assert result["intent"] == "unit_test"


import pytest

@pytest.mark.asyncio
async def test_tool_selector_structural_analysis():
    """structural_analysis intent always emits get_project_skeleton."""
    from unittest.mock import AsyncMock
    from server.agent.tool_selector import tool_selector

    mock_qdrant = AsyncMock()
    mock_qdrant.count_by_file = AsyncMock(return_value=0)

    state = {
        "intent": "structural_analysis",
        "mentioned_files": [],
        "active_file": None,
        "freshness_signal": False,
        "is_tool_result_turn": False,
        "tool_turns_used": 0,
        "messages": [],
    }
    result = await tool_selector(state, qdrant=mock_qdrant)

    assert len(result["pending_tool_calls"]) == 1
    assert result["pending_tool_calls"][0]["function"]["name"] == "get_project_skeleton"
    assert result["tool_turns_used"] == 1


@pytest.mark.asyncio
async def test_tool_selector_search():
    """search intent always emits search_symbol regardless of Qdrant state."""
    from unittest.mock import AsyncMock
    from server.agent.tool_selector import tool_selector

    mock_qdrant = AsyncMock()
    state = {
        "intent": "search",
        "mentioned_files": [],
        "active_file": None,
        "freshness_signal": False,
        "is_tool_result_turn": False,
        "tool_turns_used": 0,
        "messages": [type("M", (), {"content": "find UserService class"})()],
    }
    result = await tool_selector(state, qdrant=mock_qdrant)

    assert len(result["pending_tool_calls"]) == 1
    assert result["pending_tool_calls"][0]["function"]["name"] == "search_symbol"
    args = result["pending_tool_calls"][0]["function"]["arguments"]
    import json
    assert "UserService" in json.loads(args)["name"]


@pytest.mark.asyncio
async def test_tool_selector_code_gen_miss():
    """code_gen + Qdrant miss + file mentioned -> [index_with_deps, read_file]."""
    from unittest.mock import AsyncMock
    from server.agent.tool_selector import tool_selector

    mock_qdrant = AsyncMock()
    mock_qdrant.count_by_file = AsyncMock(return_value=0)  # miss

    state = {
        "intent": "code_gen",
        "mentioned_files": ["src/UserService.java"],
        "active_file": None,
        "freshness_signal": False,
        "is_tool_result_turn": False,
        "tool_turns_used": 0,
        "messages": [],
    }
    result = await tool_selector(state, qdrant=mock_qdrant)

    names = [c["function"]["name"] for c in result["pending_tool_calls"]]
    assert "index_with_deps" in names
    assert "read_file" in names
    assert result["tool_turns_used"] == 1


@pytest.mark.asyncio
async def test_tool_selector_code_gen_hit_no_freshness():
    """code_gen + hit + file mentioned, no freshness -> [read_file] only."""
    from unittest.mock import AsyncMock
    from server.agent.tool_selector import tool_selector

    mock_qdrant = AsyncMock()
    mock_qdrant.count_by_file = AsyncMock(return_value=10)  # hit

    state = {
        "intent": "code_gen",
        "mentioned_files": ["src/UserService.java"],
        "active_file": None,
        "freshness_signal": False,
        "is_tool_result_turn": False,
        "tool_turns_used": 0,
        "messages": [],
    }
    result = await tool_selector(state, qdrant=mock_qdrant)

    names = [c["function"]["name"] for c in result["pending_tool_calls"]]
    assert names == ["read_file"]


@pytest.mark.asyncio
async def test_tool_selector_cap():
    """tool_turns_used >= 1 always returns empty pending_tool_calls."""
    from unittest.mock import AsyncMock
    from server.agent.tool_selector import tool_selector

    mock_qdrant = AsyncMock()
    state = {
        "intent": "structural_analysis",  # would normally emit tool
        "mentioned_files": [],
        "active_file": None,
        "freshness_signal": False,
        "is_tool_result_turn": False,
        "tool_turns_used": 1,  # already used
        "messages": [],
    }
    result = await tool_selector(state, qdrant=mock_qdrant)

    assert result["pending_tool_calls"] == []
    assert result["tool_turns_used"] == 1  # not incremented again


@pytest.mark.asyncio
async def test_emit_tool_calls_sse_format():
    """emit_tool_calls streams thinking comment + two-chunk tool_calls + DONE."""
    import json
    from server.agent.emit_tool_calls import emit_tool_calls

    emitted = []

    async def mock_sse_callback(event_type: str, content: str) -> None:
        emitted.append((event_type, content))

    tool_calls = [
        {"index": 0, "id": "call_1", "type": "function",
         "function": {"name": "index_with_deps", "arguments": '{"file_path": "A.java"}'}},
    ]
    state = {"pending_tool_calls": tool_calls}
    result = await emit_tool_calls(state, sse_callback=mock_sse_callback)

    # Node returns empty dict (pure side-effect)
    assert result == {}

    # Must have emitted thinking + tool_calls events
    event_types = [e[0] for e in emitted]
    assert "thinking" in event_types
    assert "tool_calls" in event_types

    # tool_calls content is JSON-serialized tool call list
    tc_events = [e for e in emitted if e[0] == "tool_calls"]
    assert len(tc_events) == 1
    parsed = json.loads(tc_events[0][1])
    assert parsed[0]["function"]["name"] == "index_with_deps"


def test_chat_message_accepts_null_content():
    """ChatMessage must accept content=None for Turn 2 assistant messages."""
    from server.routers.chat import ChatMessage

    # Turn 2 assistant message (tool_calls response)
    msg = ChatMessage(role="assistant", content=None, tool_calls=[
        {"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}
    ])
    assert msg.content is None
    assert msg.tool_calls is not None

    # Turn 2 tool result message
    msg2 = ChatMessage(role="tool", content='{"result": "ok"}', tool_call_id="call_1")
    assert msg2.tool_call_id == "call_1"


@pytest.mark.asyncio
async def test_graph_routes_to_emit_tool_calls():
    """Graph routes to emit_tool_calls when pending_tool_calls is non-empty."""
    from unittest.mock import AsyncMock, patch
    from server.agent.graph import build_agent_graph
    from langchain_core.messages import HumanMessage

    mock_qdrant = AsyncMock()
    mock_qdrant.count_by_file = AsyncMock(return_value=0)

    emitted_events = []

    async def mock_sse_callback(event_type, content):
        emitted_events.append(event_type)

    mock_vllm = AsyncMock()

    with patch("server.agent.tool_selector.tool_selector") as mock_selector:
        mock_selector.return_value = {
            "pending_tool_calls": [{"index": 0, "id": "c1", "type": "function",
                                     "function": {"name": "get_project_skeleton", "arguments": "{}"}}],
            "tool_turns_used": 1,
        }

        agent = build_agent_graph(
            vllm_client=mock_vllm,
            model="test",
            qdrant=mock_qdrant,
            embedder=None,
            sse_callback=mock_sse_callback,
        )

        result = await agent.ainvoke({
            "messages": [HumanMessage(content="analyze architecture")],
            "intent": "", "active_file": None, "mentioned_files": [], "freshness_signal": False,
            "force_reindex": False, "rag_chunks": [], "rag_hit": False, "hash_verified": False,
            "tool_results": [], "context_assembled": "", "draft": "", "emitted_steps": [],
            "volatile_rejected": False, "pending_tool_calls": [], "is_tool_result_turn": False,
            "tool_turns_used": 0,
        })

    assert "tool_calls" in emitted_events


@pytest.mark.asyncio
async def test_graph_skips_to_rag_search():
    """Graph completes without emitting tool_calls when pending_tool_calls is empty."""
    from unittest.mock import AsyncMock, patch
    from server.agent.graph import build_agent_graph
    from langchain_core.messages import HumanMessage

    mock_qdrant = AsyncMock()
    mock_qdrant.count_by_file = AsyncMock(return_value=10)

    emitted_events = []

    async def mock_sse_callback(event_type, content):
        emitted_events.append(event_type)

    mock_vllm = AsyncMock()

    with patch("server.agent.tool_selector.tool_selector") as mock_selector:
        mock_selector.return_value = {"pending_tool_calls": [], "tool_turns_used": 0}

        agent = build_agent_graph(
            vllm_client=mock_vllm,
            model="test",
            qdrant=mock_qdrant,
            embedder=None,
            sse_callback=mock_sse_callback,
        )

        result = await agent.ainvoke({
            "messages": [HumanMessage(content="explain this code")],
            "intent": "", "active_file": None, "mentioned_files": [], "freshness_signal": False,
            "force_reindex": False, "rag_chunks": [], "rag_hit": False, "hash_verified": False,
            "tool_results": [], "context_assembled": "", "draft": "", "emitted_steps": [],
            "volatile_rejected": False, "pending_tool_calls": [], "is_tool_result_turn": False,
            "tool_turns_used": 0,
        })

    assert "tool_calls" not in emitted_events
    assert isinstance(result, dict)


@pytest.mark.asyncio
async def test_graph_turn2_skips_route_context():
    """Turn 2 (is_tool_result_turn=True) skips route_context, goes classify_intent -> tool_selector -> rag_search."""
    from unittest.mock import AsyncMock, patch
    from server.agent.graph import build_agent_graph
    from langchain_core.messages import HumanMessage, ToolMessage

    mock_qdrant = AsyncMock()
    mock_qdrant.count_by_file = AsyncMock(return_value=5)

    async def mock_sse_callback(event_type, content):
        pass

    mock_vllm = AsyncMock()

    with patch("server.agent.route_context.route_context") as mock_rc, \
         patch("server.agent.tool_selector.tool_selector") as mock_ts:
        mock_rc.return_value = {
            "mentioned_files": [], "force_reindex": False, "freshness_signal": False,
        }
        mock_ts.return_value = {"pending_tool_calls": [], "tool_turns_used": 0}

        agent = build_agent_graph(
            vllm_client=mock_vllm,
            model="test",
            qdrant=mock_qdrant,
            embedder=None,
            sse_callback=mock_sse_callback,
        )

        result = await agent.ainvoke({
            "messages": [
                HumanMessage(content="generate tests for UserService"),
                ToolMessage(content='{"chunks": 5}', tool_call_id="call_1"),
            ],
            "intent": "unit_test",
            "active_file": None, "mentioned_files": [], "freshness_signal": False,
            "force_reindex": False, "rag_chunks": [], "rag_hit": False, "hash_verified": False,
            "tool_results": [], "context_assembled": "", "draft": "", "emitted_steps": [],
            "volatile_rejected": False, "pending_tool_calls": [], "is_tool_result_turn": False,
            "tool_turns_used": 0,
        })

    # route_context should NOT have been called on Turn 2
    mock_rc.assert_not_called()
    # tool_selector should have been called (and returned empty due to is_tool_result_turn)
    mock_ts.assert_called_once()
