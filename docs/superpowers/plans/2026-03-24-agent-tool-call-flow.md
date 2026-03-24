# Agent Tool Call Flow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable the LangGraph agent to emit OpenAI-format `tool_calls` to Continue IDE, which relays them to local MCP tools, then sends tool results back for RAG-enriched code generation.

**Architecture:** Agent Python logic (not LLM) deterministically selects tools from intent+gate+Qdrant cache state. Turn 1: agent emits `tool_calls` as SSE → Continue calls MCP tools → MCP uploads chunks to `/index`. Turn 2: Continue sends `role:"tool"` results → agent generates with fresh Qdrant data. Max 1 tool round-trip per request via `tool_turns_used` cap.

**Tech Stack:** FastAPI, LangGraph `StateGraph`, `qdrant-client`, `langchain-core` (`ToolMessage`), Python 3.10+, pytest + pytest-asyncio>=0.23 (already added to `requirements.txt`)

**Spec:** `docs/superpowers/specs/2026-03-24-agent-tool-call-flow-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `server/agent/state.py` | Modify | Add 3 new `AgentState` fields |
| `server/agent/classify_intent.py` | Modify | Detect `is_tool_result_turn` as first check |
| `server/agent/tool_selector.py` | **Create** | Deterministic tool selection logic |
| `server/agent/emit_tool_calls.py` | **Create** | SSE `tool_calls` emitter node |
| `server/agent/graph.py` | Modify | Wire new nodes + edges |
| `server/routers/chat.py` | Modify | `ChatMessage` schema, message loop, `sse_callback`, `initial_state` |
| `server/streaming/sse.py` | Modify | Add `tool_calls_event()` (two-chunk format) |
| `server/rag/qdrant_client.py` | Modify | Add `count_by_file()` + `file_path` payload index |
| `tests/test_new_arch.py` | Modify | Add 10 new test cases for tool call flow |

---

## Task 1: Install `pytest-asyncio` + AgentState — Add 3 New Fields

**Files:**
- Modify: `requirements.txt` (already done — `pytest-asyncio>=0.23` added)
- Modify: `server/agent/state.py`
- Test: `tests/test_new_arch.py`

This is the foundation. All other tasks depend on these fields existing in the `TypedDict`. The `pytest-asyncio` package is required for all async tests in Tasks 2, 5, 6, 8 — without it, async tests silently pass without running.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_new_arch.py`:

```python
def test_agent_state_new_fields():
    """AgentState must declare tool call flow fields."""
    from server.agent.state import AgentState
    import typing

    hints = typing.get_type_hints(AgentState)
    assert "pending_tool_calls" in hints
    assert "is_tool_result_turn" in hints
    assert "tool_turns_used" in hints
```

- [ ] **Step 1.5: Install pytest-asyncio**

```bash
pip install pytest-asyncio>=0.23
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd c:/Users/huynmb/IdeaProjects/source/train/tabby-pipeline/ai-agent
python -m pytest tests/test_new_arch.py::test_agent_state_new_fields -v
```
Expected: FAIL — `AssertionError: assert 'pending_tool_calls' in hints`

- [ ] **Step 3: Add fields to `server/agent/state.py`**

After line 29 (`volatile_rejected: bool`), add:

```python
    pending_tool_calls: list[dict]   # tools to emit, set by tool_selector
    is_tool_result_turn: bool        # True when request contains role:"tool" messages
    tool_turns_used: int             # capped at 1, prevents > 2 round-trips
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_new_arch.py::test_agent_state_new_fields -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add server/agent/state.py tests/test_new_arch.py
git commit -m "feat: add tool call flow fields to AgentState"
```

---

## Task 2: `count_by_file()` in QdrantService + `file_path` Payload Index

**Files:**
- Modify: `server/rag/qdrant_client.py`
- Test: `tests/test_new_arch.py`

`tool_selector` calls this to determine Qdrant cache hit/miss before emitting `index_with_deps`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_new_arch.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_new_arch.py::test_count_by_file_returns_zero_when_empty -v
```
Expected: FAIL — `AttributeError: 'QdrantService' object has no attribute 'count_by_file'`

- [ ] **Step 3: Add `count_by_file()` and `file_path` index to `server/rag/qdrant_client.py`**

Add import at top (after existing model imports):
```python
from qdrant_client.models import (
    ...existing imports...,
    PayloadSchemaType,
)
```

First, add `self._collection = COLLECTION_NAME` to `__init__` (after `self._client = AsyncQdrantClient(url=url)`):
```python
    def __init__(self, url: str = "http://127.0.0.1:6333") -> None:
        self._client = AsyncQdrantClient(url=url)
        self._collection = COLLECTION_NAME
```

Add `file_path` index inside `ensure_collection()`, after the existing `chunk_type` index (line 64):
```python
            await self._client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="file_path",
                field_schema=PayloadSchemaType.KEYWORD,
            )
```

Add `count_by_file()` method after `delete_points()` (before `hybrid_search`):
```python
    async def count_by_file(self, file_path: str) -> int:
        """Return number of stored chunks for a given file_path. 0 = miss."""
        try:
            result = await self._client.count(
                collection_name=self._collection,
                count_filter=Filter(
                    must=[FieldCondition(
                        key="file_path",
                        match=MatchValue(value=file_path),
                    )]
                ),
                exact=False,
            )
            return result.count
        except Exception:
            return 0
```

> Note: `COLLECTION_NAME` is already defined as a module-level constant. `Filter`, `FieldCondition`, `MatchValue` are already imported.

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_new_arch.py::test_count_by_file_returns_zero_when_empty tests/test_new_arch.py::test_count_by_file_returns_count_when_exists tests/test_new_arch.py::test_count_by_file_returns_zero_on_exception -v
```
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add server/rag/qdrant_client.py tests/test_new_arch.py
git commit -m "feat: add count_by_file() and file_path payload index to QdrantService"
```

---

## Task 3: `tool_calls_event()` SSE Formatter

**Files:**
- Modify: `server/streaming/sse.py`
- Test: `tests/test_new_arch.py`

Two-chunk OpenAI format: delta chunk (`finish_reason: null`) + empty delta chunk (`finish_reason: "tool_calls"`).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_new_arch.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_new_arch.py::test_tool_calls_event_format -v
```
Expected: FAIL — `ImportError` or `AssertionError`

- [ ] **Step 3: Add `tool_calls_event()` to `server/streaming/sse.py`**

Add after `done_event()` (before `heartbeat_comment`):

```python
def tool_calls_event(tool_calls: list[dict], chunk_id: str = "chatcmpl-agent") -> str:
    """OpenAI tool_calls delta format for Continue.

    Emits two chunks per OpenAI streaming spec:
      Chunk 1: delta with tool_calls list, finish_reason null
      Chunk 2: empty delta, finish_reason "tool_calls"

    Continue accumulates arguments from Chunk 1 before processing Chunk 2.
    """
    chunk1 = sse_event({
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "choices": [{
            "index": 0,
            "delta": {"tool_calls": tool_calls},
            "finish_reason": None,
        }],
    })
    chunk2 = sse_event({
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "tool_calls",
        }],
    })
    return chunk1 + chunk2
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_new_arch.py::test_tool_calls_event_format -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add server/streaming/sse.py tests/test_new_arch.py
git commit -m "feat: add tool_calls_event() SSE formatter (two-chunk OpenAI format)"
```

---

## Task 4: `classify_intent` — Detect `is_tool_result_turn`

**Files:**
- Modify: `server/agent/classify_intent.py`
- Test: `tests/test_new_arch.py`

This check must run **before** all pattern matching. When `role:"tool"` messages are present, preserve the prior intent and set `is_tool_result_turn=True`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_new_arch.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_new_arch.py::test_tool_result_turn_detection tests/test_new_arch.py::test_classify_intent_not_tool_result_turn_on_normal_message -v
```
Expected: FAIL — `KeyError: 'is_tool_result_turn'` or `AssertionError`

- [ ] **Step 3: Update `server/agent/classify_intent.py`**

Add import at top:
```python
from langchain_core.messages import ToolMessage
```

Replace the `classify_intent` function body — insert the tool result detection **before** the existing pattern loop:

```python
def classify_intent(state: AgentState) -> dict:
    """Classify the user's intent from the last message.

    MUST check is_tool_result_turn first — before any pattern matching.
    Default: code_gen if none of the patterns match.
    """
    messages = state.get("messages", [])
    if not messages:
        return {"intent": "code_gen", "is_tool_result_turn": False}

    # FIRST CHECK: detect Turn 2 (tool result messages present)
    has_tool_result = any(
        isinstance(m, ToolMessage) or
        (hasattr(m, "type") and m.type == "tool") or
        (isinstance(m, dict) and m.get("role") == "tool")
        for m in messages
    )
    if has_tool_result:
        # Preserve intent from prior state; do not re-classify
        return {
            "intent": state.get("intent", "code_gen"),
            "is_tool_result_turn": True,
        }

    last_msg = messages[-1]
    if hasattr(last_msg, "content"):
        text = last_msg.content
    elif isinstance(last_msg, dict):
        text = last_msg.get("content", "")
    else:
        text = str(last_msg)

    text_lower = text.lower()

    for intent, patterns in INTENT_PATTERNS:
        for pattern in patterns:
            if pattern.search(text_lower):
                return {"intent": intent, "is_tool_result_turn": False}

    return {"intent": "code_gen", "is_tool_result_turn": False}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_new_arch.py::test_tool_result_turn_detection tests/test_new_arch.py::test_classify_intent_not_tool_result_turn_on_normal_message -v
```
Expected: PASS

Also verify all existing tests still pass (regression check):
```bash
python -m pytest tests/test_new_arch.py -v
```
Expected: all PASS (no regressions)

- [ ] **Step 5: Commit**

```bash
git add server/agent/classify_intent.py tests/test_new_arch.py
git commit -m "feat: detect is_tool_result_turn in classify_intent (Turn 2 detection)"
```

---

## Task 5: `tool_selector` Node

**Files:**
- Create: `server/agent/tool_selector.py`
- Test: `tests/test_new_arch.py`

Deterministic tool selection table from spec. Inputs: intent, gates, active_file, Qdrant cache state, tool_turns_used. Output: `pending_tool_calls` list + incremented `tool_turns_used`.

> **Signature note:** `tool_selector(state, qdrant=None)` — `qdrant` is a positional-or-keyword argument (not keyword-only). This is required for `partial(tool_selector_node, qdrant=qdrant)` in Task 8's graph wiring to work correctly. Tests call it as `tool_selector(state, qdrant=mock_qdrant)` — both patterns work with this signature.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_new_arch.py`:

```python
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
    """code_gen + Qdrant miss + file mentioned → [index_with_deps, read_file]."""
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
    """code_gen + hit + file mentioned, no freshness → [read_file] only."""
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_new_arch.py::test_tool_selector_structural_analysis -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'server.agent.tool_selector'`

- [ ] **Step 3: Create `server/agent/tool_selector.py`**

```python
"""Node: tool_selector — Agent Tool Call Flow.

Deterministically selects MCP tools based on intent, gate outputs, and Qdrant cache state.
Agent logic decides tools — not the LLM.
"""

from __future__ import annotations

import json
import re
import uuid

from server.agent.state import AgentState

# Regex to extract a symbol name from the user's message for search intent
_SYMBOL_PATTERN = re.compile(r"\b([A-Z][a-zA-Z0-9]+|`([^`]+)`|\"([^\"]+)\")\b")


def _make_tool_call(name: str, arguments: dict, index: int = 0) -> dict:
    """Build an OpenAI-format tool call dict."""
    return {
        "index": index,
        "id": f"call_{uuid.uuid4().hex[:8]}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments),
        },
    }


def _extract_symbol_name(messages: list) -> str:
    """Extract a symbol name from the last user message for search_symbol."""
    if not messages:
        return "unknown"
    last = messages[-1]
    text = last.content if hasattr(last, "content") else str(last)
    match = _SYMBOL_PATTERN.search(text)
    if match:
        return match.group(2) or match.group(3) or match.group(1)
    return "unknown"


async def tool_selector(state: AgentState, qdrant=None) -> dict:
    """Select MCP tools based on intent + gate outputs + Qdrant cache state.

    Returns:
        pending_tool_calls: list of OpenAI tool call dicts (empty = no tools needed)
        tool_turns_used: incremented if tools are emitted
    """
    # Cap: never emit tools more than once per request
    if state.get("tool_turns_used", 0) >= 1 or state.get("is_tool_result_turn", False):
        return {
            "pending_tool_calls": [],
            "tool_turns_used": state.get("tool_turns_used", 0),
        }

    intent = state.get("intent", "code_gen")
    mentioned_files: list[str] = state.get("mentioned_files", [])
    active_file: str | None = state.get("active_file")
    freshness_signal: bool = state.get("freshness_signal", False)
    messages = state.get("messages", [])

    tool_calls: list[dict] = []

    if intent == "structural_analysis":
        tool_calls = [_make_tool_call("get_project_skeleton", {"include_methods": True})]

    elif intent == "search":
        name = _extract_symbol_name(messages)
        tool_calls = [_make_tool_call("search_symbol", {"name": name, "type_filter": "any"})]

    elif intent in ("code_gen", "refine", "unit_test"):
        if mentioned_files:
            file = mentioned_files[0]
            # Check Qdrant cache
            count = await qdrant.count_by_file(file) if qdrant else 0
            cache_hit = count > 0

            if not cache_hit or freshness_signal:
                # Miss or stale: index + read
                tool_calls = [
                    _make_tool_call("index_with_deps", {"file_path": file, "depth": 2}, index=0),
                    _make_tool_call("read_file", {"file_path": file}, index=1),
                ]
            else:
                # Hit, no freshness: trust cache, just read
                tool_calls = [_make_tool_call("read_file", {"file_path": file})]

        elif active_file:
            count = await qdrant.count_by_file(active_file) if qdrant else 0
            if count == 0:
                tool_calls = [_make_tool_call("index_with_deps", {"file_path": active_file, "depth": 2})]
            # else: hit + no freshness + no explicit file → RAG sufficient, no tools

    elif intent == "explain":
        if mentioned_files:
            tool_calls = [_make_tool_call("read_file", {"file_path": mentioned_files[0]})]
        # else: RAG sufficient

    if not tool_calls:
        return {
            "pending_tool_calls": [],
            "tool_turns_used": state.get("tool_turns_used", 0),
        }

    return {
        "pending_tool_calls": tool_calls,
        "tool_turns_used": state.get("tool_turns_used", 0) + 1,
    }
```

- [ ] **Step 4: Run all tool_selector tests**

```bash
python -m pytest tests/test_new_arch.py -k "tool_selector" -v
```
Expected: all 5 PASS

- [ ] **Step 5: Commit**

```bash
git add server/agent/tool_selector.py tests/test_new_arch.py
git commit -m "feat: add tool_selector node with deterministic tool selection"
```

---

## Task 6: `emit_tool_calls` Node

**Files:**
- Create: `server/agent/emit_tool_calls.py`
- Test: `tests/test_new_arch.py`

Pure SSE side-effect node. Streams thinking comments → tool_calls chunks → `[DONE]`. Connected to `END` — must handle its own exceptions internally.

> **Interface contract:** `sse_callback(event_type: str, content: str) -> None` is an async callable with exactly 2 positional string args. This matches the closure defined in `chat.py` Task 7 (Change 4) and is used consistently in both `emit_tool_calls` and `generate` nodes. Tests mock it as `async def mock_sse_callback(event_type, content)` — same signature.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_new_arch.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_new_arch.py::test_emit_tool_calls_sse_format -v
```
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Create `server/agent/emit_tool_calls.py`**

```python
"""Node: emit_tool_calls — Agent Tool Call Flow.

Pure SSE side-effect node. Emits thinking comments and OpenAI tool_calls chunks.
Connected to END — handles exceptions internally (no graph fallback exists).
"""

from __future__ import annotations

import json
import logging

from server.agent.state import AgentState

logger = logging.getLogger("server.emit_tool_calls")


async def emit_tool_calls(state: AgentState, sse_callback=None) -> dict:
    """Emit SSE events for pending_tool_calls.

    Sequence:
    1. thinking comment (immediate, < 50ms)
    2. tool_start comment
    3. tool_calls event (two-chunk OpenAI format via sse_callback)
    4. done event

    Returns {} — no state changes, pure side-effect node.
    """
    if sse_callback is None:
        return {}

    tool_calls = state.get("pending_tool_calls", [])
    if not tool_calls:
        return {}

    tool_names = ", ".join(
        tc["function"]["name"] for tc in tool_calls if "function" in tc
    )

    try:
        # Thinking comment — immediate
        await sse_callback("thinking", "Chuẩn bị công cụ...")

        # Tool start comment
        await sse_callback("thinking", f"Gọi: {tool_names}")

        # Emit tool_calls in OpenAI format (two chunks handled by sse_callback → tool_calls_event)
        await sse_callback("tool_calls", json.dumps(tool_calls))

    except Exception as exc:
        logger.exception("emit_tool_calls failed: %s", exc)
        # Surface error as visible content delta (no graph fallback from this node)
        try:
            await sse_callback("error", f"Tool call failed: {exc}")
        except Exception:
            pass

    return {}
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_new_arch.py::test_emit_tool_calls_sse_format -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add server/agent/emit_tool_calls.py tests/test_new_arch.py
git commit -m "feat: add emit_tool_calls node (SSE tool_calls emitter)"
```

---

## Task 7: Update `chat.py` — Schema, Message Loop, `sse_callback`, `initial_state`

**Files:**
- Modify: `server/routers/chat.py`
- Test: `tests/test_new_arch.py`

Five changes: (1) imports — add `ToolMessage`, `json`, `tool_calls_event`, (2) `ChatMessage` accepts nullable `content`, (3) message loop handles `role:"tool"`, (4) `sse_callback` dispatches `"tool_calls"`, (5) `initial_state` has new fields.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_new_arch.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_new_arch.py::test_chat_message_accepts_null_content -v
```
Expected: FAIL — `ValidationError` because `content: str` rejects `None`

- [ ] **Step 3: Update `server/routers/chat.py`**

**Change 1 — `ChatMessage` model** (replace lines 43-45):
```python
class ChatMessage(BaseModel):
    role: str
    content: str | None = None        # null in assistant tool_calls messages
    tool_calls: list[dict] | None = None  # present in assistant messages for Turn 2
    tool_call_id: str | None = None   # present in role:"tool" result messages
```

**Change 2 — imports** (add `ToolMessage` to langchain imports, `json` stdlib, `tool_calls_event` to sse imports):
```python
import json
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from server.streaming.sse import (
    thinking_event, tool_start_event, tool_done_event,
    tool_error_event, content_delta_event, done_event,
    heartbeat_comment, tool_calls_event,
)
```

**Change 3 — message conversion loop** (replace lines 113-117):
```python
    messages = []
    for msg in request.messages:
        if msg.role == "user":
            messages.append(HumanMessage(content=msg.content or ""))
        elif msg.role == "tool":
            messages.append(ToolMessage(
                content=msg.content or "",
                tool_call_id=msg.tool_call_id or "",
            ))
        else:
            messages.append(AIMessage(content=msg.content or ""))
```

**Change 4 — `sse_callback`** (add `tool_calls` branch after `error` branch):
```python
        elif event_type == "tool_calls":
            # content is JSON-serialized list of tool call dicts
            await event_queue.put(tool_calls_event(json.loads(content)))
```

**Change 5 — `initial_state`** (add 3 new fields at end of `initial_state` dict):
```python
        "pending_tool_calls": [],
        "is_tool_result_turn": False,
        "tool_turns_used": 0,
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_new_arch.py::test_chat_message_accepts_null_content -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add server/routers/chat.py tests/test_new_arch.py
git commit -m "feat: update chat.py for Turn 2 message support and tool_calls SSE callback"
```

---

## Task 8: Graph — Wire `tool_selector` and `emit_tool_calls` Nodes

**Files:**
- Modify: `server/agent/graph.py`
- Test: `tests/test_new_arch.py`

Replace `structural_analysis` fast-path with universal `tool_selector` insertion. All non-volatile intents flow through `tool_selector` → conditional → `emit_tool_calls|rag_search|generate`.

**Turn 2 bypass:** When `is_tool_result_turn=True`, `route_context` must be skipped entirely — it would re-parse message content and set `force_reindex=True` or `mentioned_files` from the tool result JSON, causing unwanted side effects. The graph adds a conditional edge from `classify_intent` that routes Turn 2 directly to `tool_selector` (which immediately returns `[]` due to the cap check), skipping `route_context`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_new_arch.py`:

```python
@pytest.mark.asyncio
async def test_graph_routes_to_emit_tool_calls():
    """Graph routes to emit_tool_calls when pending_tool_calls is non-empty."""
    from unittest.mock import AsyncMock, patch
    from server.agent.graph import build_agent_graph
    from langchain_core.messages import HumanMessage

    mock_qdrant = AsyncMock()
    mock_qdrant.count_by_file = AsyncMock(return_value=0)  # miss → will emit tools

    emitted_events = []

    async def mock_sse_callback(event_type, content):
        emitted_events.append(event_type)

    # Patch vllm_client so it never gets called (we expect graph to END at emit_tool_calls)
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
    mock_qdrant.count_by_file = AsyncMock(return_value=10)  # hit → no tools

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

    # No tool_calls were emitted — graph took fast path to rag_search/generate
    assert "tool_calls" not in emitted_events
    # Graph completed (result is a state dict, not an exception)
    assert isinstance(result, dict)


@pytest.mark.asyncio
async def test_graph_turn2_skips_route_context():
    """Turn 2 (is_tool_result_turn=True) skips route_context, goes classify_intent → tool_selector → rag_search."""
    from unittest.mock import AsyncMock, patch, call
    from server.agent.graph import build_agent_graph
    from langchain_core.messages import HumanMessage, ToolMessage

    mock_qdrant = AsyncMock()
    mock_qdrant.count_by_file = AsyncMock(return_value=5)

    route_context_called = []

    async def mock_sse_callback(event_type, content):
        pass

    mock_vllm = AsyncMock()

    # Patch route_context to detect if it's called
    with patch("server.agent.route_context.route_context", wraps=None) as mock_rc, \
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
            "intent": "unit_test",  # preserved from Turn 1
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_new_arch.py::test_graph_routes_to_emit_tool_calls tests/test_new_arch.py::test_graph_skips_to_rag_search -v
```
Expected: FAIL — `tool_selector` not in graph

- [ ] **Step 3: Update `server/agent/graph.py`**

Add imports at top:
```python
from server.agent.tool_selector import tool_selector as tool_selector_node
from server.agent.emit_tool_calls import emit_tool_calls as emit_tool_calls_base
```

Replace `_route_after_intent` function:
```python
def _route_after_intent(state: AgentState) -> str:
    """Preserved — volatile gate must fire before tool_selector."""
    if state.get("volatile_rejected"):
        return "reject_volatile"
    return "tool_selector"  # all other intents go through tool_selector
```

Add new routing functions after `_route_after_rag`:
```python
def _route_after_classify(state: AgentState) -> str:
    """Turn 2 bypass: skip route_context when tool results are present.

    On Turn 2, route_context would re-parse ToolMessage JSON content and
    set force_reindex/mentioned_files from the tool result text — causing
    unwanted side effects. Skip directly to tool_selector (which returns
    empty list due to is_tool_result_turn cap).
    """
    if state.get("is_tool_result_turn"):
        return "tool_selector"  # skip route_context entirely
    return "route_context"


def _route_after_tool_selector(state: AgentState) -> str:
    """Route based on whether tools need to be emitted."""
    if state.get("pending_tool_calls"):
        return "emit_tool_calls"
    if state.get("intent") == "structural_analysis":
        return "generate"  # structural_analysis: skeleton was not needed, generate directly
    return "rag_search"
```

Inside `build_agent_graph`, replace the entry edge and volatile routing block:

```python
    # --- Entry edge: classify_intent → conditional ---
    # Replace the existing: graph.add_edge("classify_intent", "route_context")
    # with conditional edge for Turn 2 bypass:
    graph.add_conditional_edges(
        "classify_intent",
        _route_after_classify,
        {
            "route_context": "route_context",
            "tool_selector": "tool_selector",  # Turn 2: skip route_context
        },
    )

    # Add new nodes
    graph.add_node(
        "tool_selector",
        partial(tool_selector_node, qdrant=qdrant),
    )
    graph.add_node(
        "emit_tool_calls",
        partial(emit_tool_calls_base, sse_callback=sse_callback),
    )

    # Update route_context edge: all non-volatile → tool_selector
    graph.add_conditional_edges(
        "route_context",
        _route_after_intent,
        {
            "reject_volatile": "reject_volatile",
            "tool_selector": "tool_selector",
        },
    )

    # After tool_selector: emit tools OR proceed to RAG/generate
    graph.add_conditional_edges(
        "tool_selector",
        _route_after_tool_selector,
        {
            "emit_tool_calls": "emit_tool_calls",
            "rag_search": "rag_search",
            "generate": "generate",
        },
    )

    graph.add_edge("emit_tool_calls", END)  # Turn 1 ends here
```

Delete these existing lines (find by pattern, not line number):
- Delete `graph.add_edge("classify_intent", "route_context")` — replaced by conditional edge above

> **CRITICAL — Find-and-replace, NOT line numbers:**
> 1. Find the existing `graph.add_conditional_edges("route_context", _route_after_intent, {...})` block (search for `add_conditional_edges("route_context"`) — **delete the entire call** including the `{...}` map with `"reject_volatile"`, `"generate"`, `"rag_search"` keys.
> 2. Replace it with the new `add_conditional_edges("route_context", ...)` call above (only `"reject_volatile"` and `"tool_selector"` in the map).
> 3. Find the old `def _route_after_intent(state)` function — **replace the entire function body**. The old one has 3 return paths (`reject_volatile`, `generate`, `rag_search`). The new one has 2 (`reject_volatile`, `tool_selector`).
> 4. Do NOT add a second `add_conditional_edges` call alongside the old one — LangGraph raises a compile error for duplicate source nodes.

- [ ] **Step 4: Run graph tests**

```bash
python -m pytest tests/test_new_arch.py::test_graph_routes_to_emit_tool_calls tests/test_new_arch.py::test_graph_skips_to_rag_search -v
```
Expected: PASS

Also run all existing agent tests to check for regressions:
```bash
python -m pytest tests/test_new_arch.py -v
```
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add server/agent/graph.py tests/test_new_arch.py
git commit -m "feat: wire tool_selector and emit_tool_calls into LangGraph agent"
```

---

## Task 9: Full Run + Regression Check

- [ ] **Step 1: Run the complete test suite**

```bash
python -m pytest tests/test_new_arch.py -v --tb=short
```
Expected: all 10 new tests + all prior tests PASS (no regressions)

- [ ] **Step 2: Verify server starts without import errors**

```bash
python -c "from server.agent.graph import build_agent_graph; print('OK')"
python -c "from server.routers.chat import router; print('OK')"
python -c "from server.streaming.sse import tool_calls_event; print('OK')"
```
Expected: all print `OK`

- [ ] **Step 3: Final commit**

```bash
git add -p  # stage any unstaged changes
git commit -m "feat: agent tool call flow complete (two-turn Continue MCP relay)"
```

---

## Quick Reference: Key Interfaces

**`tool_selector(state, qdrant) → dict`**
- Returns `{"pending_tool_calls": [...], "tool_turns_used": int}`
- Empty list = fast path (no tool turn)

**`emit_tool_calls(state, sse_callback) → {}`**
- Pure side-effect: streams SSE events via `sse_callback`
- Connected to `END` — always terminal

**`sse_callback(event_type, content)`**
- `"content"` → content delta
- `"thinking"` → SSE comment (invisible to Continue)
- `"tool_calls"` → `tool_calls_event(json.loads(content))` (two chunks)
- `"error"` → `tool_error_event(...)` (visible content delta)

**Turn 2 detection**
- `classify_intent` checks `isinstance(m, ToolMessage)` first
- Sets `is_tool_result_turn=True`, preserves prior intent
- `tool_selector` cap fires immediately → `pending_tool_calls = []`
