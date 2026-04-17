# Chat Completions Native Tool-Call Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor `/v1/chat/completions` to use native tool-call, disable RAG via flag, stateless history, add server tool registry, fix streaming P0 bugs.

**Architecture:** LangGraph agent with simplified intent-based routing. Server defines tool schemas (read_file, grep_code), merges with client tools, forwards to vLLM with `tools=[...]`. Model decides tool_calls natively; Continue IDE executes tools client-side. Code review flow stays batch (non-streaming).

**Tech Stack:** FastAPI, LangGraph, OpenAI async SDK (vLLM), pytest, pydantic.

**Spec:** [docs/superpowers/specs/2026-04-15-chat-completions-native-toolcall-design.md](../specs/2026-04-15-chat-completions-native-toolcall-design.md)

---

## File Structure

**New files:**
- `server/tools/__init__.py` — package init
- `server/tools/registry.py` — tool schemas + `merge_tools()`
- `tests/test_tool_registry.py`
- `tests/test_generate_toolcall.py`
- `tests/test_graph_routing_v2.py`

**Modified files:**
- `server/agent/state.py` — add `client_tools`, `tool_choice`
- `server/agent/generate.py` — native tool-call streaming, drop RAG assembly
- `server/agent/graph.py` — remove tool_selector/emit_tool_calls nodes, intent router, RAG flag
- `server/agent/review_analyze.py` — ensure `stream=False`
- `server/routers/chat.py` — forward tools, P0 fixes (task ref, disconnect, cancel), message conversion
- `server/config.py` — `ENABLE_RAG` flag
- `server/app.py` — pass `enable_rag` into `build_agent_graph`
- `.env.example` — `ENABLE_RAG=false`

**Deprecated (kept, not imported):**
- `server/agent/tool_selector.py`
- `server/agent/emit_tool_calls.py`
- `server/agent/rag_search.py` (kept, unreferenced when flag off)
- `server/agent/plan_steps.py` (kept, unreferenced when flag off)

---

## Task 1: Tool Registry

**Files:**
- Create: `server/tools/__init__.py`
- Create: `server/tools/registry.py`
- Test: `tests/test_tool_registry.py`

- [ ] **Step 1: Write failing test**

`tests/test_tool_registry.py`:
```python
from server.tools.registry import TOOL_SCHEMAS, merge_tools


def test_tool_schemas_has_read_file_and_grep_code():
    names = {t["function"]["name"] for t in TOOL_SCHEMAS}
    assert names == {"read_file", "grep_code"}
    for t in TOOL_SCHEMAS:
        assert t["type"] == "function"
        assert "parameters" in t["function"]
        assert t["function"]["parameters"]["type"] == "object"


def test_merge_tools_empty_client_returns_server_schemas():
    merged = merge_tools(None)
    assert len(merged) == len(TOOL_SCHEMAS)
    assert merged == TOOL_SCHEMAS


def test_merge_tools_client_only_tool_preserved():
    client = [{"type": "function", "function": {"name": "foo", "parameters": {"type": "object", "properties": {}}}}]
    merged = merge_tools(client)
    names = {t["function"]["name"] for t in merged}
    assert names == {"read_file", "grep_code", "foo"}


def test_merge_tools_server_overrides_on_name_collision():
    fake_read_file = {"type": "function", "function": {"name": "read_file", "description": "HACKED", "parameters": {"type": "object", "properties": {}}}}
    merged = merge_tools([fake_read_file])
    read_file = next(t for t in merged if t["function"]["name"] == "read_file")
    assert read_file["function"]["description"] != "HACKED"
```

- [ ] **Step 2: Run test — expect failure**

```
pytest tests/test_tool_registry.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement registry**

`server/tools/__init__.py`:
```python
```

`server/tools/registry.py`:
```python
from __future__ import annotations

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read contents of a file by absolute path. Optionally limit to a line range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute file path"},
                    "start_line": {"type": "integer", "description": "Optional 1-indexed start line"},
                    "end_line": {"type": "integer", "description": "Optional 1-indexed end line"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep_code",
            "description": "Search for a regex pattern across the codebase using ripgrep-compatible syntax.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern"},
                    "path": {"type": "string", "description": "Optional path to restrict search"},
                    "glob": {"type": "string", "description": "Optional glob filter, e.g. *.java"},
                },
                "required": ["pattern"],
            },
        },
    },
]


def merge_tools(client_tools: list[dict] | None) -> list[dict]:
    """Merge server registry with client-provided tools.

    Dedupe by function.name. Server schemas take priority (overwrite client).
    """
    out: dict[str, dict] = {}
    for t in client_tools or []:
        name = t.get("function", {}).get("name")
        if name:
            out[name] = t
    for t in TOOL_SCHEMAS:
        out[t["function"]["name"]] = t
    return list(out.values())
```

- [ ] **Step 4: Run test — expect pass**

```
pytest tests/test_tool_registry.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add server/tools/__init__.py server/tools/registry.py tests/test_tool_registry.py
git commit -m "feat(tools): add server-side tool registry with read_file + grep_code"
```

---

## Task 2: Extend AgentState

**Files:**
- Modify: `server/agent/state.py`

- [ ] **Step 1: Add fields to AgentState**

Append inside the `AgentState` TypedDict (after `auto_post`):
```python
    # --- Native tool-call (client-forwarded) ---
    client_tools: list[dict]             # raw tool schemas from ChatRequest.tools
    tool_choice: str | dict | None       # forwarded tool_choice
```

- [ ] **Step 2: Verify nothing breaks**

```
pytest tests/ -q -x
```
Expected: same pass/fail ratio as before (new fields are optional via `total=False`).

- [ ] **Step 3: Commit**

```bash
git add server/agent/state.py
git commit -m "feat(state): add client_tools + tool_choice fields for native tool-call"
```

---

## Task 3: Feature Flag `ENABLE_RAG`

**Files:**
- Modify: `server/config.py`
- Modify: `.env.example`

- [ ] **Step 1: Read current config.py**

Open `server/config.py` to find the settings/class pattern. Add flag there (bool, default False).

- [ ] **Step 2: Add flag**

Add inside the settings class/dataclass/function:
```python
    enable_rag: bool = False  # False = bypass rag_search/plan_steps nodes
```

And ensure it reads from env:
```python
os.getenv("ENABLE_RAG", "false").lower() in ("1", "true", "yes")
```

- [ ] **Step 3: Update `.env.example`**

Append:
```
# RAG retrieval — keep false until new RAG design ships
ENABLE_RAG=false
```

- [ ] **Step 4: Commit**

```bash
git add server/config.py .env.example
git commit -m "feat(config): add ENABLE_RAG flag (default false)"
```

---

## Task 4: Refactor `generate` — Native Tool-Call Streaming

**Files:**
- Modify: `server/agent/generate.py`
- Test: `tests/test_generate_toolcall.py`

- [ ] **Step 1: Write failing tests**

`tests/test_generate_toolcall.py`:
```python
import json
import pytest
from unittest.mock import AsyncMock, MagicMock
from langchain_core.messages import HumanMessage

from server.agent.generate import generate


class FakeChoice:
    def __init__(self, delta, finish_reason=None):
        self.delta = delta
        self.finish_reason = finish_reason


class FakeChunk:
    def __init__(self, delta=None, finish_reason=None):
        self.choices = [FakeChoice(delta or MagicMock(content=None, tool_calls=None), finish_reason)]


class FakeStream:
    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        self._iter = iter(self._chunks)
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration


@pytest.mark.asyncio
async def test_generate_streams_content():
    calls = []

    async def sse_cb(event_type, content):
        calls.append((event_type, content))

    d1 = MagicMock(content="Hello ", tool_calls=None)
    d2 = MagicMock(content="world", tool_calls=None)
    stream = FakeStream([FakeChunk(d1), FakeChunk(d2, finish_reason="stop")])

    vllm = MagicMock()
    vllm.chat.completions.create = AsyncMock(return_value=stream)

    state = {"messages": [HumanMessage(content="Hi")], "client_tools": []}
    result = await generate(state, vllm_client=vllm, model="m", sse_callback=sse_cb)

    assert result["draft"] == "Hello world"
    assert ("content", "Hello ") in calls
    assert ("content", "world") in calls


@pytest.mark.asyncio
async def test_generate_captures_tool_calls():
    tc_delta = [{"index": 0, "id": "call_1", "type": "function",
                 "function": {"name": "read_file", "arguments": '{"path":"/a"}'}}]
    d1 = MagicMock(content=None, tool_calls=tc_delta)
    stream = FakeStream([FakeChunk(d1, finish_reason="tool_calls")])

    vllm = MagicMock()
    vllm.chat.completions.create = AsyncMock(return_value=stream)

    state = {"messages": [HumanMessage(content="read /a")], "client_tools": []}
    result = await generate(state, vllm_client=vllm, model="m", sse_callback=None)

    assert result["pending_tool_calls"]
    assert result["pending_tool_calls"][0]["function"]["name"] == "read_file"


@pytest.mark.asyncio
async def test_generate_forwards_merged_tools_to_vllm():
    d1 = MagicMock(content="ok", tool_calls=None)
    stream = FakeStream([FakeChunk(d1, finish_reason="stop")])
    vllm = MagicMock()
    vllm.chat.completions.create = AsyncMock(return_value=stream)

    state = {
        "messages": [HumanMessage(content="hi")],
        "client_tools": [{"type": "function", "function": {"name": "foo", "parameters": {"type": "object", "properties": {}}}}],
    }
    await generate(state, vllm_client=vllm, model="m", sse_callback=None)

    call_kwargs = vllm.chat.completions.create.call_args.kwargs
    tool_names = {t["function"]["name"] for t in call_kwargs["tools"]}
    assert tool_names == {"read_file", "grep_code", "foo"}
    assert call_kwargs["stream"] is True
```

- [ ] **Step 2: Run — expect failure**

```
pytest tests/test_generate_toolcall.py -v
```
Expected: failures (tool_calls not captured, tools not forwarded).

- [ ] **Step 3: Rewrite `generate.py`**

Replace entire content of `server/agent/generate.py`:
```python
"""Node: generate — native tool-call streaming.

Forwards merged tools (server registry + client) to vLLM, streams content
tokens, and captures tool_call deltas to accumulate a final pending_tool_calls
list for Turn 2.
"""

from __future__ import annotations

import logging

from openai import AsyncOpenAI

from server.agent.state import AgentState
from server.tools.registry import merge_tools

logger = logging.getLogger("server.agent.generate")

BASE_SYSTEM_PROMPT = (
    "You are an expert coding assistant. Provide accurate, well-structured answers. "
    "If you need to inspect files or search the codebase, call the provided tools. "
    "Respond in the same language as the user's query (Vietnamese or English)."
)


def _to_openai_messages(state: AgentState) -> list[dict]:
    """Convert LangChain messages to OpenAI chat format, preserving tool_calls."""
    out: list[dict] = [{"role": "system", "content": BASE_SYSTEM_PROMPT}]
    for msg in state.get("messages", []):
        mtype = getattr(msg, "type", None)
        if mtype == "human":
            out.append({"role": "user", "content": msg.content or ""})
        elif mtype == "ai":
            item: dict = {"role": "assistant", "content": msg.content or ""}
            tc = (getattr(msg, "additional_kwargs", {}) or {}).get("tool_calls")
            if tc:
                item["tool_calls"] = tc
                item["content"] = None
            out.append(item)
        elif mtype == "tool":
            out.append({
                "role": "tool",
                "content": msg.content or "",
                "tool_call_id": getattr(msg, "tool_call_id", "") or "",
            })
        elif mtype == "system":
            out.append({"role": "system", "content": msg.content or ""})
        else:
            out.append({"role": "user", "content": str(getattr(msg, "content", msg))})
    return out


def _merge_tool_call_delta(acc: list[dict], delta_list: list[dict]) -> None:
    """Accumulate OpenAI streaming tool_call deltas by index."""
    for d in delta_list:
        idx = d.get("index", 0)
        while len(acc) <= idx:
            acc.append({"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
        slot = acc[idx]
        if d.get("id"):
            slot["id"] = d["id"]
        if d.get("type"):
            slot["type"] = d["type"]
        fn = d.get("function") or {}
        if fn.get("name"):
            slot["function"]["name"] += fn["name"] if isinstance(fn["name"], str) else ""
        if fn.get("arguments"):
            slot["function"]["arguments"] += fn["arguments"]


async def generate(
    state: AgentState,
    vllm_client: AsyncOpenAI,
    model: str,
    sse_callback=None,
) -> dict:
    merged_tools = merge_tools(state.get("client_tools"))
    messages = _to_openai_messages(state)

    kwargs: dict = {
        "model": model,
        "messages": messages,
        "tools": merged_tools,
        "stream": True,
        "max_tokens": 4096,
        "temperature": 0.3,
    }
    tc = state.get("tool_choice")
    if tc is not None:
        kwargs["tool_choice"] = tc

    content_buf: list[str] = []
    tool_calls_acc: list[dict] = []

    try:
        stream = await vllm_client.chat.completions.create(**kwargs)
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if getattr(delta, "content", None):
                token = delta.content
                content_buf.append(token)
                if sse_callback:
                    await sse_callback("content", token)
            if getattr(delta, "tool_calls", None):
                tc_delta = delta.tool_calls
                if hasattr(tc_delta[0], "model_dump"):
                    tc_delta = [t.model_dump() for t in tc_delta]
                _merge_tool_call_delta(tool_calls_acc, tc_delta)
    except Exception as e:
        logger.error("vLLM generation failed: %s", e)
        if sse_callback:
            await sse_callback("error", str(e))
        return {"draft": f"Generation error: {e}", "pending_tool_calls": []}

    return {
        "draft": "".join(content_buf),
        "pending_tool_calls": tool_calls_acc,
    }
```

- [ ] **Step 4: Run — expect pass**

```
pytest tests/test_generate_toolcall.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add server/agent/generate.py tests/test_generate_toolcall.py
git commit -m "refactor(generate): native tool-call streaming with merged registry"
```

---

## Task 5: Refactor Graph — Intent Router + Drop tool_selector/emit_tool_calls + RAG Flag

**Files:**
- Modify: `server/agent/graph.py`
- Test: `tests/test_graph_routing_v2.py`

- [ ] **Step 1: Write failing tests**

`tests/test_graph_routing_v2.py`:
```python
import pytest
from unittest.mock import AsyncMock, MagicMock
from langchain_core.messages import HumanMessage

from server.agent.graph import build_agent_graph


@pytest.fixture
def fake_services():
    vllm = MagicMock()
    # generate() is stubbed via monkeypatch in each test
    qdrant = MagicMock()
    embedder = MagicMock()
    return vllm, qdrant, embedder


@pytest.mark.asyncio
async def test_graph_rag_off_code_gen_goes_direct_to_generate(fake_services, monkeypatch):
    vllm, qdrant, embedder = fake_services
    calls = []

    async def fake_generate(state, **kw):
        calls.append("generate")
        return {"draft": "ok", "pending_tool_calls": []}

    monkeypatch.setattr("server.agent.graph.generate", fake_generate)

    async def fake_classify(state):
        return {"intent": "code_gen", "is_tool_result_turn": False}
    monkeypatch.setattr("server.agent.graph.classify_intent", fake_classify)

    async def fake_route_ctx(state):
        return {"volatile_rejected": False, "mentioned_files": []}
    monkeypatch.setattr("server.agent.graph.route_context", fake_route_ctx)

    graph = build_agent_graph(vllm, "m", qdrant, embedder, enable_rag=False)
    result = await graph.ainvoke({"messages": [HumanMessage(content="gen a function")]})

    assert "generate" in calls
    assert result["draft"] == "ok"


@pytest.mark.asyncio
async def test_graph_tool_result_turn_bypasses_route_context(fake_services, monkeypatch):
    vllm, qdrant, embedder = fake_services

    seen = []

    async def fake_classify(state):
        seen.append("classify")
        return {"intent": "explain", "is_tool_result_turn": True}

    async def fake_route_ctx(state):
        seen.append("route_context")
        return {}

    async def fake_generate(state, **kw):
        seen.append("generate")
        return {"draft": "x", "pending_tool_calls": []}

    monkeypatch.setattr("server.agent.graph.classify_intent", fake_classify)
    monkeypatch.setattr("server.agent.graph.route_context", fake_route_ctx)
    monkeypatch.setattr("server.agent.graph.generate", fake_generate)

    graph = build_agent_graph(vllm, "m", qdrant, embedder, enable_rag=False)
    await graph.ainvoke({"messages": [HumanMessage(content="tool result")]})

    assert "route_context" not in seen
    assert "generate" in seen
```

- [ ] **Step 2: Run — expect failure**

```
pytest tests/test_graph_routing_v2.py -v
```
Expected: failures (build_agent_graph has no `enable_rag` kwarg; graph still has old nodes).

- [ ] **Step 3: Rewrite `server/agent/graph.py`**

Replace entire content:
```python
"""LangGraph agent graph — native tool-call edition.

Flow:
  classify_intent
    ├─ is_tool_result_turn=True → generate → post_process → END
    └─ else → route_context
         ├─ volatile_rejected → reject_volatile → END
         └─ else → <intent router>
              ├─ code_review → review_analyze → review_format
              │                 ├─ auto_post → upsert_mr_comment → END
              │                 └─ else → post_process → END
              └─ else → [rag_search → plan_steps?] → generate → post_process → END

rag_search / plan_steps only wired when enable_rag=True.
"""

from __future__ import annotations

from functools import partial

from langgraph.graph import END, StateGraph

from server.agent.state import AgentState
from server.agent.classify_intent import classify_intent
from server.agent.route_context import route_context
from server.agent.generate import generate
from server.agent.post_process import post_process
from server.agent.review_analyze import review_analyze
from server.agent.review_format import review_format
from server.agent.upsert_mr_comment import upsert_mr_comment

_VOLATILE_RESPONSE = (
    "Xin lỗi, tính năng này chưa được hỗ trợ trong phiên bản hiện tại (V1). "
    "Hệ thống chưa thể truy cập dữ liệu real-time như git diff, runtime logs, "
    "live metrics, hoặc error stack traces từ process đang chạy. "
    "Vui lòng mô tả vấn đề cụ thể để tôi hỗ trợ dựa trên source code."
)


def _reject_volatile(state: AgentState) -> dict:
    return {"draft": _VOLATILE_RESPONSE}


def _route_after_classify(state: AgentState) -> str:
    if state.get("is_tool_result_turn"):
        return "generate"
    return "route_context"


def _route_after_context(state: AgentState) -> str:
    if state.get("volatile_rejected"):
        return "reject_volatile"
    if state.get("intent") == "code_review":
        return "review_analyze"
    return "generate"


def _route_after_review_format(state: AgentState) -> str:
    return "upsert_mr_comment" if state.get("auto_post") else "post_process"


def build_agent_graph(
    vllm_client,
    model: str,
    qdrant,
    embedder,
    sse_callback=None,
    *,
    enable_rag: bool = False,
):
    graph = StateGraph(AgentState)

    graph.add_node("classify_intent", classify_intent)
    graph.add_node("route_context", route_context)
    graph.add_node("reject_volatile", _reject_volatile)
    graph.add_node(
        "generate",
        partial(generate, vllm_client=vllm_client, model=model, sse_callback=sse_callback),
    )
    graph.add_node("post_process", post_process)

    graph.add_node(
        "review_analyze",
        partial(review_analyze, vllm_client=vllm_client, model=model),
    )
    graph.add_node("review_format", review_format)
    graph.add_node("upsert_mr_comment", upsert_mr_comment)

    graph.set_entry_point("classify_intent")

    graph.add_conditional_edges(
        "classify_intent",
        _route_after_classify,
        {"route_context": "route_context", "generate": "generate"},
    )

    graph.add_conditional_edges(
        "route_context",
        _route_after_context,
        {
            "reject_volatile": "reject_volatile",
            "review_analyze": "review_analyze",
            "generate": "generate",
        },
    )

    graph.add_edge("reject_volatile", END)

    graph.add_edge("review_analyze", "review_format")
    graph.add_conditional_edges(
        "review_format",
        _route_after_review_format,
        {"upsert_mr_comment": "upsert_mr_comment", "post_process": "post_process"},
    )
    graph.add_edge("upsert_mr_comment", END)

    graph.add_edge("generate", "post_process")
    graph.add_edge("post_process", END)

    # RAG path — re-enable in a future change by flipping enable_rag.
    if enable_rag:
        from server.agent.rag_search import rag_search
        from server.agent.plan_steps import plan_steps

        graph.add_node("rag_search", partial(rag_search, qdrant=qdrant, embedder=embedder))
        graph.add_node("plan_steps", partial(plan_steps, vllm_client=vllm_client, model=model))
        # Intentionally left un-wired: future task will insert rag_search before generate
        # and plan_steps for code_gen. Keeping nodes resident to simplify re-activation.

    return graph.compile()
```

- [ ] **Step 4: Run — expect pass**

```
pytest tests/test_graph_routing_v2.py -v
```
Expected: 2 passed.

- [ ] **Step 5: Run existing tests; remove/skip dead ones**

```
pytest tests/ -q
```
If `tests/test_new_arch.py` has tool_selector/emit_tool_calls tests, skip them with `@pytest.mark.skip(reason="tool_selector deprecated — native tool-call")` at class/function level. Do NOT delete yet.

- [ ] **Step 6: Commit**

```bash
git add server/agent/graph.py tests/test_graph_routing_v2.py tests/test_new_arch.py
git commit -m "refactor(graph): drop tool_selector/emit_tool_calls, intent router, enable_rag flag"
```

---

## Task 6: Deprecate Unused Nodes (Header Banner Only)

**Files:**
- Modify: `server/agent/tool_selector.py`
- Modify: `server/agent/emit_tool_calls.py`

- [ ] **Step 1: Add deprecation banner to `tool_selector.py`**

Prepend:
```python
"""DEPRECATED — superseded by native tool-call flow in generate.py.

Kept for historical reference; not imported by the agent graph when using
the native tool-call path. Do not add new call sites.
"""
```

- [ ] **Step 2: Same for `emit_tool_calls.py`**

Prepend the identical banner (update file name in first line if preferred).

- [ ] **Step 3: Verify nothing imports them**

```
grep -rn "from server.agent.tool_selector\|from server.agent.emit_tool_calls\|import server.agent.tool_selector\|import server.agent.emit_tool_calls" server/ tests/
```
Expected: no matches outside the files themselves (tests may still import — if so, skip those tests per Task 5 step 5).

- [ ] **Step 4: Commit**

```bash
git add server/agent/tool_selector.py server/agent/emit_tool_calls.py
git commit -m "chore: mark tool_selector and emit_tool_calls as deprecated"
```

---

## Task 7: Refactor `chat.py` — Forward tools, Message Conversion, P0 Fixes

**Files:**
- Modify: `server/routers/chat.py`

- [ ] **Step 1: Update `ChatRequest` + `ChatMessage`**

Replace the existing Pydantic models in `server/routers/chat.py`:
```python
class ChatMessage(BaseModel):
    role: str
    content: str | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None


class ChatRequest(BaseModel):
    model_config = {"extra": "allow"}

    messages: list[ChatMessage]
    model: str = ""
    stream: bool = True
    tools: list[dict] | None = None           # forwarded to vLLM
    tool_choice: str | dict | None = None     # forwarded to vLLM
    active_file: str | None = None
    repo_path: str | None = None
```

- [ ] **Step 2: Update message conversion**

In `_stream_response`, replace the message conversion block with:
```python
from langchain_core.messages import SystemMessage

messages = []
for msg in request.messages:
    if msg.role == "user":
        messages.append(HumanMessage(content=msg.content or ""))
    elif msg.role == "tool":
        messages.append(ToolMessage(
            content=msg.content or "",
            tool_call_id=msg.tool_call_id or "",
        ))
    elif msg.role == "assistant":
        ai = AIMessage(content=msg.content or "")
        if msg.tool_calls:
            ai.additional_kwargs["tool_calls"] = msg.tool_calls
        messages.append(ai)
    elif msg.role == "system":
        messages.append(SystemMessage(content=msg.content or ""))
    else:
        messages.append(HumanMessage(content=msg.content or ""))
```

- [ ] **Step 3: Add `client_tools` + `tool_choice` + `enable_rag` to initial state / graph build**

Replace the `initial_state` dict:
```python
initial_state = {
    "messages": messages,
    "intent": "",
    "active_file": active_file,
    "mentioned_files": [],
    "freshness_signal": False,
    "force_reindex": False,
    "rag_chunks": [],
    "rag_hit": False,
    "hash_verified": False,
    "tool_results": [],
    "context_assembled": "",
    "draft": "",
    "emitted_steps": [],
    "volatile_rejected": False,
    "pending_tool_calls": [],
    "is_tool_result_turn": False,
    "tool_turns_used": 0,
    "client_tools": request.tools or [],
    "tool_choice": request.tool_choice,
}
```

And pass `enable_rag` when building the graph:
```python
from server.config import get_settings  # adapt to your actual accessor
settings = get_settings()

agent = build_agent_graph(
    vllm_client=vllm_client,
    model=model,
    qdrant=qdrant,
    embedder=embedder,
    sse_callback=sse_callback,
    enable_rag=settings.enable_rag,
)
```

> If `server/config.py` uses a module-level `ENABLE_RAG` instead of settings object, import that symbol directly.

- [ ] **Step 4: Emit tool_calls to SSE when present after generate**

Extend `sse_callback`: the generate node currently only emits `content` events through callback. After agent finishes, `agent_result["pending_tool_calls"]` may be non-empty. Add, right before `yield done_event()`:
```python
if isinstance(agent_result, dict):
    tc = agent_result.get("pending_tool_calls") or []
    if tc and not content_streamed:
        yield tool_calls_event(tc)
```

- [ ] **Step 5: P0 fix — hold task ref + cancel in finally**

Replace the agent-run + loop section with:
```python
import contextlib

run_task = asyncio.create_task(_run_agent())

yield thinking_event("Đang xử lý...")

last_event_time = time.monotonic()
agent_result = None
try:
    while True:
        if await req.is_disconnected():
            logger.info("Client disconnected, cancelling agent")
            break
        try:
            event = await asyncio.wait_for(event_queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            if time.monotonic() - last_event_time > 15:
                yield heartbeat_comment()
                last_event_time = time.monotonic()
            continue

        if isinstance(event, tuple) and len(event) == 2 and event[0] is _SENTINEL:
            agent_result = event[1]
            break

        yield event
        last_event_time = time.monotonic()
finally:
    if not run_task.done():
        run_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await run_task
```

- [ ] **Step 6: Run existing chat endpoint tests**

```
pytest tests/ -q -k chat
```
Expected: passing (or expected failures documented).

- [ ] **Step 7: Commit**

```bash
git add server/routers/chat.py
git commit -m "refactor(chat): forward tools, preserve tool_calls, P0 fixes (disconnect+cancel)"
```

---

## Task 8: Ensure Review Flow Stays Batch (Non-Streaming)

**Files:**
- Modify: `server/agent/review_analyze.py`

- [ ] **Step 1: Read `review_analyze.py`**

Verify the vLLM call uses `stream=False` (or no `stream` arg, which defaults to non-stream).

- [ ] **Step 2: Make it explicit**

Wherever the file calls `vllm_client.chat.completions.create(...)`, ensure:
```python
stream=False,
```
is passed. If already non-streaming, add an inline comment `# batch response — V1 per spec` so future readers don't flip it.

- [ ] **Step 3: Run review tests**

```
pytest tests/test_review_format.py tests/test_tools_review.py -v
```
Expected: passing.

- [ ] **Step 4: Commit**

```bash
git add server/agent/review_analyze.py
git commit -m "chore(review): pin stream=False (batch response for V1)"
```

---

## Task 9: Integration Smoke Test — Chat Endpoint End-to-End

**Files:**
- Test: `tests/test_chat_endpoint_v2.py`

- [ ] **Step 1: Write integration test**

`tests/test_chat_endpoint_v2.py`:
```python
import json
import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock, MagicMock

from server.app import create_app


class FakeChoice:
    def __init__(self, delta, finish_reason=None):
        self.delta = delta
        self.finish_reason = finish_reason


class FakeChunk:
    def __init__(self, delta, finish_reason=None):
        self.choices = [FakeChoice(delta, finish_reason)]


class FakeStream:
    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        self._it = iter(self._chunks)
        return self

    async def __anext__(self):
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration


@pytest.mark.asyncio
async def test_chat_endpoint_streams_content(monkeypatch):
    app = create_app()

    d1 = MagicMock(content="Hi", tool_calls=None)
    d2 = MagicMock(content=" there", tool_calls=None)
    stream = FakeStream([FakeChunk(d1), FakeChunk(d2, finish_reason="stop")])

    app.state.vllm_client.chat.completions.create = AsyncMock(return_value=stream)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post(
            "/v1/chat/completions",
            headers={"X-Api-Key": "test-key"},  # adjust to match auth
            json={"messages": [{"role": "user", "content": "say hi"}], "stream": True},
        )
        body = r.text
    assert "Hi" in body or "there" in body
    assert "[DONE]" in body
```

> If `create_app` signature or auth differs, adapt. If the app factory is not suitable, instantiate routers directly.

- [ ] **Step 2: Run — iterate until green**

```
pytest tests/test_chat_endpoint_v2.py -v
```
Address real bugs found here — this is the acceptance gate.

- [ ] **Step 3: Commit**

```bash
git add tests/test_chat_endpoint_v2.py
git commit -m "test(chat): add end-to-end streaming smoke test for native tool-call"
```

---

## Task 10: Manual Verification

- [ ] **Step 1: Start server**

```
uvicorn server.app:app --reload --port 8080
```

- [ ] **Step 2: Smoke plain chat**

With Continue (or curl), send:
```
curl -N -H "X-Api-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello"}],"stream":true}' \
  http://localhost:8080/v1/chat/completions
```
Expected: SSE `data:` lines with content deltas + final `[DONE]`.

- [ ] **Step 3: Smoke tool-call round trip via Continue**

Ask "read /path/to/foo.java". Verify Continue shows tool_call, executes, sends Turn 2, final answer streams.

- [ ] **Step 4: Verify client disconnect cancels**

Start a request, Ctrl+C the curl mid-stream, check server logs for "Client disconnected, cancelling agent" and no orphan task warnings.

- [ ] **Step 5: Final commit (if any tweaks needed)**

If manual test revealed fixes, commit them with descriptive messages.

---

## Done when
- All 10 tasks' tests pass.
- Manual smoke (plain chat, tool round-trip, disconnect) works.
- `ENABLE_RAG=false` by default; graph does not wire rag_search/plan_steps.
- Deprecated files have banners; no production imports reference them.
