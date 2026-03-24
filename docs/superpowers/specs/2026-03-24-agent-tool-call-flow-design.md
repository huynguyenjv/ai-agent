# Agent Tool Call Flow — Design Spec
**Date:** 2026-03-24
**Status:** Approved

---

## Overview

This spec describes how the Cloud VM FastAPI server (agent) integrates with Continue IDE's MCP tool calling mechanism. The goal is to give the LangGraph agent the ability to trigger local MCP tools (index_with_deps, read_file, get_project_skeleton, search_symbol) on the developer's machine via Continue's tool calling relay, ensuring Qdrant always has fresh, relevant data before generation.

---

## Context & Constraints

- **Continue → Cloud VM → vLLM**: Continue sends standard OpenAI chat completion requests to the Cloud VM. Cloud VM is the proxy between Continue and vLLM.
- **MCP Server runs locally** on the developer machine, spawned by Continue as a stdio process. It has direct filesystem access.
- **Agent logic decides tools**, not the LLM (Qwen2.5-Coder). This avoids dependency on vLLM's function calling support and gives deterministic, predictable routing.
- **Max 1 tool turn per request**: After 1 round-trip of tool calls, the agent generates with whatever context it has. This caps latency and prevents multi-turn loops.
- **Thinking events must stream immediately** (< 50ms) even in Turn 1 before tool_calls are emitted, preventing blank screen.

---

## Architecture

### Two-Turn Flow

```
TURN 1 — Tool Phase
────────────────────────────────────────────────────────────────
Continue → POST /v1/chat/completions (user messages only)

  classify_intent
       ↓
  route_context (5-gate)
       ↓
  tool_selector (NEW)
  ├── checks intent + gates + Qdrant cache state
  ├── tool_turns_used >= 1 → pending_tool_calls = []
  └── produces pending_tool_calls: list[ToolCall]
       ↓
  [if pending_tool_calls not empty]
  emit_tool_calls (NEW)
  ├── stream: ": thinking: ..." (SSE comment, < 50ms)
  ├── stream: ": tool_start: ..." (SSE comment)
  └── stream: OpenAI tool_calls delta + finish_reason:"tool_calls"
  → END Turn 1

Continue receives tool_calls
  → calls MCP tools in parallel (index_with_deps, read_file, etc.)
  → MCP uploads chunks to POST /index if index_with_deps called
  → Continue sends new POST /v1/chat/completions with role:"tool" messages

TURN 2 — Generate Phase
────────────────────────────────────────────────────────────────
Continue → POST /v1/chat/completions (messages include role:"tool" results)

  classify_intent (detects is_tool_result_turn=True from role:"tool" in messages)
       ↓
  [is_tool_result_turn=True → SKIP route_context]
  tool_selector → pending_tool_calls = [] (is_tool_result_turn skips tools)
       ↓
  rag_search (Qdrant now has fresh data from index_with_deps)

  NOTE: route_context is skipped on Turn 2 to prevent it from re-parsing
  ToolMessage JSON content and setting force_reindex/mentioned_files incorrectly.
       ↓
  plan_steps (code_gen intent only)
       ↓
  generate → post_process → stream content → END
```

### Fast Path (No Tools Needed)

If `tool_selector` returns empty `pending_tool_calls`, the graph skips `emit_tool_calls` entirely and goes directly to `rag_search → generate` **in the same Turn 1 request**. Simple queries respond in 1 round-trip.

---

## Components

### New Node: `tool_selector`

**Location:** `server/agent/tool_selector.py`

**Inputs:** AgentState (intent, gates output, active_file, mentioned_files, freshness_signal, tool_turns_used)

**Output:** `{"pending_tool_calls": list[dict], "tool_turns_used": int}`

**Tool selection table:**

| Intent | Qdrant State | Gates | Tools Emitted |
|--------|-------------|-------|---------------|
| `structural_analysis` | any | — | `[get_project_skeleton]` |
| `search` | any | — | `[search_symbol(name)]` — always; exact name lookup beats semantic search. `name` is extracted from the last user message via regex: first capitalized identifier or quoted word. |
| `code_gen` / `refine` / `unit_test` | **miss** | file mentioned | `[index_with_deps(file), read_file(file)]` |
| `code_gen` / `refine` / `unit_test` | **hit** | file mentioned + freshness_signal | `[index_with_deps(file), read_file(file)]` |
| `code_gen` / `refine` / `unit_test` | **hit** | file mentioned, no freshness | `[read_file(file)]` — trust Qdrant cache |
| `code_gen` / `refine` / `unit_test` | **miss** | no file, active_file detected | `[index_with_deps(active_file)]` |
| `code_gen` / `refine` / `unit_test` | **hit** | no file, no freshness_signal | `[]` — RAG sufficient |
| `code_gen` / `refine` / `unit_test` | **miss** | no file, no active_file | `[]` — nothing to index, RAG only |
| `explain` | any | file mentioned | `[read_file(file)]` |
| `explain` | any | no file | `[]` — RAG sufficient |

**Qdrant cache check:** Before emitting `index_with_deps`, `tool_selector` calls `qdrant.count_by_file(file_path)` to determine if chunks exist. If count > 0, it's a "hit". On any exception, treat as miss.

`count_by_file()` implementation in `server/rag/qdrant_client.py`:
```python
async def count_by_file(self, file_path: str) -> int:
    """Return number of stored chunks for a given file_path. 0 = miss."""
    try:
        result = await self._client.count(
            collection_name=self._collection,
            count_filter=models.Filter(
                must=[models.FieldCondition(
                    key="file_path",
                    match=models.MatchValue(value=file_path),
                )]
            ),
            exact=False,  # approximate count is sufficient for hit/miss
        )
        return result.count
    except Exception:
        return 0
```

`ensure_collection()` must also create a payload index on `file_path` for this query to be fast:
```python
await self._client.create_payload_index(
    collection_name=self._collection,
    field_name="file_path",
    field_schema=models.PayloadSchemaType.KEYWORD,
)
```

**Cap logic:**
```python
if state["tool_turns_used"] >= 1 or state["is_tool_result_turn"]:
    return {"pending_tool_calls": [], "tool_turns_used": state["tool_turns_used"]}
```

**Counter increment:** When `tool_selector` decides to emit tools (i.e., `pending_tool_calls` is non-empty), it increments `tool_turns_used` in the same return dict:
```python
return {
    "pending_tool_calls": tool_calls,
    "tool_turns_used": state["tool_turns_used"] + 1,
}
```
This ensures the cap fires on the next call within the same session. `tool_turns_used` is per-request (initialized to 0 in `initial_state` for every new POST request), so it resets automatically between independent requests.

### New Node: `emit_tool_calls`

**Location:** `server/agent/emit_tool_calls.py`

**Inputs:** AgentState (pending_tool_calls)

**Behavior:**
1. SSE comment: `": thinking: Chuẩn bị công cụ...\n\n"` — immediate, < 50ms
2. SSE comment: `": tool_start: <tool_names>\n\n"`
3. OpenAI tool_calls delta chunks (one per tool in pending_tool_calls)
4. Final chunk with `finish_reason: "tool_calls"`
5. `data: [DONE]`

**Output:** `{}` — no state changes, pure SSE side-effect node

### `sse_callback` Extension

The existing `sse_callback(event_type, content)` in `chat.py` gains a new event type:
```python
elif event_type == "tool_calls":
    # content is JSON-serialized list of tool call dicts
    await event_queue.put(tool_calls_event(json.loads(content)))
```

New SSE formatter in `sse.py` — **must emit two separate SSE data lines** per OpenAI streaming spec (Continue accumulates tool_call arguments from Chunk 1 before processing finish_reason from Chunk 2):
```python
def tool_calls_event(tool_calls: list[dict], chunk_id: str = "chatcmpl-agent") -> str:
    """OpenAI tool_calls delta format for Continue.

    Emits two chunks:
      Chunk 1: delta with tool_calls list, finish_reason null
      Chunk 2: empty delta, finish_reason "tool_calls"
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

### AgentState Changes

Three new fields added to `server/agent/state.py`:

```python
pending_tool_calls: list[dict]   # tools to emit, set by tool_selector
is_tool_result_turn: bool        # True when request contains role:"tool" messages
tool_turns_used: int             # capped at 1, prevents > 2 round-trips
```

**`ChatMessage` model must accept Turn 2 message shapes** — the assistant message in Turn 2 has `content: null` and a `tool_calls` field. `ChatMessage` must be updated:
```python
class ChatMessage(BaseModel):
    role: str
    content: str | None = None        # null in assistant tool_calls messages
    tool_calls: list[dict] | None = None  # present in assistant messages for Turn 2
    tool_call_id: str | None = None   # present in role:"tool" result messages
```

**Message conversion loop in `chat.py` must handle `role:"tool"`** — use LangChain `ToolMessage`:
```python
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

for msg in request.messages:
    if msg.role == "user":
        messages.append(HumanMessage(content=msg.content or ""))
    elif msg.role == "tool":
        messages.append(ToolMessage(content=msg.content or "", tool_call_id=msg.tool_call_id or ""))
    else:
        messages.append(AIMessage(content=msg.content or ""))
```

Initial values in `chat.py`:
```python
"pending_tool_calls": [],
"is_tool_result_turn": False,
"tool_turns_used": 0,
```

**`is_tool_result_turn` detection** in `classify_intent` — **must be the first check, before any pattern matching**:
```python
from langchain_core.messages import ToolMessage

# MUST run first — before intent pattern loop
has_tool_result = any(
    isinstance(m, ToolMessage) or                        # LangChain ToolMessage (all versions)
    (hasattr(m, "type") and m.type == "tool") or         # fallback for older LangChain
    (isinstance(m, dict) and m.get("role") == "tool")    # raw dict fallback
    for m in messages
)
if has_tool_result:
    # Preserve intent from prior state; do not re-classify
    return {"intent": state.get("intent", "code_gen"), "is_tool_result_turn": True}
```

> **Implementation note:** `AgentState` must have `is_tool_result_turn` added before this change lands, as LangGraph will reject the returned key otherwise. Both changes are atomic.

### Graph Changes

**Location:** `server/agent/graph.py`

**Turn 2 bypass:** `classify_intent` now has a conditional edge. On Turn 2 (`is_tool_result_turn=True`), the graph skips `route_context` entirely and goes directly to `tool_selector`. This prevents `route_context` from re-parsing ToolMessage JSON and setting `force_reindex`/`mentioned_files` from tool result text.

```python
# classify_intent → conditional: Turn 2 bypass
graph.add_conditional_edges(
    "classify_intent",
    _route_after_classify,
    {
        "route_context": "route_context",     # Turn 1: normal flow
        "tool_selector": "tool_selector",     # Turn 2: skip route_context
    },
)

# Existing volatile edge preserved
graph.add_conditional_edges(
    "route_context",
    _route_after_intent,  # still routes volatile_rejected → reject_volatile
    {
        "reject_volatile": "reject_volatile",
        "tool_selector": "tool_selector",   # replaces previous "generate" and "rag_search" targets
    },
)

graph.add_node("tool_selector", tool_selector_node)
graph.add_node("emit_tool_calls", partial(emit_tool_calls_node, sse_callback=sse_callback))

graph.add_conditional_edges(
    "tool_selector",
    _route_after_tool_selector,
    {
        "emit_tool_calls": "emit_tool_calls",
        "rag_search": "rag_search",
        "generate": "generate",  # structural_analysis fast-path (no RAG needed)
    },
)

graph.add_edge("emit_tool_calls", END)  # Turn 1 ends here
```

```python
def _route_after_classify(state: AgentState) -> str:
    """Turn 2 bypass: skip route_context when tool results are present."""
    if state.get("is_tool_result_turn"):
        return "tool_selector"
    return "route_context"


def _route_after_intent(state: AgentState) -> str:
    """Preserved — volatile gate must fire before tool_selector."""
    if state.get("volatile_rejected"):
        return "reject_volatile"
    return "tool_selector"  # all other intents go through tool_selector


def _route_after_tool_selector(state: AgentState) -> str:
    if state.get("pending_tool_calls"):
        return "emit_tool_calls"
    if state.get("intent") == "structural_analysis":
        return "generate"  # structural_analysis skips RAG
    return "rag_search"
```

### MCP Tool Schemas (Continue config)

Continue requires tool schemas declared in the model provider config or inferred from MCP server. The MCP server already registers all 4 tools via `@server.list_tools()`. Continue auto-discovers them and maps tool call names from the agent.

Tool argument schemas:

**`index_with_deps`:**
```json
{"file_path": "string", "depth": "integer (default 2)"}
```

**`read_file`:**
```json
{"file_path": "string", "start_line": "integer (default 1)", "end_line": "integer (default 150)"}
```

**`get_project_skeleton`:**
```json
{"include_methods": "boolean (default true)"}
```

**`search_symbol`:**
```json
{"name": "string", "type_filter": "string (default 'any')"}
```

---

## SSE Stream Examples

### Turn 1 stream (tools needed):
```
: thinking: Phân tích intent...

: thinking: Cần index và đọc file...

: tool_start: index_with_deps, read_file

data: {"id":"chatcmpl-x","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"index_with_deps","arguments":"{\"file_path\":\"src/UserService.java\",\"depth\":2}"}},{"index":1,"id":"call_2","type":"function","function":{"name":"read_file","arguments":"{\"file_path\":\"src/UserService.java\"}"}}]},"finish_reason":null}]}

data: {"id":"chatcmpl-x","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}

data: [DONE]
```

> **Note:** Two separate `data:` lines are required. Chunk 1 carries the tool_call arguments with `finish_reason: null`. Chunk 2 carries the empty delta with `finish_reason: "tool_calls"`. Continue processes them sequentially — combining them into one chunk causes argument truncation.

### Turn 2 stream (generate):
```
: thinking: Đang xử lý...

data: {"id":"chatcmpl-x","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Here is the refactored..."},"finish_reason":null}]}

... more content tokens ...

data: [DONE]
```

---

## Error Handling

- **Tool selector Qdrant check fails** → treat as miss (emit index_with_deps to be safe), log warning
- **emit_tool_calls node crashes** → the node is connected to `END`, so there is no graph edge to fall back to `generate`. The SSE stream will terminate. The `emit_tool_calls` implementation must use a try/except internally: on exception, yield `tool_error_event("emit_tool_calls", str(exc))` as a content delta before the node returns `{}`. This surfaces the error visibly to the user without crashing the server.
- **Turn 2 arrives but Qdrant still empty** (MCP index failed) → rag_search returns empty, generate proceeds with message content context only, no crash

---

## Testing

New unit tests in `tests/test_new_arch.py`:

1. `test_tool_selector_structural_analysis` — verify `[get_project_skeleton]` for structural intent
2. `test_tool_selector_search` — verify `[search_symbol]` regardless of Qdrant state
3. `test_tool_selector_code_gen_miss` — file mentioned, Qdrant miss → `[index_with_deps, read_file]`
4. `test_tool_selector_code_gen_hit_no_freshness` — file mentioned, hit, no freshness → `[read_file]`
5. `test_tool_selector_cap` — tool_turns_used=1 → always `[]`
6. `test_tool_result_turn_detection` — messages with role:"tool" → is_tool_result_turn=True
7. `test_emit_tool_calls_sse_format` — verify OpenAI tool_calls SSE format
8. `test_graph_routes_to_emit_tool_calls` — verify graph edge when pending_tool_calls non-empty
9. `test_graph_skips_to_rag_search` — verify graph edge when pending_tool_calls empty
10. `test_graph_turn2_skips_route_context` — verify Turn 2 bypasses route_context, goes classify_intent → tool_selector → rag_search

---

## Files Changed

| File | Change |
|------|--------|
| `server/agent/tool_selector.py` | **NEW** — tool selection logic |
| `server/agent/emit_tool_calls.py` | **NEW** — SSE tool_calls emitter |
| `server/agent/state.py` | Add `pending_tool_calls`, `is_tool_result_turn`, `tool_turns_used` |
| `server/agent/classify_intent.py` | Detect `is_tool_result_turn` from role:"tool" messages |
| `server/agent/graph.py` | Add `tool_selector`, `emit_tool_calls` nodes and edges |
| `server/routers/chat.py` | Add new fields to initial_state, extend sse_callback |
| `server/streaming/sse.py` | Add `tool_calls_event()` formatter |
| `server/rag/qdrant_client.py` | Add `count_by_file()` method for cache check |
| `tests/test_new_arch.py` | Add 10 new test cases |
