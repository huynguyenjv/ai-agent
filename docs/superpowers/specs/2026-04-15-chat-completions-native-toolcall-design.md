# Design Spec — `/v1/chat/completions` Native Tool-Call + Intent-Aware Graph

**Date:** 2026-04-15
**Branch:** `feature/new-architecture`
**Status:** Draft — pending user approval

---

## 1. Mục tiêu

Đơn giản hóa pipeline `/v1/chat/completions` để:

1. **Tận dụng native tool-call** của model (Qwen3-30B-A3B-Instruct-2507-FP8 đã support OpenAI tool-call spec) — bỏ module `tool_selector` / `emit_tool_calls` heuristic.
2. **Pending RAG** — tắt `rag_search` + `plan_steps` qua feature flag, giữ code để bật lại sau khi RAG được thiết kế lại.
3. **Stateless history** — client (Continue) gửi full history mỗi request, server không lưu session. Dễ triển khai, token overhead server = 0.
4. **Server tool registry** — server định nghĩa schema của một số tool nội bộ (`read_file`, `grep_code`), merge với tools từ client trước khi gửi vLLM.
5. **Fix P0 bugs** của streaming endpoint hiện tại (task GC, client disconnect, agent không cancel).

Non-goals (giai đoạn này):
- RAG retrieval / embedding / qdrant usage trong chat flow.
- Inter-session summarization.
- Server-side session storage.
- Server-side tool execution (Continue tự thực thi tool).
- **Streaming review kết quả** — code_review flow trả về **batch response** (non-streaming) cho V1; chỉ `/v1/chat/completions` chat path streaming.

---

## 2. Kiến trúc

### 2.1. Request / Response flow

```
Continue (IDE)
   │ POST /v1/chat/completions
   │ { messages: [...full history...], tools?: [...], stream: true }
   ▼
FastAPI router (server/routers/chat.py)
   │  - verify_api_key
   │  - convert messages → LangChain format
   │  - build initial AgentState
   ▼
LangGraph agent
   classify_intent → route_context → <intent router> → post_process → END
   │
   │ (generate node)
   ▼
vLLM client
   POST /v1/chat/completions
   { model, messages, tools: merge(server_registry, client_tools), stream: true }
   │
   ▼
Stream chunks → SSE callback → back to Continue
```

**Turn 2 (tool result):**
- Continue chạy tool xong, gửi lại request chứa `role: "assistant"` với `tool_calls` + `role: "tool"` với kết quả.
- `classify_intent` detect `is_tool_result_turn=True` → bypass `route_context` → đi thẳng vào `generate`.
- `generate` gửi full history (bao gồm tool results) xuống vLLM → stream final content.

### 2.2. Graph mới

```
                    ┌──────────────────┐
                    │ classify_intent  │
                    └────────┬─────────┘
                             │
              is_tool_result_turn?
                ┌────────────┴────────────┐
               yes                        no
                │                          │
                ▼                          ▼
            generate            ┌──────────────────┐
                                │  route_context   │
                                └────────┬─────────┘
                                         │
                                  volatile_rejected?
                                  ┌──────┴──────┐
                                 yes            no
                                  │              │
                                  ▼              ▼
                          reject_volatile   <intent router>
                                  │              │
                                  │   ┌──────────┼───────────┐
                                  │   │          │           │
                                  │   ▼          ▼           ▼
                                  │ review   generate   (code_gen
                                  │ _analyze            → generate
                                  │   │                 vì RAG off)
                                  │   ▼
                                  │ review_format
                                  │   │
                                  │   ├── auto_post? → upsert_mr_comment → END
                                  │   └── → post_process
                                  │              │
                                  └──────────────┴─────────→ END
```

**Xóa khỏi graph:**
- `tool_selector`
- `emit_tool_calls`

**Tắt qua flag `ENABLE_RAG=false` (code giữ nguyên, chỉ bỏ edge):**
- `rag_search`
- `plan_steps`

**Intent router** — hàm `_route_by_intent(state)`:

| Intent | Route (RAG off) | Route (RAG on, tương lai) |
|---|---|---|
| `code_review` | `review_analyze` (batch, non-stream) | `review_analyze` (batch) |
| `code_gen` | `generate` | `rag_search → plan_steps → generate` |
| `unit_test` | `generate` | `rag_search → generate` |
| `explain` | `generate` | `rag_search → generate` |
| `search` | `generate` | `rag_search → generate` |
| `refine` | `generate` | `rag_search → generate` |
| `structural_analysis` | `generate` | `generate` (không RAG) |

### 2.3. Tool Registry (server-side)

**File mới:** `server/tools/registry.py`

```python
from __future__ import annotations

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read contents of a file by absolute path.",
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
            "description": "Search for a regex pattern across the codebase.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern (ripgrep syntax)"},
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

**Execution model:** server KHÔNG thực thi tool. Continue nhận `tool_calls` chunk → tự chạy tool → gửi lại kết quả ở Turn 2. Registry chỉ khai báo schema để model biết có gì mà gọi.

### 2.4. Generate node — streaming native tool-call

File: `server/agent/generate.py` (refactor).

```python
async def generate(state, *, vllm_client, model, sse_callback=None):
    merged_tools = merge_tools(state.get("client_tools"))
    resp = await vllm_client.chat.completions.create(
        model=model,
        messages=_to_openai(state["messages"]),
        tools=merged_tools,
        stream=True,
    )
    accumulated_content = ""
    accumulated_tool_calls: list[dict] = []
    async for chunk in resp:
        delta = chunk.choices[0].delta
        if delta.content:
            accumulated_content += delta.content
            if sse_callback:
                await sse_callback("content", delta.content)
        if delta.tool_calls:
            _merge_tool_call_delta(accumulated_tool_calls, delta.tool_calls)
            if sse_callback:
                await sse_callback("tool_calls_delta", json.dumps(delta.tool_calls))
        if chunk.choices[0].finish_reason == "tool_calls":
            if sse_callback:
                await sse_callback("tool_calls_final", json.dumps(accumulated_tool_calls))
    return {
        "draft": accumulated_content,
        "pending_tool_calls": accumulated_tool_calls,
    }
```

**Ghi chú:**
- Stream chunk relay trực tiếp — Continue đã hiểu OpenAI delta format.
- `client_tools` được lấy từ request (`ChatRequest.tools`) và đưa vào `initial_state`.
- Tool-calls delta được forward theo đúng 2-chunk pattern của OpenAI (đã có `tool_calls_event` trong sse.py).

### 2.5. SSE layer

Giữ nguyên `server/streaming/sse.py`. Thêm callback handler cho 2 event mới:

```python
# trong chat.py::sse_callback
elif event_type == "tool_calls_delta":
    # chunk giữa — emit delta
    await event_queue.put(tool_calls_delta_event(json.loads(content)))
elif event_type == "tool_calls_final":
    # finish_reason chunk
    await event_queue.put(tool_calls_finish_event())
```

Hoặc đơn giản hơn: accumulate trong generate, emit 1 lần ở cuối bằng `tool_calls_event` có sẵn (2 chunks).

### 2.6. Feature flag RAG

**`.env`:**
```
ENABLE_RAG=false
```

**`server/config.py`:** đọc `os.getenv("ENABLE_RAG", "false").lower() == "true"`.

**`server/agent/graph.py`:**
```python
def build_agent_graph(..., enable_rag: bool = False):
    ...
    if enable_rag:
        graph.add_node("rag_search", partial(rag_search, qdrant=qdrant, embedder=embedder))
        graph.add_node("plan_steps", partial(plan_steps, ...))
        # wire code_gen → rag_search → plan_steps → generate
        # wire <other rag intents> → rag_search → generate
    else:
        # all intents → generate directly
```

### 2.7. P0 fixes trong `server/routers/chat.py`

```python
# FIX 1: giữ reference để task không bị GC
run_task = asyncio.create_task(_run_agent())

try:
    # FIX 2: check disconnect mỗi vòng
    while True:
        if await req.is_disconnected():
            break
        try:
            event = await asyncio.wait_for(event_queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            ...
        ...
finally:
    # FIX 3: cancel agent task khi client disconnect / exception
    if not run_task.done():
        run_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await run_task
```

---

## 3. Data model

### 3.1. `ChatRequest` bổ sung

```python
class ChatRequest(BaseModel):
    model_config = {"extra": "allow"}
    messages: list[ChatMessage]
    model: str = ""
    stream: bool = True
    tools: list[dict] | None = None        # NEW — forwarded to vLLM
    tool_choice: str | dict | None = None  # NEW — passthrough
    active_file: str | None = None
    repo_path: str | None = None
```

### 3.2. `AgentState` bổ sung

```python
# server/agent/state.py
client_tools: list[dict]     # tools từ request
tool_choice: str | dict | None
```

### 3.3. Message conversion

```python
# server/routers/chat.py
for msg in request.messages:
    if msg.role == "user":
        messages.append(HumanMessage(content=msg.content or ""))
    elif msg.role == "tool":
        messages.append(ToolMessage(
            content=msg.content or "",
            tool_call_id=msg.tool_call_id or "",
        ))
    elif msg.role == "assistant":
        # NEW: preserve tool_calls on assistant message (Turn 2)
        ai_msg = AIMessage(
            content=msg.content or "",
            additional_kwargs={"tool_calls": msg.tool_calls} if msg.tool_calls else {},
        )
        messages.append(ai_msg)
    else:  # system
        messages.append(SystemMessage(content=msg.content or ""))
```

---

## 4. Files thay đổi

| File | Loại | Ghi chú |
|---|---|---|
| `server/tools/registry.py` | NEW | Tool schemas + `merge_tools()` |
| `server/tools/__init__.py` | NEW | Package init |
| `server/agent/generate.py` | REFACTOR | Native tool-call streaming |
| `server/agent/graph.py` | REFACTOR | Bỏ tool_selector/emit_tool_calls, intent router, RAG flag |
| `server/agent/state.py` | EDIT | Thêm `client_tools`, `tool_choice` |
| `server/routers/chat.py` | REFACTOR | P0 fixes, forward tools, message conversion |
| `server/config.py` | EDIT | `ENABLE_RAG` flag |
| `.env.example` | EDIT | Thêm `ENABLE_RAG=false` |
| `server/streaming/sse.py` | OPTIONAL | Thêm helper nếu cần split tool_calls delta |
| `server/agent/tool_selector.py` | DEAD | Không import, giữ file để tham chiếu — đánh dấu deprecated |
| `server/agent/emit_tool_calls.py` | DEAD | Tương tự |
| `server/agent/rag_search.py` | KEEP | Giữ nguyên, không import khi flag off |
| `server/agent/plan_steps.py` | KEEP | Tương tự |
| `tests/test_new_arch.py` | UPDATE | Bỏ test tool_selector/emit_tool_calls; thêm test merge_tools, native tool-call streaming, intent routing mới |

---

## 5. Testing

### Unit tests
1. `test_registry.py` — `merge_tools()`:
   - Empty client → return server schemas.
   - Client có tool trùng tên → server override.
   - Client có tool riêng → merge cả hai.
2. `test_generate_toolcall.py` — mock vLLM stream:
   - Response có `tool_calls` finish_reason → state có `pending_tool_calls`.
   - Response plain content → state có `draft`.
   - Stream callback được gọi đúng lượt content / tool_calls.
3. `test_graph_routing.py` — với mỗi intent, verify path (RAG off).
4. `test_chat_endpoint.py` — integration:
   - Turn 1 request → stream chunks có content delta.
   - Turn 1 với tool-capable prompt → stream có tool_calls + `[DONE]`.
   - Turn 2 (tool result in messages) → bypass route_context, stream content.
   - Client disconnect → task cancelled.

### Manual
- Start server, run Continue, gửi câu hỏi plain chat → stream OK.
- Gửi câu hỏi cần tool → Continue render tool call, thực thi, gửi lại → final answer.

---

## 5b. Review flow — batch response (V1)

- Endpoint `/review/pr` và các entrypoint code_review: **không streaming**.
- `review_analyze` gọi vLLM với `stream=False`, nhận full response, parse findings.
- `review_format` build markdown/JSON.
- Response trả về client 1 cục (JSON hoặc markdown), không SSE.
- Lý do: findings cần parse/validate toàn bộ trước khi render; streaming nửa vời dễ gây UX tệ và phức tạp cho phía consumer (GitLab/GitHub MR comment upsert).
- V2 có thể thêm streaming progress events sau.

---

## 6. Rollout

1. Merge trên branch `feature/new-architecture`.
2. Default `ENABLE_RAG=false` — behavior mới.
3. Giữ các node RAG / plan_steps / tool_selector / emit_tool_calls trong repo để dễ rollback / reactivate.
4. Khi RAG design xong: flip flag + wire lại trong `build_agent_graph`.

---

## 7. Open questions

- `tool_choice` passthrough: forward nguyên hay server override? → **Forward nguyên** (client biết rõ hơn).
- System prompt có cần inject mặc định khi client không gửi không? → **Không**, giữ stateless. Client tự gửi system prompt.
- `structural_analysis` intent hiện skip RAG — có cần special-case không khi sau này bật RAG? → Giữ như bảng ở 2.2.
- Heartbeat 15s có cần điều chỉnh khi tool_calls turn rất nhanh (< 1s)? → Không, heartbeat chỉ fire khi idle.

---

## 8. Risks

| Risk | Mitigation |
|---|---|
| Model trả tool_calls không đúng schema | vLLM + Qwen3 native support đã verify; registry schema dùng OpenAI spec chuẩn. |
| Continue không forward `tool_calls` delta đúng | Giữ `tool_calls_event` 2-chunk pattern đã test. |
| Client disconnect giữa stream làm leak vLLM connection | P0 fix: cancel task trong `finally`. |
| Dead code (tool_selector, emit_tool_calls) gây confusion | Đánh dấu `# DEPRECATED` ở đầu file. |
| Flag off nhưng vẫn import qdrant/embedder ở startup | app.py vẫn init chúng; không sao vì agent không gọi. |
