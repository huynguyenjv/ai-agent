# Kiến trúc hệ thống — AI Coding Agent

> **Phiên bản:** 2.0.0 — branch `feature/new-architecture`
> **Cập nhật:** 2026-06-04
> **Phạm vi:** Mô tả kiến trúc *thực tế đang có trong code* (thư mục `server/`, `mcp_server/`, `eval/`, `gitlab-review-runner/`).

> ⚠️ **Lưu ý:** File `README.md` mô tả kiến trúc cũ ("JUnit5 Test Generator" với thư mục `agent/` và `server/api.py`). Code hiện tại đã được viết lại hoàn toàn theo "native tool-call edition" trong thư mục `server/agent/` + `server/app.py`. Tài liệu này phản ánh code thực tế, không phải README.

---

## 1. Tổng quan

AI Coding Agent là một **self-hosted coding assistant** chạy phía server, expose API tương thích OpenAI (`/v1/chat/completions`). Hệ thống được thiết kế để:

- Tích hợp với IDE qua giao thức **Continue / Tabby** (client gửi full conversation history mỗi request — server **stateless**).
- Hỗ trợ **native tool-call**: model tự quyết định gọi tool; client (Continue) thực thi tool phía client rồi gửi kết quả về ở turn kế tiếp.
- Dùng **RAG** (Qdrant + embeddings) để cung cấp context từ codebase.
- Dùng **local LLM** qua vLLM (OpenAI-compatible endpoint).
- Orchestration bằng **LangGraph** (state machine có điều kiện rẽ nhánh + retry loop).
- Có pipeline **code review** riêng cho GitLab Merge Request.

### Các "process" / deployable chính

| Thành phần | Vai trò | Transport |
|-----------|---------|-----------|
| **FastAPI server** (`server/`) | Brain chính: graph orchestration, RAG, generate, review | HTTP/SSE |
| **MCP server** (`mcp_server/`) | Tool provider spawn bởi Continue IDE phía client | stdio (MCP) |
| **gitlab-review-runner** | Service CI riêng, fetch diff + post comment lên GitLab | HTTP → server |
| **eval** (`eval/`) | Offline benchmark + feedback analysis | CLI/import |

---

## 2. Sơ đồ kiến trúc tổng thể

```
┌─────────────────────────────────────────────────────────────────────┐
│  IDE Client (Continue / Tabby)                                        │
│   - Gửi full conversation history (stateless server)                  │
│   - Spawn MCP server (stdio) để thực thi tool phía client             │
└───────────┬───────────────────────────────────┬───────────────────────┘
            │ HTTP/SSE                            │ stdio (MCP)
            ▼                                     ▼
┌──────────────────────────────────┐   ┌──────────────────────────────┐
│  FastAPI Server (server/app.py)   │   │  MCP Server (mcp_server/)     │
│  - Auth (X-Api-Key)               │   │  Tools:                       │
│  - Rate limit                     │   │   vtrip_read_file             │
│  - CORS + correlation-id logging  │   │   vtrip_search_symbol         │
│                                   │   │   vtrip_get_project_skeleton  │
│  Routers:                         │   │   vtrip_index_with_deps       │
│   /v1/chat/completions  (chat)    │   │   get_pr_diff / get_mr_note   │
│   /index                (index)   │   │   upsert_mr_comment           │
│   /review/analyze       (review)  │   │  Language plugins (tree-sitter)│
│   /metrics              (metrics) │   │   java/go/python/ts/csharp/hcl │
│   /feedback             (feedback)│   │  Sandbox + hash_store + uploader│
└───────────┬───────────────────────┘   └──────────────┬─────────────────┘
            │                                            │ upload chunks
            ▼                                            ▼
┌──────────────────────────────────────────────────────────────────────┐
│  LangGraph Agent (server/agent/graph.py)                               │
│                                                                        │
│  classify_intent → route_context (5-gate) → planner                    │
│        │ (tool result turn)        │ volatile           │ simple/complex│
│        ▼                           ▼                     ▼              │
│     generate                  reject_volatile    [rag_search] → generate│
│        │                                                  │            │
│        ▼                                                  ▼            │
│   verify_result ──pass(complex)──▶ critic ──▶ post_process ──▶ END     │
│        │ fail (retry)               │ fail (retry ≤2)                  │
│        └─────────────▶ generate ◀───┘                                  │
│                                                                        │
│   [intent=code_review] → review_analyze → review_format → post_process │
└───────┬────────────────────────┬───────────────────────┬──────────────┘
        ▼                         ▼                        ▼
┌───────────────┐      ┌──────────────────┐      ┌──────────────────┐
│ vLLM client   │      │ RAG (server/rag/)│      │ Session / Cache  │
│ (AsyncOpenAI) │      │  Qdrant+Embedder │      │ in-mem / Redis   │
└───────────────┘      └────────┬─────────┘      └──────────────────┘
                                ▼
                        ┌───────────────┐
                        │  Qdrant       │
                        │  (vector DB)  │
                        └───────────────┘

Infra (docker-compose): ai-agent + qdrant + postgres(metrics) + prometheus + grafana
```

---

## 3. FastAPI layer (`server/`)

### 3.1 Bootstrap

- **`main.py`** — entry point: `load_dotenv()` → cấu hình logging (JSON nếu `LOG_FORMAT=json`) → `uvicorn.run(app)` trên `HOST:PORT` (mặc định `0.0.0.0:8000`).
- **`server/app.py`** — `create_app()`:
  - **Lifespan handler** khởi tạo: `QdrantService`, `Embedder`, `AsyncOpenAI` (vLLM client), lưu vào `app.state`. Các dependency khởi tạo "non-blocking" — retry ở request đầu nếu chưa sẵn sàng.
  - **Middleware:** CORS (qua `CORS_ORIGINS`), request logging với **correlation-id** (`X-Correlation-ID`, dùng `contextvar`).
  - **Routers:** `chat`, `index`, `review`, `metrics`, `feedback`.
  - **`/health`** → `{status: ok, version: 2.0.0}`.

### 3.2 Routers (`server/routers/`)

| Router | Endpoint | Mô tả |
|--------|----------|-------|
| `chat.py` | `POST /v1/chat/completions` | Endpoint chính. **Luôn SSE stream**. Auth + rate limit. Build LangGraph rồi `ainvoke`, đẩy event qua queue ra SSE. Hỗ trợ multi-turn qua `conversation_id` (session store). |
| `index.py` | `POST /index` | Nhận chunk đã parse (từ MCP `index_with_deps`) để embed + upsert vào Qdrant. |
| `review.py` | `POST /review/analyze` | **Stateless code review** — nhận diff payload, trả markdown + findings + inline_comments. Không tự nói chuyện với GitLab. |
| `metrics.py` | `GET /metrics` | Prometheus-style metrics. |
| `feedback.py` | `POST /feedback` | Ghi nhận feedback người dùng cho feedback analyzer. |

### 3.3 Cross-cutting (`server/`)

- **`auth.py`** — verify API key qua header `X-Api-Key` / `Authorization`.
- **`rate_limit.py`** — rate limiter theo API key hoặc IP, trả `429 + Retry-After`.
- **`session.py`** — session store (in-memory hoặc Redis-backed) lưu `last_intent`, `active_file`, `mentioned_files`, `context_summary` theo `conversation_id`.
- **`cache.py`** — `LRUCache` thread-safe có TTL (cho embeddings/LLM response; production gợi ý Redis).
- **`continue_compat.py`** — trích `active_file` từ message của Continue.
- **`logging_config.py`** — cấu hình logging + `correlation_id_var`.
- **`utils/`** — `sanitize` (chống prompt injection cho user input & tool output), `content` (normalize content), `json_parser`, `async_io` (async file I/O).
- **`streaming/sse.py`** — helper tạo các SSE event: `thinking_event`, `content_delta_event`, `tool_calls_event`, `tool_error_event`, `heartbeat_comment`, `done_event`.

---

## 4. Agent orchestration — LangGraph (`server/agent/`)

`build_agent_graph()` trong **`graph.py`** dựng `StateGraph(AgentState)`. State là `TypedDict` (`state.py`) chứa toàn bộ dữ liệu chảy qua các node (messages, intent, rag_chunks, draft, tool calls, complexity, critic info...).

### 4.1 Luồng graph

```
classify_intent
  ├─ is_tool_result_turn=True ─▶ generate ─▶ verify_result ─▶ post_process ─▶ END
  └─ else ─▶ route_context
       ├─ volatile_rejected ─▶ reject_volatile ─▶ END
       ├─ intent=code_review ─▶ review_analyze ─▶ review_format ─▶ post_process ─▶ END
       └─ else ─▶ planner
            ├─ (rag_enabled & beneficial) ─▶ rag_search ─▶ generate
            └─ else ─▶ generate
                 generate ─▶ verify_result
                   ├─ fail ─▶ generate (retry)
                   ├─ complex ─▶ critic
                   │     ├─ pass ─▶ post_process
                   │     └─ fail (≤2 retries) ─▶ generate
                   └─ simple ─▶ post_process ─▶ END
```

### 4.2 Các node

| Node | File | Trách nhiệm |
|------|------|-------------|
| `classify_intent` | `classify_intent.py` | Phân loại intent: `code_gen`, `unit_test`, `explain`, `structural_analysis`, `search`, `refine`, `code_review`, `debug`. Phát hiện turn chứa kết quả tool. |
| `route_context` | `route_context.py` | **5-Gate decision flow**: (1) explicit file mention / deictic, (2) freshness keyword → force reindex, (3) volatile data (git diff, runtime log...) → reject, (4-5) RAG lookup. Phát hiện GitLab MR URL cho code review. |
| `planner` | `planner.py`, `plan_steps.py` | Quyết định `complexity` (simple/complex) và task plan. |
| `rag_search` | `rag_search.py` | (chỉ khi `enable_rag`) Embed query + search Qdrant, enrich context. |
| `generate` | `generate.py` | **Core**: build prompt theo intent, merge MCP tools + client tools, stream từ vLLM, gom tool_call deltas, validate/map/dedup tool calls. Có retry (exponential backoff), token budgeting, truncation. |
| `verify_result` | `verify_result.py`, `verify_sources.py` | Kiểm tra kết quả; fail → retry generate. |
| `critic` | `critic.py` | (chỉ task complex) Chấm điểm 0-10, tìm issues; fail → retry generate (tối đa 2 lần). |
| `post_process` | `post_process.py` | Validation warning cuối + format output. |
| `review_analyze` | `review_analyze.py` | Phân tích diff cho code review (OWASP Top 10 + CWE Top 25). |
| `review_format` | `review_format.py` | Format kết quả review → markdown + inline comments. |

### 4.3 Module hỗ trợ agent

- **`context_builder.py`** — assemble context theo priority + token optimization.
- **`summarize.py`** — tóm tắt conversation/context.
- **`parallel_tools.py`** — hỗ trợ chạy song song tool resolution.
- **`task_queue.py`** — hàng đợi task.
- **`rules_loader.py`** — load `config/rules.yaml` (keyword phân loại intent, file extension/suffix, freshness keyword). Hot-reload qua identity của rules dict.
- **`prompts/`** — prompt cho review (`review_system.md`, `review_user_pr.md`, `review_user_file.md`, `review_output_template.md`) — sửa markdown, không cần đổi code.
- **`input_guard.py`** *(mới, untracked)* — phát hiện **prompt injection**: role hijack, instruction override, delimiter injection, jailbreak, data exfiltration, encoding/unicode attack. Có chế độ block/sanitize theo threat level.

### 4.4 System prompt theo intent

`generate.py` chứa `INTENT_PROMPTS` — mỗi intent có persona "senior level" riêng (Senior QA, Principal Engineer, Software Architect, Refactoring Expert, ...). Tất cả đều "respond in user's language (Vietnamese/English)".

---

## 5. Native tool-call

Hệ thống dùng cơ chế **native OpenAI tool-call** thay vì tự parse text.

- **Server-side tool schema** (`generate.py` → `MCP_TOOLS`): `vtrip_read_file`, `vtrip_search_symbol`, `vtrip_get_project_skeleton`, `vtrip_index_with_deps`, `vtrip_run_command`, `vtrip_diff_preview`, `vtrip_apply_edits`, và nhóm git (`vtrip_git_status/diff/log/commit/branch`).
- **Merge tools:** MCP tools + client tools (MCP ưu tiên). Forward `tool_choice`.
- **Tolerance cho model khác:** `TOOL_NAME_MAP` (vd `read_file`→`vtrip_read_file`, `grep`→`vtrip_search_symbol`) và `ARG_NAME_MAP` (vd `path`→`file_path`).
- **An toàn:** validate required args, **dedup** tool call (theo normalized key), giới hạn `MAX_TOOL_TURNS` (mặc định 5), strip `<tool_call>` tag thừa.
- **Round-trip:** model trả `pending_tool_calls` → client (Continue) thực thi → gửi `role:"tool"` results về → graph chạy lại nhánh `is_tool_result_turn`.

---

## 6. MCP Server (`mcp_server/`)

Server stdio (giao thức **Model Context Protocol**) được Continue IDE spawn phía client. Cấu hình qua env do Continue inject (`REPO_PATH`, `SERVER_URL`, `API_KEY`, `TOKEN_BUDGET`, `DEPTH_DEFAULT`).

### 6.1 Tools expose

| Tool | Mô tả |
|------|-------|
| `vtrip_read_file` | Đọc range dòng trong file (fresh content). |
| `vtrip_search_symbol` | Tìm class/function/method theo tên → file + line. |
| `vtrip_get_project_skeleton` | Tổng quan cấu trúc repo (compact). |
| `vtrip_index_with_deps` | Parse file + dependency (BFS depth ≤3), upload chunk thay đổi lên server để embed. |
| `get_pr_diff` / `get_mr_note` / `upsert_mr_comment` | Thao tác GitLab MR (cho luồng review). |

### 6.2 Plugin theo ngôn ngữ (`plugins/`)

Parser dựa **tree-sitter**, đăng ký qua `PluginRegistry`:
`JavaPlugin`, `GoPlugin`, `PythonPlugin`, `TypeScriptPlugin`, `CSharpPlugin`, `HCLPlugin`, và `FallbackPlugin` (mặc định). Base interface ở `plugins/base.py`.

### 6.3 Module phụ trợ

- **`hash_store.py`** — track hash chunk để chỉ upload phần thay đổi (incremental).
- **`dep_classifier.py`** — phân loại dependency project-local vs external.
- **`uploader.py`** — upload chunk lên `SERVER_URL/index`.
- **`token_budget.py`** — giới hạn token khi index.
- **`sandbox.py`** *(mới, untracked)* — **CommandSandbox**: whitelist command (test/lint/build/git-read/file-read/search), chặn `DANGEROUS_PATTERNS` (rm -rf, sudo, curl|sh, fork bomb...), giới hạn timeout/output, chống path traversal, **audit log**. Git chỉ cho phép subcommand read-only.
- **`tools_review.py`** — implement thao tác GitLab MR cho MCP.

---

## 7. RAG subsystem (`server/rag/`)

| Module | Trách nhiệm |
|--------|-------------|
| `embedder.py` | `Embedder` — model `all-MiniLM-L6-v2` (HuggingFace), lazy-load. |
| `qdrant_client.py` | `QdrantService` — kết nối Qdrant, `ensure_collection`, upsert/search. |
| `chunking.py` | Chia code thành chunk. |
| `context_retrieval.py` | **Parent-child retrieval** — enrich chunk với context xung quanh (N dòng trước/sau), tìm class/function bao quanh, build context prompt theo giới hạn ký tự (có file cache). |
| `hash_verifier.py` | Xác thực hash chunk (freshness). |
| `hyde.py` | **HyDE** — hypothetical document embedding để cải thiện recall. |
| `query_expand.py` | Mở rộng query. |
| `reranker.py` | Re-rank kết quả search. |

**Vector DB:** Qdrant (port 6333). **Chu trình index:** MCP parse file → upload chunk → `/index` embed → upsert Qdrant. **Chu trình query:** `rag_search` node embed query → search → enrich → đưa vào context của `generate`.

---

## 8. Code Review pipeline (GitLab)

Hai đường vào:

1. **Qua chat** (Continue): intent `code_review` → graph chạy `review_analyze → review_format`. Continue gửi nội dung file/diff trong message.
2. **Qua `gitlab-review-runner`** (service riêng, chạy trong CI):
   - Bắt event MR, fetch diff qua GitLab API (`gitlab_client.py`).
   - POST `/review/analyze` (server **stateless**, không tự nói chuyện với GitLab).
   - Nhận markdown + inline_comments → post comment lên MR.
   - Marker `<!-- AI_REVIEW_MARKER:v1 -->` để update tại chỗ; inline tag `<!-- AI_REVIEW_INLINE:v1 -->`.
   - Frameworks: **OWASP Top 10 (2021)** + **CWE Top 25 (2024)** + lint rules.
   - Prompt ở `server/agent/prompts/` (sửa markdown, không cần đổi code).

`gitlab-review-runner/` có Dockerfile + CI example riêng (`ai_client.py`, `gitlab_client.py`, `config.py`, `main.py`).

---

## 9. Observability & Metrics (`server/metrics/`)

- **`counter.py`** — `RequestTimer` đo TTFT (time-to-first-token), tổng thời gian, token in/out, intent, success/error; `MetricsCounter` ghi nhận.
- **`prometheus.py`** — export Prometheus metric (`record_request`, `record_tokens`, `ACTIVE_REQUESTS`).
- **`models.py`** — schema metric, lưu Postgres `metrics_db` (`init-db/001_schema.sql`).
- **Logging** — JSON structured + correlation-id mỗi request.
- **Stack giám sát** (docker-compose): Prometheus (scrape `/metrics`) + Grafana (dashboard `deploy/grafana/dashboards/ai-agent.json`).

---

## 10. Evaluation & Feedback (`eval/`, `server/feedback_analyzer.py`)

- **`eval/benchmark.py`** — offline benchmark suite (17+ case: code gen, unit test, review, explain, refactor, search, edge case). Scoring theo keyword + length + intent correctness; so sánh baseline để phát hiện regression.
- **`server/feedback_analyzer.py`** — phân loại feedback (`positive`/`negative`/`correction`/`retry`), phát hiện pattern (high retry rate, keyword issue...), gợi ý cải thiện prompt, tính satisfaction rate.

---

## 11. Bảo mật (Phase 10 — đang triển khai)

| Lớp | Cơ chế |
|-----|--------|
| **Prompt injection** | `server/agent/input_guard.py` — pattern + unicode/homoglyph + threat level + block/sanitize. |
| **Input/output sanitize** | `server/utils/sanitize.py` — `sanitize_user_input`, `sanitize_tool_output` (gọi trong `chat.py` khi convert message). |
| **Command sandbox** | `mcp_server/sandbox.py` — whitelist + dangerous-pattern block + timeout + path traversal + audit log. |
| **Auth** | API key (`server/auth.py`). |
| **Rate limit** | `server/rate_limit.py`. |

> Các file `input_guard.py`, `sandbox.py` và test tương ứng hiện ở trạng thái **untracked** (chưa commit) — thuộc Phase 10 trong `docs/improvement-plan.md`.

---

## 12. Hạ tầng & Deployment

**`docker-compose.yml`** dựng:

| Service | Image | Port (bind localhost) | Vai trò |
|---------|-------|----------------------|---------|
| `ai-agent` | build từ `Dockerfile` | `${PORT:-8080}` | App chính (limit 2G RAM, healthcheck `/health`). |
| `qdrant` | `qdrant/qdrant` | `6333` | Vector DB (volume `qdrant_data`). |
| `postgres` | `postgres:16-alpine` | `5432` | DB metrics (`metrics_db`, init từ `init-db/`). |
| `prometheus` | `prom/prometheus` | `9090` | Scrape metrics (retention 15d). |
| `grafana` | `grafana/grafana` | `3000` | Dashboard. |

- **vLLM chạy ngoài compose** (bare-metal hoặc network khác), trỏ qua `VLLM_BASE_URL` (vd `http://host.docker.internal:8000/v1`).
- HF cache mount volume `hf_cache` để embedder không tải lại model.
- `deploy/nginx.conf` cho reverse proxy; `deploy/certs` cho GitLab self-signed CA.

### Biến môi trường chính

| Biến | Mặc định | Mô tả |
|------|----------|-------|
| `QDRANT_URL` | `http://127.0.0.1:6333` | Qdrant endpoint |
| `VLLM_BASE_URL` | `http://localhost:8000/v1` | vLLM endpoint |
| `VLLM_MODEL` | `qwen2.5-coder` | Model id |
| `HOST` / `PORT` | `0.0.0.0` / `8000` | Bind app |
| `ENABLE_RAG` | `true` | Bật RAG |
| `MAX_TOOL_TURNS` | `5` | Giới hạn vòng tool-call |
| `MAX_INPUT_TOKENS` | `24000` | Token budget input |
| `LLM_MAX_RETRIES` | `3` | Retry vLLM |
| `CORS_ORIGINS` | `*` | CORS |
| `LOG_FORMAT` | (text) | `json` để structured logging |
| `DATABASE_URL` | postgres metrics | Kết nối metrics DB |

---

## 13. Cấu trúc thư mục (rút gọn)

```
ai-agent/
├── main.py                      # Entry point → uvicorn
├── docker-compose.yml / Dockerfile
├── requirements.txt
│
├── server/                      # FastAPI app
│   ├── app.py                   # create_app + lifespan + middleware
│   ├── auth.py / rate_limit.py / session.py / cache.py
│   ├── continue_compat.py / logging_config.py / feedback_analyzer.py
│   ├── routers/                 # chat, index, review, metrics, feedback
│   ├── agent/                   # LangGraph orchestration
│   │   ├── graph.py             # build_agent_graph (state machine)
│   │   ├── state.py             # AgentState (TypedDict)
│   │   ├── classify_intent.py / route_context.py / planner.py
│   │   ├── generate.py          # native tool-call + intent prompts
│   │   ├── verify_result.py / critic.py / post_process.py
│   │   ├── review_analyze.py / review_format.py / rag_search.py
│   │   ├── context_builder.py / summarize.py / parallel_tools.py
│   │   ├── rules_loader.py / task_queue.py / verify_sources.py
│   │   ├── input_guard.py       # (untracked) prompt-injection guard
│   │   └── prompts/             # review prompt markdown
│   ├── rag/                     # embedder, qdrant_client, chunking,
│   │                            # context_retrieval, hyde, reranker, ...
│   ├── metrics/                 # counter, prometheus, models
│   ├── streaming/sse.py
│   └── utils/                   # sanitize, content, json_parser, async_io
│
├── mcp_server/                  # MCP stdio server (client-side)
│   ├── server.py                # tool registry + dispatch
│   ├── tools.py / tools_indexer.py / tools_review.py / tools_refactor.py
│   ├── plugins/                 # java/go/python/ts/csharp/hcl/fallback
│   ├── hash_store.py / dep_classifier.py / uploader.py / token_budget.py
│   └── sandbox.py               # (untracked) command sandbox
│
├── gitlab-review-runner/        # CI service review MR (riêng biệt)
├── eval/                        # benchmark.py (offline eval)
├── config/                      # rules.yaml
├── deploy/                      # nginx, prometheus, grafana, certs
├── init-db/                     # 001_schema.sql (metrics DB)
├── docs/                        # plans + tài liệu (file này)
├── report/                      # báo cáo từng phase
└── tests/                       # unit tests
```

---

## 14. Quyết định thiết kế then chốt

1. **Stateless server** — client gửi full history mỗi request → dễ scale ngang, không cần sticky session (session store chỉ là tối ưu multi-turn).
2. **Native tool-call thay vì text parsing** — tận dụng khả năng function-calling của model, có lớp tolerance (name/arg mapping) cho nhiều model khác nhau.
3. **LangGraph state machine** — tách rõ từng node, dễ thêm bước (planner, critic, rag_search), có retry loop và conditional routing.
4. **Tách MCP server khỏi FastAPI** — tool đọc file/symbol chạy phía client (gần source code), server chỉ lo LLM + RAG + orchestration.
5. **Code review stateless** — server không bao giờ gọi GitLab; runner riêng lo I/O với GitLab → tách concern, dễ test.
6. **Always-SSE streaming** — kể cả `stream:false`, để đồng nhất pipeline và phản hồi sớm (TTFT thấp).

---

## 15. Hạn chế hiện tại & hướng phát triển

Theo `docs/improvement-plan.md` (current score ~7.5/10):

- **State persistence:** session/cache phần lớn in-memory → Phase 11 thêm Redis, circuit breaker, deep health check.
- **Scalability:** single instance, chưa có queue/K8s/HPA (Phase 12 — chưa làm).
- **RAG:** chưa có hybrid search (BM25+semantic), delta indexing, cached AST (Phase 13).
- **Observability:** chưa có OpenTelemetry/Jaeger tracing (Phase 14).
- **Multi-agent:** hiện single-agent graph; chưa có message bus / specialized agents (Phase 15).
- **Security:** sandbox + input guard mới ở dạng untracked, chưa có secret scanning/audit log đầy đủ (Phase 10 đang dở).

---

*Tài liệu sinh dựa trên đọc trực tiếp source code branch `feature/new-architecture` ngày 2026-06-04.*
```
