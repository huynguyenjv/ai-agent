# AI Agent Architecture Overview

**Version:** 2.0.0  
**Date:** 2026-05-28  
**Stack:** Python 3.12 + FastAPI + LangGraph + vLLM + Qdrant

---

## 1. System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLIENT LAYER                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │  Continue    │  │  Cline IDE   │  │  Tabby IDE   │  │  Custom App  │    │
│  │  IDE Plugin  │  │              │  │              │  │  (OpenAI SDK)│    │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘    │
│         │                 │                 │                 │             │
│         └─────────────────┴────────┬────────┴─────────────────┘             │
│                                    │ HTTP (OpenAI-compatible)               │
└────────────────────────────────────┼────────────────────────────────────────┘
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MCP SERVER (Client Machine)                        │
│                              mcp_server/                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  FastMCP Server (stdio)                                             │    │
│  │  • vtrip_read_file      - Read file content                         │    │
│  │  • vtrip_search_symbol  - Find symbols in codebase                  │    │
│  │  • vtrip_get_project_skeleton - Project structure overview          │    │
│  │  • vtrip_index_with_deps - Index file + dependencies                │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                         │
│  ┌─────────────────────────────────┼─────────────────────────────────────┐  │
│  │  Language Plugins (tree-sitter based)                                 │  │
│  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐   │  │
│  │  │ Java   │ │ Python │ │ TypeScript│ │ Go   │ │ C#     │ │ HCL    │   │  │
│  │  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘ └────────┘   │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │ HTTP
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CLOUD VM (FastAPI Server)                          │
│                                 server/                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  FastAPI Application (server/app.py)                                │    │
│  │                                                                     │    │
│  │  Endpoints:                                                         │    │
│  │  • POST /v1/chat/completions  ─── OpenAI-compatible chat (SSE)      │    │
│  │  • POST /review/analyze       ─── Code review (stateless)           │    │
│  │  • POST /index                ─── Chunk ingestion                   │    │
│  │  • GET  /metrics              ─── Prometheus metrics                │    │
│  │  • GET  /health               ─── Health check                      │    │
│  │                                                                     │    │
│  │  Middleware:                                                        │    │
│  │  • CORS                                                             │    │
│  │  • Request logging with correlation ID                              │    │
│  │  • API key authentication                                           │    │
│  └──────────────────────────────────┬──────────────────────────────────┘    │
│                                     │                                        │
│  ┌──────────────────────────────────▼──────────────────────────────────┐    │
│  │  LangGraph Agent (server/agent/)                                    │    │
│  │                                                                     │    │
│  │  ┌─────────────────┐                                                │    │
│  │  │ classify_intent │ ─── Intent detection (LLM + Python fallback)   │    │
│  │  └────────┬────────┘                                                │    │
│  │           │                                                         │    │
│  │           ├─── is_tool_result_turn ───┐                             │    │
│  │           │                           │                             │    │
│  │  ┌────────▼────────┐                  │                             │    │
│  │  │  route_context  │ ─── 5-Gate decision flow                       │    │
│  │  └────────┬────────┘                  │                             │    │
│  │           │                           │                             │    │
│  │     ┌─────┼─────┐                     │                             │    │
│  │     │     │     │                     │                             │    │
│  │     ▼     ▼     ▼                     ▼                             │    │
│  │  ┌─────┐ ┌──────────────┐ ┌────────────────┐                        │    │
│  │  │reject│ │review_analyze│ │    generate    │ ─── LLM + tool calls  │    │
│  │  └──┬──┘ └──────┬───────┘ └───────┬────────┘                        │    │
│  │     │           │                 │                                 │    │
│  │     │    ┌──────▼───────┐         │                                 │    │
│  │     │    │review_format │         │                                 │    │
│  │     │    └──────┬───────┘         │                                 │    │
│  │     │           │                 │                                 │    │
│  │     │           └────────┬────────┘                                 │    │
│  │     │                    │                                          │    │
│  │     │           ┌────────▼────────┐                                 │    │
│  │     │           │  post_process   │ ─── Lightweight validation      │    │
│  │     │           └────────┬────────┘                                 │    │
│  │     │                    │                                          │    │
│  │     └────────────────────┴──────────► END                           │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                     │                                        │
│  ┌──────────────────────────────────┼──────────────────────────────────┐    │
│  │  Supporting Services                                                │    │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐   │    │
│  │  │ SSE Stream  │ │   Metrics   │ │   Session   │ │    Utils    │   │    │
│  │  │ streaming/  │ │  metrics/   │ │  session.py │ │   utils/    │   │    │
│  │  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                     │                                        │
│  ┌──────────────────────────────────┼──────────────────────────────────┐    │
│  │  RAG System (server/rag/)                                           │    │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐     │    │
│  │  │    Embedder     │  │  QdrantService  │  │  HashVerifier   │     │    │
│  │  │ (dense+sparse)  │  │ (hybrid search) │  │ (change detect) │     │    │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              │                       │                       │
              ▼                       ▼                       ▼
┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐
│      vLLM Server    │  │       Qdrant        │  │   GitLab Runner     │
│  (qwen2.5-coder)    │  │   (Vector DB)       │  │ gitlab-review-runner│
│                     │  │                     │  │                     │
│  • Chat completions │  │  • Hybrid search    │  │  • MR event handler │
│  • Tool calling     │  │  • Dense (384-dim)  │  │  • Fetch diff       │
│  • Streaming        │  │  • Sparse (BM25)    │  │  • Post comments    │
└─────────────────────┘  └─────────────────────┘  └─────────────────────┘
```

---

## 2. Component Details

### 2.1 MCP Server (`mcp_server/`)

**Purpose:** Runs on client machine, provides codebase access tools to the AI.

```
mcp_server/
├── server.py              # FastMCP stdio server
├── tools.py               # Tool definitions
├── tools_indexer.py       # Index with dependencies
├── tools_review.py        # Code review tools
├── models.py              # CodeChunk, ExtractionMode
├── token_budget.py        # Token estimation
├── hash_store.py          # File hash tracking
├── uploader.py            # Chunk upload to server
├── dep_classifier.py      # Dependency classification
└── plugins/               # Language parsers
    ├── base.py            # LanguagePlugin ABC + _walk()
    ├── registry.py        # Plugin registry
    ├── java_plugin.py
    ├── python_plugin.py
    ├── typescript_plugin.py
    ├── go_plugin.py
    ├── csharp_plugin.py
    ├── hcl_plugin.py
    └── fallback.py        # Generic fallback
```

**Tools exposed:**
| Tool | Description |
|------|-------------|
| `vtrip_read_file` | Read file with line range |
| `vtrip_search_symbol` | Find class/function/method |
| `vtrip_get_project_skeleton` | Project structure |
| `vtrip_index_with_deps` | Parse + upload chunks |

---

### 2.2 FastAPI Server (`server/`)

**Purpose:** Cloud VM, handles all API requests.

```
server/
├── app.py                 # FastAPI app factory + lifespan
├── auth.py                # API key verification
├── session.py             # In-memory session store
├── logging_config.py      # Structured logging
├── continue_compat.py     # Continue IDE compatibility
│
├── routers/               # API endpoints
│   ├── chat.py            # /v1/chat/completions
│   ├── review.py          # /review/analyze
│   ├── index.py           # /index
│   └── metrics.py         # /metrics
│
├── agent/                 # LangGraph nodes
│   ├── graph.py           # Graph definition
│   ├── state.py           # AgentState TypedDict
│   ├── classify_intent.py # Intent classification
│   ├── route_context.py   # 5-Gate routing
│   ├── generate.py        # LLM generation + tools
│   ├── post_process.py    # Validation
│   ├── review_analyze.py  # PR review analysis
│   ├── review_format.py   # Markdown formatting
│   ├── rules_loader.py    # Hot-reload rules.yaml
│   ├── rag_search.py      # RAG node (optional)
│   ├── plan_steps.py      # Planning node (optional)
│   └── prompts/           # Prompt templates
│
├── streaming/             # SSE protocol
│   └── sse.py             # Event formatters
│
├── rag/                   # Vector search
│   ├── qdrant_client.py   # Hybrid search
│   ├── embedder.py        # Dense + sparse
│   └── hash_verifier.py   # Change detection
│
├── metrics/               # Observability
│   ├── models.py          # RequestMetrics, AggregatedMetrics
│   └── counter.py         # MetricsCounter + storage
│
└── utils/                 # Shared utilities
    ├── content.py         # normalize_content()
    └── json_parser.py     # parse_json_safe()
```

---

### 2.3 LangGraph Agent Flow

```
                    ┌─────────────────────┐
                    │   classify_intent   │
                    │                     │
                    │  • LLM classifier   │
                    │  • Python fallback  │
                    │  • Tool result check│
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
    is_tool_result      route_context      (other)
              │                │                │
              │    ┌───────────┼───────────┐    │
              │    │           │           │    │
              │    ▼           ▼           ▼    │
              │  volatile   code_review  normal │
              │    │           │           │    │
              │    ▼           ▼           ▼    │
              │  reject   review_analyze generate
              │    │           │           │    │
              │    │    review_format      │    │
              │    │           │           │    │
              │    │           └─────┬─────┘    │
              │    │                 │          │
              │    │                 ▼          │
              │    │          post_process      │
              │    │                 │          │
              └────┴─────────────────┴──────────┘
                               │
                               ▼
                              END
```

**Intents supported:**
| Intent | Description | Tools |
|--------|-------------|-------|
| `unit_test` | Generate unit tests | read_file, index_with_deps |
| `code_review` | Review code/PR | read_file, get_pr_diff |
| `code_gen` | Implement features | read_file, index_with_deps |
| `explain` | Explain code | read_file |
| `search` | Find symbols | search_symbol |
| `debug` | Fix bugs | read_file, search_symbol |
| `refine` | Refactor code | read_file, index_with_deps |
| `structural_analysis` | Architecture overview | get_project_skeleton |

---

### 2.4 GitLab Review Runner (`gitlab-review-runner/`)

**Purpose:** Runs in GitLab CI, triggers code reviews on MR events.

```
gitlab-review-runner/
├── main.py            # MR event handler
├── config.py          # Environment config
├── gitlab_client.py   # GitLab API client
├── ai_client.py       # AI server client
└── Dockerfile
```

**Flow:**
```
GitLab MR Event
      │
      ▼
┌─────────────┐
│ main.py     │
│             │
│ 1. Get diff │──────► GitLab API
│ 2. Analyze  │──────► POST /review/analyze
│ 3. Comment  │──────► GitLab API
└─────────────┘
```

---

## 3. Data Flow

### 3.1 Chat Request Flow

```
Continue IDE                    Server                         vLLM
     │                            │                              │
     │  POST /v1/chat/completions │                              │
     │  {messages, tools}         │                              │
     ├───────────────────────────►│                              │
     │                            │                              │
     │                            │  classify_intent             │
     │                            │  route_context               │
     │                            │                              │
     │                            │  generate()                  │
     │                            ├─────────────────────────────►│
     │                            │  stream chunks               │
     │                            │◄─────────────────────────────┤
     │                            │                              │
     │  SSE: content deltas       │                              │
     │◄───────────────────────────┤                              │
     │                            │                              │
     │  SSE: tool_calls (if any)  │                              │
     │◄───────────────────────────┤                              │
     │                            │                              │
     │  Execute tools locally     │                              │
     │                            │                              │
     │  POST (Turn 2 with results)│                              │
     ├───────────────────────────►│                              │
     │                            │                              │
```

### 3.2 Code Review Flow

```
GitLab CI                       Server                         vLLM
     │                            │                              │
     │  POST /review/analyze      │                              │
     │  {diff, pr_context}        │                              │
     ├───────────────────────────►│                              │
     │                            │                              │
     │                            │  Split diff by file          │
     │                            │                              │
     │                            │  Parallel LLM calls          │
     │                            ├─────────────────────────────►│
     │                            │◄─────────────────────────────┤
     │                            │                              │
     │                            │  Merge findings              │
     │                            │  Format markdown             │
     │                            │                              │
     │  {markdown, findings,      │                              │
     │   inline_comments}         │                              │
     │◄───────────────────────────┤                              │
     │                            │                              │
```

---

## 4. Configuration

### 4.1 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| **Server** |
| `SERVER_PORT` | `8080` | API server port |
| `CORS_ORIGINS` | `*` | Allowed origins |
| `API_KEY` | - | Authentication key |
| **vLLM** |
| `VLLM_BASE_URL` | `http://localhost:8000/v1` | vLLM server |
| `VLLM_MODEL` | `qwen2.5-coder` | Model name |
| **Qdrant** |
| `QDRANT_URL` | `http://127.0.0.1:6333` | Qdrant server |
| **Agent** |
| `MAX_TOOL_TURNS` | `5` | Max tool rounds |
| `MAX_INPUT_TOKENS` | `24000` | Input token limit |
| `ENABLE_RAG` | `false` | Enable RAG nodes |
| **Review** |
| `REVIEW_LLM_TIMEOUT_SECS` | `60` | LLM timeout |
| `REVIEW_MAX_DIFF_CHARS_PER_FILE` | `40000` | Max diff size |
| `AI_REVIEWER_MARKER` | `AI_REVIEW_MARKER:v1` | Comment marker |
| **GitLab** |
| `GITLAB_TOKEN` | **Required** | GitLab API token |
| `AI_SERVER_URL` | **Required** | AI server URL |

### 4.2 Rules Configuration (`config/rules.yaml`)

```yaml
intents:
  - name: unit_test
    keywords_vi: ["viết test", "tạo test", ...]
    keywords_en: ["write test", "create test", ...]
    tools: ["vtrip_read_file", "vtrip_index_with_deps"]
    requires_file: true
    priority: 1
  # ... more intents

freshness_keywords:
  - "vừa sửa"
  - "just changed"
  # ...

file_extensions:
  - .java
  - .py
  # ...

confidence_threshold: 0.7
rules_reload_interval_seconds: 30
```

---

## 5. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **LangGraph over custom orchestrator** | Declarative, easy to extend, built-in state management |
| **Stateless review endpoint** | Runner handles GitLab API, server stays simple |
| **MCP tools on client** | Fresh codebase access, no server-side cloning |
| **Hybrid RAG (dense + sparse)** | Better recall for code search |
| **Hot-reload rules.yaml** | Change routing without restart |
| **SSE streaming** | Real-time token output, compatible with Continue |
| **Lightweight validation** | Catch common issues without blocking |

---

## 6. Deployment

```yaml
# docker-compose.yml
services:
  ai-agent:
    build: .
    ports: ["8080:8080"]
    environment:
      - VLLM_BASE_URL=http://vllm:8000/v1
      - QDRANT_URL=http://qdrant:6333
    depends_on: [vllm, qdrant]

  vllm:
    image: vllm/vllm-openai
    ports: ["8000:8000"]
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]

  qdrant:
    image: qdrant/qdrant
    ports: ["6333:6333"]
    volumes:
      - qdrant_data:/qdrant/storage
```

---

## 7. Metrics & Observability

**Tracked metrics:**
- Request count by intent
- Token usage (input/output)
- Latency (TTFT, total, p50/p95/p99)
- Tool calls per request
- Validation warnings
- Success/failure rate

**Endpoints:**
- `GET /metrics` - Prometheus format
- `GET /health` - Health check

---

*Architecture documented: 2026-05-28*
