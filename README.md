# AI Agent — JUnit5 Test Generator

Self-hosted AI agent that generates **JUnit5 + Mockito** unit tests for Java services.  
Uses RAG (Qdrant) + local LLM (vLLM) and is orchestrated by **LangGraph** for modular, maintainable, and extensible workflows.

## Architecture

```
Continue / Tabby IDE
        │
        ▼
┌──────────────────────────────────────────────────────────┐
│  FastAPI Server  (OpenAI-compatible /v1/chat/completions) │
│  server/api.py — toggleable graph vs legacy backend       │
└──────────┬────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────┐
│  LangGraph Orchestrator (agent/graph.py)                  │
│  Supervisor → intent routing → SubGraph                   │
│                                                           │
│  ┌─────────────────────────────────────────────────────┐  │
│  │  UnitTest SubGraph (agent/subgraphs/unit_test.py)   │  │
│  │                                                     │  │
│  │  retrieve → check_strategy                          │  │
│  │    → [single_pass] → build_prompt                   │  │
│  │    → [two_phase]   → analyze → build_prompt         │  │
│  │  → call_llm → validate                              │  │
│  │    → [pass]   → human_review → save_result → END    │  │
│  │    → [fail]   → repair → validate (loop)            │  │
│  └─────────────────────────────────────────────────────┘  │
└──────┬──────────────┬────────────────┬───────────────────┘
       │              │                │
       ▼              ▼                ▼
┌────────────┐ ┌────────────┐  ┌──────────────┐
│ RAG Client │ │ vLLM Client│  │ Intelligence │
│  (Qdrant)  │ │  (Qwen2.5) │  │ (Graph+Symbol│
│  rag/      │ │  vllm/     │  │  Analysis)   │
└──────┬─────┘ └────────────┘  └──────────────┘
       │
       ▼
┌────────────┐
│  Indexer   │
│ (tree-sitter│
│  → Qdrant) │
└────────────┘
```

## Key Features

- **LangGraph orchestration** — modular node-based pipeline with conditional routing, retry loops, and checkpointed state
- **OpenAI-compatible API** — drop-in for Continue IDE and Tabby IDE
- **Two-Phase Strategy** — auto-detects complex services (complexity ≥ threshold) and uses LLM analysis before generation
- **7-pass validation** — structural, forbidden patterns, required annotations, AAA, quality metrics, anti-patterns, RAG cross-check
- **3-level escalating repair** — targeted → reasoning → regenerate, with FailureMemory
- **Human-in-the-loop review** — optional `interrupt()` for CI/CD pipelines (auto-approve for IDE clients)
- **Streaming support** — per-node SSE streaming with phase progress
- **Smart context assembly** — priority-based snippet selection + token optimization + domain registry
- **Graph intelligence** — file-level dependency graph + global symbol table
- **Session memory** — in-memory or Redis-backed conversation persistence
- **Event bus + metrics** — decoupled observability

## Quick Start

### 1. Prerequisites

- Python 3.11+
- Docker (for Qdrant)
- vLLM server with a code model (e.g. `Qwen/Qwen2.5-Coder-32B-Instruct-AWQ`)

### 2. Setup

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
.\venv\Scripts\Activate.ps1     # Windows PowerShell

pip install -r requirements.txt

cp env.example .env
# Edit .env — set QDRANT_HOST, VLLM_BASE_URL, JAVA_REPO_PATH
```

### 3. Start Dependencies

```bash
docker-compose up -d qdrant      # Development
docker-compose up -d             # Full stack (production)
```

### 4. Index & Run

```bash
python main.py

# Index your Java codebase
curl -X POST http://localhost:8080/reindex \
  -H "Content-Type: application/json" \
  -d '{"repo_path": "/path/to/java/repo"}'
```

### 5. Connect IDE

**Continue IDE** — add to `config.json`:
```json
{
  "models": [{
    "title": "AI Agent",
    "provider": "openai",
    "model": "Qwen/Qwen2.5-Coder-32B-Instruct-AWQ",
    "apiBase": "http://localhost:8080/v1",
    "apiKey": "token-abc123"
  }]
}
```

**Tabby IDE** — see [TABBY_SETUP.md](TABBY_SETUP.md).

## LangGraph Nodes

Each node wraps a battle-tested existing module:

| Node | Module | Purpose |
|------|--------|---------|
| `retrieve` | `rag/client.py` + `context/` | Fetch RAG context + parallel dependency resolution |
| `check_strategy` | Complexity calc | Route single-pass vs two-phase |
| `analyze` | `two_phase_strategy.py` | Phase 1 LLM analysis (two-phase only) |
| `build_prompt` | `agent/prompt.py` | Construct system + user prompts |
| `call_llm` | `vllm/client.py` | Generate code via vLLM |
| `validate` | `agent/validation.py` | 7-pass validation pipeline |
| `repair` | `agent/repair.py` | 3-level escalating repair |
| `human_review` | LangGraph `interrupt()` | Optional human approval |
| `save_result` | `agent/memory.py` | Persist result + update session |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat (IDE integration) |
| `GET`  | `/v1/models` | List available models |
| `POST` | `/review/pr` | **Code review MR** (GitLab V1) — fetch diff, review OWASP/CWE, auto-post/update MR comment. See [curl examples](docs/code-review-curl-examples.md) |
| `POST` | `/review/{run_id}` | Submit human review (LangGraph only) |
| `GET`  | `/runs/{run_id}` | Poll run status (LangGraph only) |
| `POST` | `/generate-test` | Native test generation |
| `POST` | `/refine-test` | Refine with feedback |
| `POST` | `/reindex` | Index/re-index a Java repo |
| `GET`  | `/health` | Health check |
| `GET`  | `/metrics` | Prometheus-style metrics |

## Code Review Pipeline (GitLab)

- `.gitlab-ci.yml` triggers `POST /review/pr` on MR events.
- Frameworks: **OWASP Top 10 (2021)** + **CWE Top 25 (2024)** + language lint rules.
- Output: markdown comment on MR with marker `<!-- AI_REVIEW_MARKER:v1 -->`, update in-place on new commits, keeps last 3 review summaries in collapsible `<details>`.
- Prompts in [server/agent/prompts/](server/agent/prompts/) (system + user + output template) — edit markdown files, no code change needed.
- Config env: see [.env.example](.env.example) (`GITLAB_URL`, `GITLAB_TOKEN`, `GITLAB_CA_BUNDLE`, `REVIEW_TIMEOUT_SECS`, ...).
- Full docs: [docs/code-review-curl-examples.md](docs/code-review-curl-examples.md).

## API Examples

### Non-streaming
```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ai-agent",
    "stream": false,
    "messages": [{
      "role": "user",
      "content": "Generate unit tests for UserService\n\n```src/main/java/com/example/UserService.java\npublic class UserService { }\n```"
    }]
  }'
```

### Streaming (SSE)
```bash
curl -N -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ai-agent",
    "stream": true,
    "messages": [{
      "role": "user",
      "content": "Generate tests for OrderService\n\n```src/main/java/com/example/OrderService.java\npublic class OrderService { }\n```"
    }]
  }'
```

## Project Structure

```
ai-agent/
├── main.py                     # Entry point (uvicorn)
├── requirements.txt
├── Dockerfile / docker-compose.yml
│
├── agent/                      # Core orchestration
│   ├── graph.py                # LangGraph factory (supervisor + subgraphs + checkpointer)
│   ├── graph_adapter.py        # GraphOrchestrator — drop-in for AgentOrchestrator
│   ├── state.py                # LangGraph state schemas (AgentState, UnitTestState)
│   ├── supervisor.py           # Regex-based intent classifier
│   ├── nodes/                  # LangGraph node functions
│   │   ├── retrieve.py         # RAG + ContextBuilder
│   │   ├── check_strategy.py   # Complexity routing
│   │   ├── analyze.py          # Two-Phase Phase 1
│   │   ├── build_prompt.py     # Prompt construction
│   │   ├── call_llm.py         # vLLM generation
│   │   ├── validate.py         # 7-pass validation
│   │   ├── repair.py           # 3-level escalating repair
│   │   ├── human_review.py     # interrupt() for human approval
│   │   └── save_result.py      # Session memory update
│   ├── subgraphs/
│   │   └── unit_test.py        # UnitTest StateGraph definition
│   ├── orchestrator.py         # Legacy orchestrator (fallback)
│   ├── prompt.py               # PromptBuilder
│   ├── validation.py           # ValidationPipeline (7-pass)
│   ├── repair.py               # RepairStrategySelector
│   ├── two_phase_strategy.py   # TwoPhaseStrategy + ComplexityCalculator
│   ├── memory.py / memory_store.py
│   ├── events.py / metrics.py
│   └── ...
│
├── server/                     # FastAPI HTTP layer
│   └── api.py                  # OpenAI-compatible endpoints + LangGraph wiring
├── rag/                        # Qdrant vector search
├── vllm/                       # LLM client
├── indexer/                    # Java parsing + embedding + indexing
├── context/                    # Smart context assembly
├── intelligence/               # Repo structural analysis
├── config/                     # YAML configurations
│   └── agent.yaml              # Agent + two-phase + langgraph config
└── tests/                      # Test suites
    ├── test_graph_structure.py  # Graph compilation + routing tests
    └── test_graph_e2e.py        # Full pipeline E2E test
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `QDRANT_HOST` | `localhost` | Qdrant server host |
| `QDRANT_PORT` | `6333` | Qdrant server port |
| `VLLM_BASE_URL` | `http://localhost:8000/v1` | vLLM server URL |
| `VLLM_MODEL` | `Qwen/Qwen2.5-Coder-7B-Instruct-AWQ` | Model identifier |
| `SERVER_PORT` | `8080` | API server port |
| `USE_LEGACY_ORCHESTRATOR` | `false` | Set `true` to use legacy orchestrator |
| `LANGGRAPH_CHECKPOINT_DB` | `checkpoints.db` | SQLite path for LangGraph state |

### LangGraph Config (`config/agent.yaml`)

```yaml
langgraph:
  checkpoint_db: "checkpoints.db"
  require_human_review: false     # true for CI/CD pipelines
  use_legacy: false
  max_retries: 3
```

## Backend Toggle

```bash
# LangGraph (default)
python main.py

# Legacy orchestrator
USE_LEGACY_ORCHESTRATOR=true python main.py
```

## Running Tests

```bash
# Graph structure tests
python tests/test_graph_structure.py

# Full E2E pipeline test
python tests/test_graph_e2e.py

# Legacy phase tests
python -m pytest tests/test_phase3_4.py -v
```

## License

Internal use only.
