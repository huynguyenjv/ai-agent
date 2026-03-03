# AI Agent — JUnit5 Test Generator

Self-hosted AI agent that generates **JUnit5 + Mockito** unit tests for Java services.  
Uses RAG (Qdrant) + local LLM (vLLM) to produce tests that are context-aware, compilable, and follow DDD conventions.

## Architecture

```
Tabby / Continue IDE
        │
        ▼
┌──────────────────────────────────────────────────────┐
│  FastAPI Server  (OpenAI-compatible /v1/chat/...)     │
│  server/api.py                                        │
└──────────┬───────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────┐
│  Agent Orchestrator        agent/orchestrator.py      │
│  StateMachine → Planner → Retrieve → Generate        │
│               → Validate → Repair                     │
└──────┬──────────────┬────────────────┬───────────────┘
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
│  indexer/  │
└────────────┘
```

## Features

- **OpenAI-compatible API** — plug into Tabby IDE or Continue IDE as a custom model provider
- **Tree-sitter Java parsing** — extracts classes, records, enums, interfaces, Lombok annotations, fields, methods
- **Semantic vector index** — Qdrant with sentence-transformers (`all-MiniLM-L6-v2`)
- **DDD-aware layer detection** — auto-classifies service/repository/domain/controller layers
- **Multi-phase pipeline** — Plan → Retrieve → Generate → Validate → Repair
- **7-pass validation** — structural checks, anti-patterns, construction cross-checking against RAG metadata
- **Targeted repair** — per-category repair strategies with focused LLM prompts
- **Smart context assembly** — priority-based snippet selection + token-budget optimization
- **Graph intelligence** — file-level dependency graph + global symbol table for smart mock detection
- **Session memory** — in-memory or Redis-backed conversation persistence
- **Event bus + metrics** — decoupled observability with timing, counters, quality rates
- **Streaming support** — token-by-token streaming to avoid vLLM KV cache blocking

## Quick Start

### 1. Prerequisites

- Python 3.11+
- Docker (for Qdrant)
- vLLM server with a code model (e.g. `Qwen/Qwen2.5-Coder-32B-Instruct-AWQ`)

### 2. Setup

```bash
# Clone & create venv
python -m venv venv
source venv/bin/activate        # Linux/Mac
.\venv\Scripts\Activate.ps1     # Windows PowerShell

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp env.example .env
# Edit .env — set QDRANT_HOST, VLLM_BASE_URL, JAVA_REPO_PATH
```

### 3. Start Dependencies

```bash
# Qdrant only (development)
docker-compose up -d qdrant

# Full stack (production)
docker-compose up -d
```

### 4. Index a Java Repository

```bash
# Start the server
python main.py

# Index your Java codebase (via API)
curl -X POST http://localhost:8080/reindex \
  -H "Content-Type: application/json" \
  -d '{"repo_path": "/path/to/java/repo"}'
```

### 5. Connect IDE

See [TABBY_SETUP.md](TABBY_SETUP.md) for detailed Tabby IDE configuration.

For Continue IDE, add to `config.json`:
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

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat (main IDE integration) |
| `GET`  | `/v1/models` | List available models |
| `POST` | `/generate-test` | Native test generation endpoint |
| `POST` | `/refine-test` | Refine generated test with feedback |
| `POST` | `/reindex` | Index/re-index a Java repository |
| `GET`  | `/index/stats` | Qdrant index statistics |
| `GET`  | `/index/lookup/{class_name}` | Diagnostic: inspect indexed class payload |
| `GET`  | `/health` | Health check |
| `GET`  | `/metrics` | Prometheus-style metrics |
| `POST` | `/session/create` | Create a new session |
| `GET`  | `/session/{id}` | Get session state |
| `DELETE` | `/session/{id}` | Delete session |

## Project Structure

```
ai-agent/
├── main.py                 # Entry point (uvicorn server)
├── requirements.txt        # Python dependencies
├── env.example             # Environment template
├── Dockerfile              # Container image
├── docker-compose.yml      # Full stack (agent + qdrant + vllm + redis)
├── download_model.py       # Download embedding model for offline use
├── benchmark.py            # API performance benchmark suite
├── TABBY_SETUP.md          # Tabby IDE integration guide
│
├── agent/                  # Core orchestration (state machine, planner, prompt, validation, repair)
├── server/                 # FastAPI HTTP layer (OpenAI-compatible API)
├── rag/                    # Qdrant vector search client
├── vllm/                   # LLM client (OpenAI-compatible)
├── indexer/                # Java parsing + embedding + indexing
├── context/                # Smart context assembly (snippet selection, token optimization)
├── intelligence/           # Repo structural intelligence (graph, symbol map)
├── config/                 # YAML configurations
├── models/                 # Downloaded embedding model (gitignored)
├── benchmark/              # Benchmark results (gitignored)
└── tests/                  # Development test suites
```

Each folder has its own `README.md` with detailed documentation.

## Configuration

All configuration is via environment variables (see [env.example](env.example)):

| Variable | Default | Description |
|----------|---------|-------------|
| `QDRANT_HOST` | `localhost` | Qdrant server host |
| `QDRANT_PORT` | `6333` | Qdrant server port |
| `QDRANT_COLLECTION` | `java_codebase` | Qdrant collection name |
| `VLLM_BASE_URL` | `http://localhost:8000/v1` | vLLM server URL |
| `VLLM_MODEL` | `Qwen/Qwen2.5-Coder-7B-Instruct-AWQ` | Model identifier |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `JAVA_REPO_PATH` | — | Java repo to index |
| `SERVER_PORT` | `8080` | API server port |
| `MEMORY_BACKEND` | `memory` | `memory` (dev) or `redis` (prod) |

YAML config files in `config/` provide detailed tuning for agent behavior, RAG search, and LLM generation parameters.

## Docker

```bash
# Development: only dependencies
docker-compose up -d qdrant redis

# Production: full stack (requires NVIDIA GPU for vLLM)
docker-compose up -d

# With monitoring tools
docker-compose --profile tools up -d
```

## Running Tests

```bash
# Phase 3/4 tests (validation, repair, events, metrics)
python -m pytest tests/test_phase3_4.py -v

# Phase 1 tests (state machine, planner)
python tests/test_phase1.py

# Phase 2 tests (intelligence, context)
python tests/test_phase2.py

# E2E trace test (AuthUseCaseService scenario)
python tests/_test_e2e_trace.py
```

## License

Internal use only.

