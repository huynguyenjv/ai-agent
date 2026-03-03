# server/ — FastAPI HTTP Layer

Exposes the agent via HTTP endpoints. Provides an **OpenAI-compatible API** for Tabby/Continue IDE integration, plus native endpoints for test generation, indexing, and session management.

## Files

| File | Description |
|------|-------------|
| `api.py` | Main FastAPI application. All endpoint definitions, middleware (CORS, rate limiting), streaming support, tool-calling protocol. |
| `session.py` | `SessionManager` — thin CRUD for session tracking at API level. |

## Key Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat completion (main IDE integration point) |
| `GET` | `/v1/models` | List available models |
| `POST` | `/generate-test` | Direct test generation (non-OpenAI format) |
| `POST` | `/refine-test` | Refine a generated test with user feedback |
| `POST` | `/reindex` | Index or re-index a Java repository |
| `GET` | `/index/stats` | Qdrant index statistics |
| `GET` | `/index/lookup/{class_name}` | Diagnostic: inspect full Qdrant payload for a class |
| `GET` | `/health` | Health check |
| `GET` | `/metrics` | Prometheus-style metrics (timing, counts, quality) |
| `POST` | `/session/create` | Create new session |
| `GET` | `/session/{id}` | Get session state |
| `DELETE` | `/session/{id}` | Delete session |

## Features

- **Streaming** — SSE streaming for `/v1/chat/completions` (token-by-token)
- **Rate limiting** — configurable per-IP rate limiting
- **CORS** — configurable origins for cross-origin requests
- **Tool calling** — supports OpenAI function/tool call protocol for IDE integration
- **Request timeout** — configurable per-request timeout

## Dependencies

- `agent/` — orchestration logic
- `indexer/` — Java parsing and indexing
- `rag/` — vector search
- `vllm/` — LLM client
