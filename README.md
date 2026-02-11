# AI Coding Agent

A self-hosted AI coding agent that generates JUnit5 + Mockito unit tests for large DDD Java repositories using RAG (Retrieval Augmented Generation).

## Architecture

```
Tabby (IDE) → Agent Orchestrator → Local RAG Index (Qdrant) → vLLM (Qwen2.5)
```

## Features

- **Tabby IDE Integration**: OpenAI-compatible API for seamless IDE integration
- **Semantic Code Understanding**: Uses tree-sitter to parse Java code and extract services, entities, repositories, and methods
- **RAG-Powered Context**: Precomputed semantic index avoids loading full source into context
- **DDD-Aware**: Understands application, domain, and infrastructure layers
- **Test Generation Rules**: Enforces JUnit5 + Mockito patterns, AAA structure, no Spring test context
- **Session Memory**: Maintains context across multiple interactions

## Prerequisites

- Python 3.11+
- Docker & Docker Compose
- vLLM server running with Qwen2.5-Coder model
- Java repository to index

## Quick Start

### 1. Clone and Setup

```bash
cd ai-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp env.example .env
# Edit .env with your configuration
```

### 2. Start Services

```bash
# Bước 1: Chạy Qdrant (vector database)
docker-compose up -d qdrant

# Bước 2: Chạy AI Agent app (kết nối với vLLM server đã forward port về localhost:8000)
python main.py

# Hoặc chạy với auto-reload khi dev
python -m uvicorn server.api:app --host 0.0.0.0 --port 8080 --reload
```

> **Note**: Giả định vLLM server đã chạy và forward port về `localhost:8000`

### 3. Index Your Java Repository

```bash
# Using curl
curl -X POST http://localhost:8080/reindex \
  -H "Content-Type: application/json" \
  -d '{
    "repo_path": "/path/to/your/java/repo",
    "recreate": true
  }'
```

### 4. Configure Tabby IDE

Xem chi tiết tại [TABBY_SETUP.md](./TABBY_SETUP.md)

**Quick setup trong VS Code:**

1. Mở Settings (Ctrl+,)
2. Tìm "Tabby"
3. Đặt **Tabby: Endpoint** = `http://localhost:8080/v1`

Hoặc trong `settings.json`:
```json
{
  "tabby.endpoint": "http://localhost:8080/v1"
}
```

### 5. Generate Tests

```bash
# Qua Tabby chat trong IDE:
# "Generate unit test for UserService"

# Hoặc qua API trực tiếp:
curl -X POST http://localhost:8080/generate-test \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "src/main/java/com/example/service/UserService.java",
    "task_description": "Generate comprehensive unit tests covering all public methods"
  }'
```

## API Endpoints

### OpenAI-Compatible (for Tabby)

| Endpoint | Mô tả |
|----------|-------|
| `GET /v1/models` | List available models |
| `POST /v1/chat/completions` | Chat completions (main endpoint) |
| `POST /v1/completions` | Text completions (legacy) |
| `GET /v1/health` | Health check |

### Native Endpoints

### Test Generation

#### `POST /generate-test`

Generate unit tests for a Java class.

**Request:**
```json
{
  "file_path": "src/main/java/com/example/service/UserService.java",
  "class_name": "UserService",
  "task_description": "Generate tests for user registration flow",
  "session_id": "optional-session-id"
}
```

**Response:**
```json
{
  "success": true,
  "test_code": "// Generated test code...",
  "class_name": "UserService",
  "validation_passed": true,
  "validation_issues": [],
  "session_id": "uuid-session-id",
  "rag_chunks_used": 8,
  "tokens_used": 2500
}
```

#### `POST /refine-test`

Refine a previously generated test based on feedback.

**Request:**
```json
{
  "session_id": "uuid-session-id",
  "feedback": "Add more edge case tests for null inputs"
}
```

### Indexing

#### `POST /reindex`

Index or reindex a Java repository.

**Request:**
```json
{
  "repo_path": "/path/to/java/repo",
  "recreate": false
}
```

#### `GET /index/stats`

Get statistics about the vector index.

### Session Management

#### `POST /session` - Create new session
#### `GET /session/{session_id}` - Get session info
#### `DELETE /session/{session_id}` - Delete session
#### `GET /sessions` - List all sessions

### Health

#### `GET /health`

Check service health status.

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `QDRANT_HOST` | Qdrant server host | `localhost` |
| `QDRANT_PORT` | Qdrant server port | `6333` |
| `QDRANT_COLLECTION` | Collection name | `java_codebase` |
| `VLLM_BASE_URL` | vLLM API base URL | `http://localhost:8000/v1` |
| `VLLM_MODEL` | Model name | `Qwen/Qwen2.5-Coder-7B-Instruct` |
| `VLLM_API_KEY` | vLLM API key | `token-abc123` |
| `EMBEDDING_MODEL` | Sentence transformer model | `sentence-transformers/all-MiniLM-L6-v2` |
| `SERVER_HOST` | Server bind host | `0.0.0.0` |
| `SERVER_PORT` | Server port | `8080` |

### YAML Configuration Files

- `config/agent.yaml` - Agent orchestrator settings
- `config/rag.yaml` - RAG and Qdrant settings
- `config/vllm.yaml` - vLLM client settings

## Project Structure

```
ai-agent/
├── agent/
│   ├── orchestrator.py    # Main workflow coordination
│   ├── prompt.py          # Prompt construction
│   ├── rules.py           # Test generation rules
│   └── memory.py          # Session memory management
├── indexer/
│   ├── parse_java.py      # Tree-sitter Java parser
│   ├── summarize.py       # Code summarization
│   └── build_index.py     # Qdrant index builder
├── rag/
│   ├── client.py          # RAG query client
│   └── schema.py          # Data models
├── server/
│   ├── api.py             # FastAPI endpoints
│   └── session.py         # Session management
├── vllm/
│   └── client.py          # vLLM API client
├── config/
│   ├── agent.yaml
│   ├── rag.yaml
│   └── vllm.yaml
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

## Test Generation Rules

The agent enforces these rules for generated tests:

1. **JUnit 5 Only**: Uses `@Test`, `@BeforeEach`, `@DisplayName`, `@Nested`
2. **Mockito Only**: Uses `@Mock`, `@InjectMocks`, `@ExtendWith(MockitoExtension.class)`
3. **No Spring Context**: Never uses `@SpringBootTest`, `@DataJpaTest`, `@WebMvcTest`
4. **AAA Pattern**: All tests follow Arrange-Act-Assert with clear comments
5. **Interaction Verification**: Uses `verify()` to check mock interactions
6. **Meaningful Names**: Uses `@DisplayName` for readable test descriptions

## Example Usage with curl

```bash
# 1. Check health
curl http://localhost:8080/health

# 2. Index repository
curl -X POST http://localhost:8080/reindex \
  -H "Content-Type: application/json" \
  -d '{"repo_path": "/app/repo", "recreate": true}'

# 3. Check index stats
curl http://localhost:8080/index/stats

# 4. Create a session
curl -X POST http://localhost:8080/session

# 5. Generate test
curl -X POST http://localhost:8080/generate-test \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "src/main/java/com/example/service/OrderService.java",
    "session_id": "your-session-id"
  }'

# 6. Refine test with feedback
curl -X POST http://localhost:8080/refine-test \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "your-session-id",
    "feedback": "Add tests for concurrent order processing"
  }'
```

## Running vLLM

The agent expects a vLLM server running with an OpenAI-compatible API:

```bash
# Example: Run vLLM with Qwen2.5-Coder
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --api-key token-abc123
```

## Development

```bash
# Run with auto-reload
python -m uvicorn server.api:app --reload --host 0.0.0.0 --port 8080

# Run tests (if you add them)
pytest tests/

# Format code
black .
isort .
```

## License

MIT License

