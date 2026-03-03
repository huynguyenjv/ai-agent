# vllm/ — LLM Client

OpenAI-compatible client for the vLLM server. Handles generation requests with streaming, retry logic, and connection pooling.

## Files

| File | Description |
|------|-------------|
| `client.py` | `VLLMClient` — `generate()` (streaming by default), `generate_streaming()` (yields tokens), `close()`. Uses httpx with connection pooling and tenacity retry. |

## Features

- **Streaming by default** — avoids vLLM KV cache blocking for long generations
- **Retry with backoff** — 3 attempts with exponential backoff via tenacity
- **Connection pooling** — httpx `AsyncClient` with persistent connections
- **Configurable parameters** — temperature, max_tokens, top_p, stop sequences

## Configuration

Via environment variables (see `env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_BASE_URL` | `http://localhost:8000/v1` | vLLM server URL |
| `VLLM_MODEL` | `Qwen/Qwen2.5-Coder-7B-Instruct-AWQ` | Model identifier |
| `VLLM_API_KEY` | `token-abc123` | API key |

Detailed tuning in `config/vllm.yaml` (temperature, max_tokens, retry settings).

## Public API

```python
from vllm import VLLMClient

client = VLLMClient()
response = client.generate(
    system_prompt="You are a test generator.",
    user_prompt="Generate tests for UserService.",
)
# response.content → generated code
# response.usage → token counts

# Streaming
for token in client.generate_streaming(system_prompt, user_prompt):
    print(token, end="")
```

## Dependencies

- `httpx` — async HTTP client
- `tenacity` — retry logic
