# config/ — YAML Configuration

YAML-based configuration for the three main subsystems. Values use `${ENV_VAR:default}` syntax but are typically resolved via `os.getenv()` in each module.

## Files

| File | Description |
|------|-------------|
| `agent.yaml` | Agent orchestrator: `max_context_tokens`, `top_k_results`, `session_timeout`, prompt template, DDD layer detection rules. |
| `rag.yaml` | Qdrant connection, embedding model, indexing chunk config (150-250 tokens), search defaults (`top_k=10`, `threshold=0.5`). |
| `vllm.yaml` | vLLM server URL, model name, generation params (`temperature=0.2`, `max_tokens=4096`), retry settings. |

## Usage

These configs serve as documentation and defaults. Runtime values are primarily controlled via environment variables (see `env.example` in the project root).
