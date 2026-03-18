# Continue.dev + AI Agent Indexer — Setup Guide

## 1. Install MCP dependency

```bash
pip install "mcp>=1.0.0"
```

## 2. Configure Continue.dev

Add the MCP server to your Continue config file.

### Option A: Workspace config (`.continue/config.yaml`)

```yaml
mcpServers:
  - name: java-indexer
    command: python
    args:
      - C:/Users/huynmb/IdeaProjects/source/train/tabby-pipeline/ai-agent/scripts/mcp_index_server.py
    env:
      AI_AGENT_URL: http://localhost:8080
```

### Option B: Global config (`~/.continue/config.yaml`)

Same content as above, added under the `mcpServers:` key.

## 3. Available Tools in Continue

Once configured, you can use these tools in Continue chat:

| Tool | Description | Example prompt |
|------|-------------|----------------|
| `index_file` | Index 1 Java file | *"Index the file at C:/path/to/MyService.java"* |
| `index_directory` | Index all .java files in a folder | *"Index all Java files in C:/path/to/src"* |
| `index_current_file` | Quick-index current file | *"Index my current file"* |

### Example prompts in Continue:

```
@java-indexer Index the file at C:/Users/me/project/src/main/java/com/example/UserService.java
```

```
@java-indexer Index all Java files in C:/Users/me/project/src/main/java
```

## 4. Optional: Embeddings Provider

To also use the AI Agent's embedding model for Continue's built-in `@codebase` feature:

```yaml
models:
  - name: ai-agent-embed
    provider: openai
    model: all-MiniLM-L6-v2-onnx
    apiBase: http://localhost:8080/v1
    roles:
      - embed
```

## 5. Verify

1. Start AI Agent server: `python main.py`
2. Open Continue in VS Code / JetBrains
3. Type in Continue chat: `@java-indexer index the file at <path>`
4. Check Qdrant dashboard at `http://localhost:6333/dashboard`
