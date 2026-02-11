# Kết nối AI Agent với Tabby IDE

## Tổng quan

AI Agent cung cấp OpenAI-compatible API để tích hợp với Tabby IDE. Tabby sẽ gửi request đến AI Agent, agent sẽ:
1. Tìm context liên quan từ RAG index
2. Gọi vLLM để generate response
3. Trả về kết quả cho Tabby

```
Tabby IDE → AI Agent (localhost:8080) → RAG (Qdrant) + vLLM (localhost:8000)
```

## Cách 1: Cấu hình Tabby sử dụng Custom API Endpoint

### Bước 1: Chạy AI Agent

```bash
cd ai-agent

# Chạy Qdrant
docker-compose up -d qdrant

# Chạy AI Agent
python main.py
```

### Bước 2: Cấu hình Tabby

Mở file cấu hình Tabby (thường ở `~/.tabby-client/agent/config.toml` hoặc trong settings của IDE):

```toml
# Tabby Agent Configuration

[server]
# Trỏ đến AI Agent server
endpoint = "http://localhost:8080/v1"
token = "token-abc123"

[completion]
# Sử dụng AI Agent cho code completion
enabled = true

[chat]
# Sử dụng AI Agent cho chat
enabled = true
```

### Bước 3: Cấu hình trong VS Code / IDE

1. Mở VS Code Settings (Ctrl+,)
2. Tìm "Tabby"
3. Đặt **Tabby: Endpoint** = `http://localhost:8080/v1`
4. Đặt **Tabby: API Token** = `token-abc123` (hoặc bỏ trống)

Hoặc trong `settings.json`:

```json
{
  "tabby.endpoint": "http://localhost:8080/v1",
  "tabby.api.token": "token-abc123"
}
```

## Cách 2: Sử dụng Tabby với External Model

Nếu bạn muốn Tabby kết nối trực tiếp với vLLM và chỉ dùng AI Agent cho RAG:

### Cấu hình Tabby Server

Trong `~/.tabby/config.toml`:

```toml
[model.completion.http]
kind = "openai/completion"
api_endpoint = "http://localhost:8000/v1"
model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"

[model.chat.http]
kind = "openai/chat"
api_endpoint = "http://localhost:8080/v1"  # AI Agent cho chat (có RAG)
model_name = "ai-agent"
```

## API Endpoints

AI Agent cung cấp các endpoint tương thích OpenAI:

| Endpoint | Mô tả |
|----------|-------|
| `GET /v1/models` | Danh sách models |
| `POST /v1/chat/completions` | Chat completions (chính) |
| `POST /v1/completions` | Text completions (legacy) |
| `GET /v1/health` | Health check |

### Ví dụ Request

```bash
# Chat completion
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer token-abc123" \
  -d '{
    "model": "ai-agent",
    "messages": [
      {"role": "user", "content": "Generate unit test for UserService"}
    ],
    "file_path": "src/main/java/com/example/service/UserService.java"
  }'
```

### Custom Fields

AI Agent hỗ trợ thêm các field custom trong request:

```json
{
  "model": "ai-agent",
  "messages": [...],
  "file_path": "path/to/current/file.java",  // File đang edit
  "workspace_path": "/path/to/project"        // Root của project
}
```

Khi có `file_path`, AI Agent sẽ tự động:
1. Tìm context liên quan từ RAG
2. Thêm vào prompt trước khi gọi LLM

## Test Generation

Để generate unit test, gửi message chứa keyword "test", "junit", "unit test":

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ai-agent-test-generator",
    "messages": [
      {"role": "user", "content": "Generate comprehensive unit tests for this service"}
    ],
    "file_path": "src/main/java/com/example/service/OrderService.java"
  }'
```

## Workflow đề xuất

1. **Index codebase trước**:
   ```bash
   curl -X POST http://localhost:8080/reindex \
     -H "Content-Type: application/json" \
     -d '{"repo_path": "/path/to/java/project", "recreate": true}'
   ```

2. **Cấu hình Tabby** trỏ đến `http://localhost:8080/v1`

3. **Sử dụng trong IDE**:
   - Mở file Java cần test
   - Dùng Tabby chat: "Generate unit test for this class"
   - AI Agent sẽ tự động tìm context và generate test

## Troubleshooting

### Tabby không kết nối được

1. Kiểm tra AI Agent đang chạy:
   ```bash
   curl http://localhost:8080/health
   ```

2. Kiểm tra endpoint trong Tabby settings

3. Xem logs:
   ```bash
   # AI Agent logs
   python main.py  # Xem output
   ```

### Response chậm

- RAG search mất thời gian với index lớn
- vLLM generation phụ thuộc vào GPU
- Giảm `top_k` trong config để tăng tốc

### Không có context từ RAG

1. Kiểm tra đã index chưa:
   ```bash
   curl http://localhost:8080/index/stats
   ```

2. Reindex nếu cần:
   ```bash
   curl -X POST http://localhost:8080/reindex \
     -d '{"repo_path": "/path/to/repo", "recreate": true}'
   ```

