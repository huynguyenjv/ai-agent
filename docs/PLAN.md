# Orchestration Plan: Architecture Review Implementation

## 1. Context & Task
Theo yêu cầu từ kết quả review kiến trúc (file `architecture_review.md.resolved`), dự án cần thực hiện các cải tiến quan trọng để tăng bộ bền vững (robustness), hiệu năng (performance) và độ sạch của code (clean code).

Các mục tiêu chính:
1. **Refactor Prompts**: Chuyển logic từ `agent/prompt.py` sang **Jinja2** templates.
2. **Go Fully Native Async**: Sử dụng `.astream()` cho LangGraph và dọn dẹp các ThreadPool blocking calls.
3. **Decouple Validation Rules**: Tách các pattern regex cứng nhắc trong `ValidationPipeline` thành Rule-Engine (hoặc dynamic rules).
4. **Isolate Testing**: Mock Intelligence layer (Qdrant, Parse Java) để unit test LangGraph chạy dưới 2 giây.
5. **Clean Legacy Code**: Dọn dẹp `graph_adapter.py` và các models cũ.

---

## 2. Parallel Agent Orchestration (Phase 2)
Để thực hiện một khối lượng công việc liên quan đến nhiều domain (Kiến trúc, Hiệu năng, Kiểm thử), tôi sẽ điều phối **3 Agent** chạy song song (sau khi bản Plan này được duyệt):

| Nhóm | Agent | Nhiệm vụ cụ thể (Task Breakdown) | File ảnh hưởng |
|------|-------|----------------------------------|----------------|
| **Core API & Logic** | 🤖 `@backend-specialist` | - Tích hợp `Jinja2` vào `agent/prompt.py` (tạo class `TemplateEngine`).<br>- Refactor `ValidationPipeline` để load rules từ YAML/JSON/Dict cấu hình thay vì if-else regex.<br>- Xóa bỏ `graph_adapter.py`, tích hợp trực tiếp `GraphOrchestrator` vào API endpoint. | `agent/prompt.py`, `agent/validation.py`, `server/api.py`, `agent/graph_adapter.py` (DEL) |
| **Performance** | 🤖 `@performance-optimizer` | - Refactor các blocking calls khi gọi LLM và LangGraph. Đổi `graph.stream()` qua `graph.astream()`.<br>- Đảm bảo `QdrantClient` sử dụng AsyncIO hoặc không chặn Event Loop của FastAPI.<br>- Audit bộ nhớ và thread counts. | `agent/graph.py`, `server/api.py`, `rag/client.py` |
| **Testing & QA** | 🤖 `@test-engineer` | - Viết Mock layer bằng `pytest-mock` cho các external services (Qdrant, VLLM).<br>- Viết Unit tests cho `AgentState` và các Nodes của Graph mà không cần bật LLM/DB thực.<br>- Chạy toàn bộ bộ test và đảm bảo không vỡ luồng cũ. | `tests/test_graph_offline.py`, `tests/conftest.py` |

---

## 3. Verification Scripts (Exit Gate)
Đại diện của Agent cuối cùng (`test-engineer`) bắt buộc phải chạy các checklist QA:
1. `python .agent/skills/lint-and-validate/scripts/lint_runner.py .` -> Đảm bảo Code Clean.
2. `python .agent/skills/testing-patterns/scripts/test_runner.py .` -> Đảm bảo Unit Testing Coverage.
3. Kiểm tra tính tương thích của SSE streaming sau khi đổi sang `astream()`.

---
**Status:** ⏸️ Đợi sự phê duyệt từ User để tiến vào Phase 2 (Implementation).
