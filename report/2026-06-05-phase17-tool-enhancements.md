# Phase 17 — Tool Enhancements (17.4 + 17.5 + 17.1)

**Date:** 2026-06-05
**Branch:** `feature/new-architecture`
**Status:** ✅ COMPLETED (chosen scope) — 404 tests passing
**Scope:** 17.4 Tool Result Validation · 17.5 Tool Usage Analytics · 17.1 Multi-file Atomic Edits

> 17.2 Call Graph và 17.3 LSP **bỏ/hoãn**: LSP nặng (spawn language server) và trùng lặp với agentic search (grep/symbol) — baseline cũng loại LSP khỏi scope.

---

## 1. 17.4 Tool Result Validation — `server/agent/tool_validator.py`

- `validate_tool_result(tool_name, content) -> ToolValidation{valid, category, reason, retry_suggestion}`.
- Bắt: kết quả rỗng, `{"error": ...}`, malformed (vd `read_file` thiếu `content`), edit `success=false`.
- **Bảo thủ**: "no matches" / "tests failed" vẫn coi là **valid** (kết quả hợp lệ, không phải lỗi).
- `annotate_invalid()` chèn note model đọc được để tự sửa.
- **Wire:** `chat._convert_messages` (nhánh tool) validate sau redact, annotate nếu invalid.

## 2. 17.5 Tool Usage Analytics — `server/metrics/tools.py`

- `ToolAnalytics`: per-tool calls/success_rate/failure_rate/avg_latency + `most_used`.
- **Wire:** `chat._convert_messages` ghi `record(tool_name, success=validation.valid)` mỗi tool result.
- (Tool chạy client-side nên latency thường không có — analytics chủ yếu theo count + success.)

## 3. 17.1 Multi-file Atomic Edits — `mcp_server/tools_multifile.py`

- `apply_multi_file_edits(repo_path, edits, dry_run)`:
  - **Pre-flight**: validate path (trong repo), tính nội dung mới, **phát hiện conflict** (search không thấy / thiếu new_content) → abort **trước khi ghi**.
  - **Atomic**: ghi tất cả → verify → **rollback toàn bộ** nếu bất kỳ bước nào lỗi (khôi phục nội dung gốc; file mới tạo thì xóa).
  - **dry_run**: trả unified diff, không ghi.
- Nâng cấp so với `apply_edits` (Phase 1): thêm backup/rollback + conflict detection.
- **Parity:** đăng ký `vtrip_apply_edits_atomic` ở `mcp_server/server.py` + advertise trong `generate.MCP_TOOLS` + TOOL_INSTRUCTIONS (khuyên dùng cho multi-file). Test parity khóa.

---

## 4. Tests

`tests/test_phase17_tools.py` (13): validator (6), analytics (1), atomic edit (5: apply/search-replace/conflict-no-write/dry-run/path-guard), parity (1). 391 → **404 passed**. compileall exit 0.

## 5. Bỏ/hoãn
- **17.2 Call Graph** — đáng làm nếu cần impact analysis; chưa làm.
- **17.3 LSP** — bỏ (nặng + trùng agentic search).

## 6. Trace nhanh
```
server/agent/tool_validator.py    (mới) validate tool result + retry hint
server/metrics/tools.py           (mới) ToolAnalytics
mcp_server/tools_multifile.py     (mới) apply_multi_file_edits (atomic + rollback)
mcp_server/server.py              ~ đăng ký vtrip_apply_edits_atomic
server/agent/generate.py          ~ MCP_TOOLS + TOOL_INSTRUCTIONS (atomic edit)
server/routers/chat.py            ~ wire validate + analytics vào nhánh tool
tests/test_phase17_tools.py       + 13 tests
```

---

*Report generated: 2026-06-05. Liên quan: improvement-plan Phase 17; [[mcp-tool-execution-model]].*
