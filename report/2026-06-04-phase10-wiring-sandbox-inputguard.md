# Phase 10 (Security) — Wire Sandbox + Input Guard

**Date:** 2026-06-04
**Branch:** `feature/new-architecture`
**Status:** ✅ COMPLETED — 335 tests passing
**Scope:** Phase 10.1 (Command Sandbox) + 10.2 (Prompt Injection Guard) — **wiring only**

---

## 1. Bối cảnh

`mcp_server/sandbox.py` (CommandSandbox) và `server/agent/input_guard.py` (InputGuard) đã có code + unit test từ trước, nhưng **chưa wire** vào pipeline (dead code — giống Phase 8 trước đây):
- `run_command` vẫn dùng whitelist/blocklist cũ (Phase 4), không gọi sandbox.
- `chat.py` không gọi InputGuard (chỉ có `sanitize_user_input` của Phase 4).

Đợt này: **wire 2 module này vào luồng thật**. (Còn lại Phase 10.3 secret scanning, 10.5 audit log, 10.6 input validation — chưa làm.)

---

## 2. Thay đổi

### 2.1 Sandbox → `run_command` (10.1)

**File:** `mcp_server/tools.py`

- `run_command()` giờ **delegate sang `CommandSandbox.execute()`** (qua `get_sandbox()`), thay cho path whitelist/blocklist nội tại.
- Lợi ích thêm so với cũ: whitelist theo category, `DANGEROUS_PATTERNS` regex (rm -rf, sudo, curl|sh, fork bomb…), **env isolation** (xoá LD_PRELOAD/LD_LIBRARY_PATH/PYTHONPATH, set CI/TERM), path-traversal guard, **audit log** in-memory.
- Giữ `timeout` param: nếu khác default singleton → tạo `CommandSandbox` per-call với `SandboxConfig(timeout_seconds=...)` (không mutate singleton).
- `run_command` giữ nguyên signature + shape trả về → các caller (MCP `vtrip_run_command`) không đổi. `ALLOWED_COMMANDS`/`BLOCKED_PATTERNS` cũ giữ lại (không xoá) nhưng không còn dùng trong `run_command`.

### 2.2 InputGuard → `chat.py` (10.2)

**File:** `server/routers/chat.py`

- `_convert_messages`: với message role=user, sau `sanitize_user_input` (Phase 4) chạy thêm `get_input_guard().check_and_sanitize()` → **trung hoà** role-hijack/delimiter/zero-width/homoglyph. Message CRITICAL bị guard trả `""` (defense-in-depth).
- `_stream_response`: trước khi build graph, kiểm tra user-turn mới nhất bằng `get_input_guard().check()`. Nếu `blocked` (mặc định block khi **CRITICAL**) → trả SSE từ chối + `done`, **không** chạy graph.
- Threat mức thấp hơn (HIGH/MEDIUM) **không block**, chỉ neutralize → tránh false-positive chặn nhầm.

### 2.3 Tests

**File:** `tests/test_phase10_wiring.py` (7 test)
- Sandbox-in-run_command: chặn dangerous pattern (`rm -rf`), chặn lệnh ngoài whitelist (wording "not whitelisted" ⇒ chứng minh đã qua sandbox), chặn git write subcommand, chặn `curl | sh`.
- InputGuard-in-chat: delimiter injection bị trung hoà (`<system>` không lọt), critical injection → content rỗng, message thường giữ nguyên.

---

## 3. Verify

```
python -m compileall -q server mcp_server   # exit 0
python -m pytest                            # 335 passed (was 328, +7)
```

---

## 4. Còn lại của Phase 10 (chưa làm)

| Mục | File dự kiến | Ghi chú |
|-----|-------------|---------|
| 10.3 Secret scanning | `server/utils/secret_scanner.py` | quét output LLM + file read, redact `[REDACTED]` |
| 10.6 Input validation | `server/validation.py` | Pydantic limits: message/file size, path traversal, tool-arg schema |
| 10.5 Audit logging | `server/audit.py` + Postgres model | nặng nhất (cần DB schema), để cuối |
| 10.4 RBAC | — | optional, đã comment trong plan |

> Audit log hiện chỉ in-memory trong `CommandSandbox._audit_log` — bản Postgres đầy đủ là 10.5.

---

## 5. Trace nhanh (files chạm)

```
mcp_server/tools.py        ~ run_command() delegate sang CommandSandbox
server/routers/chat.py     + InputGuard: neutralize trong _convert_messages, block CRITICAL trong _stream_response
tests/test_phase10_wiring.py  + 7 tests
```

**Revert:** `run_command` khôi phục path whitelist cũ; gỡ import/2 block InputGuard trong chat.py.

---

*Report generated: 2026-06-04. Liên quan: sandbox.py/input_guard.py (đã commit ad726a0), improvement-plan Phase 10.*
