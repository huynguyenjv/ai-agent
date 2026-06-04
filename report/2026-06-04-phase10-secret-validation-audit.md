# Phase 10 (Security) — Secret Scanning + Input Validation + Audit Logging

**Date:** 2026-06-04
**Branch:** `feature/new-architecture`
**Status:** ✅ COMPLETED — 351 tests passing
**Scope:** Phase 10.3 (Secret Scanning) + 10.6 (Input Validation) + 10.5 (Audit Logging)

> Tiếp nối `report/2026-06-04-phase10-wiring-sandbox-inputguard.md` (đã wire 10.1 sandbox + 10.2 input-guard). Đợt này thêm 3 mục còn lại (trừ 10.4 RBAC optional).

---

## 1. 10.6 Input Validation — `server/validation.py`

- `validate_chat_request(request)` raise `ValidationError` (→ HTTP 422) khi:
  - messages rỗng / quá nhiều (`MAX_MESSAGES`=300)
  - 1 message quá lớn (`MAX_MESSAGE_CHARS`=200k) hoặc tổng hội thoại quá lớn (`MAX_TOTAL_CHARS`=1M) → chống DoS payload
  - `active_file` không an toàn (path traversal `..`, absolute, drive letter, NUL)
- `is_safe_relative_path()` helper dùng chung. Mọi limit override qua env.
- **Wire:** `chat.py` `chat_completions` gọi `validate_chat_request` ngay sau auth, raise `HTTPException(422)`.

## 2. 10.3 Secret Scanning — `server/utils/secret_scanner.py`

- `scan(text) -> [SecretFinding]`, `redact(text) -> (text, findings)`, `has_secrets(text)`.
- Patterns: AWS key, private-key block, GitHub/Slack token, Google API key, JWT, Bearer, URL basic-auth, và `KEY=value` (password/secret/api_key… — bắt cả prefix kiểu `DB_PASSWORD`).
- Entropy fallback (Shannon ≥ 4.0) cho token opaque ≥24 ký tự.
- `SecretFinding.preview` **luôn masked** (không bao giờ chứa secret gốc) — an toàn để log.
- **Wire:** `chat.py` `_convert_messages` (nhánh tool) redact secret **trước khi** tool output vào context model. Điểm rò rỉ chính (LLM đọc `.env`) được chặn.
- *Giới hạn đã biết:* output LLM stream token-by-token nên không redact hậu kỳ được; tập trung chặn ở tool-output (nguồn rò rỉ thực tế).

## 3. 10.5 Audit Logging — `server/audit.py` + `init-db/002_audit_schema.sql`

- `AuditEvent` (event_type, action, actor, outcome, detail, correlation_id, timestamp) + `AuditEventType` (tool_execution/auth/security_violation/admin).
- `AuditLogger` **best-effort** (lỗi ghi audit **không** làm hỏng request) + mirror ra logger chuẩn (SIEM scrape).
- Dual backend (giống metrics): **SQLite** mặc định (`AUDIT_DB_PATH`, hỗ trợ `:memory:` qua 1 connection chia sẻ), **PostgreSQL** khi có `DATABASE_URL` (`psycopg_pool`).
- Retention: `clear_old(days=90)`; schema Postgres ở `init-db/002_audit_schema.sql` (index theo ts/type/actor/outcome).
- **Wire:** `chat.py` `_stream_response` ghi `security_violation("prompt_injection", actor, ...)` khi InputGuard block.

---

## 4. Tests

| File | Tests |
|------|-------|
| `tests/test_phase10_security.py` | 16 — validation (6), secret scanner (6), audit (4) |

Tổng: 335 → **351 passed** (+16). compileall exit 0.

---

## 5. Trạng thái Phase 10 (sau đợt này)

| Mục | Trạng thái |
|-----|-----------|
| 10.1 Command Sandbox | ✅ wired |
| 10.2 Prompt Injection Guard | ✅ wired |
| 10.3 Secret Scanning | ✅ done + wired (tool output) |
| 10.6 Input Validation | ✅ done + wired (chat endpoint) |
| 10.5 Audit Logging | ✅ done + wired (security violations) |
| 10.4 RBAC | ⏭️ optional (commented in plan), bỏ qua |

→ **Phase 10 hoàn tất** (trừ RBAC optional).

### Việc có thể mở rộng sau (không bắt buộc)
- Audit thêm: log tool_execution mỗi lượt + auth failure trong `auth.py`; endpoint `/audit` để xem; cron retention 90 ngày.
- Secret scan: thêm redact cho file read phía MCP (client-side) + cảnh báo realtime.

---

## 6. Trace nhanh (files)

```
server/validation.py            + validate_chat_request, is_safe_relative_path
server/utils/secret_scanner.py  + scan/redact/has_secrets
server/audit.py                 + AuditEvent/AuditLogger/SQLite+Postgres backends
init-db/002_audit_schema.sql    + audit_log table
server/routers/chat.py          ~ wire validation (endpoint), secret redact (tool msg), audit (injection block)
tests/test_phase10_security.py  + 16 tests
```

---

*Report generated: 2026-06-04. Liên quan: improvement-plan Phase 10, report wiring sandbox/input-guard cùng ngày.*
