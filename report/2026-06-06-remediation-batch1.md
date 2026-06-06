# Remediation Batch 1 — P0 + quick wins

**Date:** 2026-06-06
**Branch:** `feature/new-architecture`
**Status:** ✅ DONE — 408 tests passing (deleted 2 dead-module test files; +10 remediation tests)
**Nguồn:** `docs/remediation-plan.md`

---

## Đã làm

### R1 — Sandbox isolation 🔴 (Security blocker)
`mcp_server/sandbox.py` — đóng các vector RCE qua interpreter:
- **Chặn inline-code execution**: `python -c`, `node -e`, `--eval`, `-p`, `-` (stdin) cho python/node/ruby/perl/deno/bun.
- **Chặn package-install / supply-chain**: `pip install`, `-m pip`, `npm install/exec`, **`npx`**, `gem/cargo/go install`, `poetry add`, `uv pip`.
- **`npx` gỡ khỏi whitelist** (gate `EXEC_ALLOW_NPX`).
- **`EXEC_ENABLED`** (default true) — tắt toàn bộ exec trong môi trường không tin cậy.
- **Container isolation hook**: `EXEC_ISOLATION=container` → chạy trong container ephemeral `--network=none --read-only` resource-capped (OS-isolation thật; default no-op).
- Vẫn cho `pytest`, `python -m pytest`, `go test`, `ruff`, git read-only.

### R2 — CI/CD + Eval gate
- `.github/workflows/ci.yml`: test (DEV_MODE pytest) · ruff (advisory) · docker build + Trivy · **eval-gate** job.
- `eval/ci_gate.py`: so benchmark vs baseline → exit non-zero khi regression (chặn merge).

### R3 — Audit auth failures
`server/auth.py` — ghi `security/auth` event (best-effort) khi verify_api_key fail (actor=IP, correlation_id). Audit đã default Postgres khi có `DATABASE_URL`.

### R6 — Circuit breaker mở rộng
`rag_search.py` — gọi `qdrant.hybrid_search` qua `get_circuit_breaker("qdrant")` → fail-fast khi Qdrant down (RAG path). State hiện trong `/health/deep`.

### R7 — Dead code cleanup
Xóa `server/agent/parallel_tools.py`, `server/agent/task_queue.py` + test (không nơi nào wire; tool chạy client-side nên không thể wire server-side).

### R13 — Prompt versioning đầy đủ
`config/prompts/intents.yaml` (v1.1.0) — migrate **đủ 8 intent** (thêm code_review/structural_analysis/search/debug/refine/explain). `generate` ưu tiên YAML, fallback `INTENT_PROMPTS`.

---

## Tests
`tests/test_remediation.py` (10): sandbox hardening (7), eval gate (2), prompt migration (1). Full suite **408 passed**, compileall exit 0.

## Trace nhanh
```
mcp_server/sandbox.py        ~ exec_enabled, inline-code/install guards, isolation_wrap, npx out
server/auth.py               ~ audit auth failures
server/agent/rag_search.py   ~ qdrant via circuit breaker
config/prompts/intents.yaml  ~ all 8 intents (v1.1.0)
eval/ci_gate.py              (mới) eval gate
.github/workflows/ci.yml     (mới) CI pipeline
deleted: parallel_tools.py, task_queue.py + tests
tests/test_remediation.py    + 10 tests
```

---

## Còn lại (feature-scale → pass riêng để giữ ổn định)

| | Mục | Lý do tách |
|---|-----|-----------|
| R4 | Async job queue (`/jobs`) | thay đổi luồng request; cần thiết kế status/streaming |
| R5 | Wire `summarize` an toàn | đụng tool_call pairing — cần làm cẩn thận |
| R8 | RAG repo_id isolation | chỉ khi bật RAG multi-tenant (đang off) |
| R9 | Multi-agent `/agents` | feature mới (design có sẵn) |
| R10 | Cross-session memory | feature mới (Postgres + embedding) |
| R11 | RBAC | feature mới |
| R12 | State checkpoint | **bỏ** — server stateless by design; "resume long task" thuộc R4 |

> Mỗi mục còn lại nên là 1 pass riêng (build + test + verify) để **không nhồi ẩu**, đúng yêu cầu "run mượt".

---

*Report generated: 2026-06-06.*
