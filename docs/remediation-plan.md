# Remediation Plan — fix các finding từ Architecture Review

> **Nguồn:** Architecture Review 2026-06-06 (Production Readiness 6.3/10).
> **Mục tiêu:** đóng các blocker production + thu hẹp khoảng cách narrative↔artifact.
> **Nguyên tắc:** làm theo severity, mỗi mục có acceptance criteria + test; giữ tinh thần "xong 1 việc rồi sang việc khác".

---

## Ưu tiên tổng quan

| Wave | Mục tiêu | Items |
|------|----------|-------|
| **P0** | Chặn rủi ro production | R1 Sandbox isolation · R2 CI/CD + eval gate · R3 Audit durable |
| **P1** | Khép kín vận hành | R4 Async job/non-block · R5 Context (summarize) · R6 Circuit breaker mở rộng · R7 Dead code · R8 RAG repo_id (nếu giữ) |
| **P2** | Năng lực & định vị | R9 Multi-agent /agents (hoặc relabel) · R10 Cross-session memory · R11 RBAC · R12 State checkpoint · R13 Prompt versioning đầy đủ |

---

## P0 — Blockers

### R1. Sandbox execution isolation 🔴 (finding #1, Security)
**Vấn đề:** `CommandSandbox` chỉ whitelist+regex. Whitelist chứa `python/node/npx/java` → `python -c "..."` / `npx <pkg>` = **RCE/supply-chain RCE** sau khi qua input_guard. Không thể chỉ bỏ whitelist vì `run_tests`/lint cần interpreter.

**Cách fix (chọn 1, theo môi trường):**
- **(A) Container-per-exec (khuyến nghị):** chạy command trong container ephemeral (`docker run --rm --network=none --read-only -v workdir:rw --memory=512m --cpus=1 --pids-limit`). Vá tận gốc, OS-isolation thật.
- **(B) nsjail/firejail + cgroups (Linux bare-metal):** đúng như Phase 10.1.3 từng đề ra, chưa làm. `--net none`, rlimits, seccomp.
- **(C) Tối thiểu (nếu chưa làm A/B ngay):** mặc định **`EXEC_ENABLED=false`** cho run_command/run_tests/lint; chỉ bật trong môi trường tin cậy (CI runner). + thêm explicit confirm cho lệnh exec.

**Files:** `mcp_server/sandbox.py`, `mcp_server/tools.py`, env `EXEC_ENABLED`, `EXEC_ISOLATION=container|nsjail|none`.
**Acceptance:** test escape — `python -c "..."`, `npx ...`, `> /dev/...`, network call đều bị chặn/cô lập; `pytest` hợp lệ vẫn chạy trong sandbox. **Effort:** 2-3 ngày (A).

### R2. CI/CD + Eval gate 🔴 (finding #7, #AI-eng loop)
**Vấn đề:** 429 test nhưng **không có CI**; benchmark/llm_judge rời rạc, không chặn regression.
**Fix:**
- `.github/workflows/ci.yml` (hoặc `.gitlab-ci.yml`): lint (ruff) → `pytest` → build image → Trivy scan.
- **Eval gate:** job chạy `eval/benchmark.py` (DEV_MODE mock hoặc model staging) so baseline → **fail nếu regression** > ngưỡng. Tùy chọn `llm_judge` cho PR.
**Files:** `.github/workflows/`, `eval/ci_gate.py` (wrapper benchmark→exit code).
**Acceptance:** PR fail khi test/lint/benchmark regress. **Effort:** 1-2 ngày.

### R3. Audit durable 🟠→P0 cho compliance (finding #8)
**Vấn đề:** audit best-effort, mặc định SQLite/in-memory; sandbox audit chỉ in-memory.
**Fix:** mặc định Postgres khi `DATABASE_URL` set (đã có backend) + chuyển sandbox audit qua `server.audit` (gửi event tool_execution). Thêm retention cron 90d.
**Files:** `server/audit.py`, `mcp_server/sandbox.py` (emit qua uploader/endpoint), `init-db/002_audit_schema.sql`.
**Acceptance:** mọi exec/auth/security-violation ghi Postgres; restart không mất. **Effort:** 1 ngày.

---

## P1 — Vận hành khép kín

### R4. Async job / non-blocking long tasks 🟠 (finding #5, Scalability)
**Vấn đề:** long task (full index, batch) block request; SSE giữ 1 worker slot.
**Fix (nhẹ trước):** background task + `/jobs/{id}` status (không cần Celery). Long op → trả job_id, chạy nền, poll/stream tiến trình. Cân nhắc tách executor pool.
**Files:** `server/jobs.py`, router `/jobs`.
**Acceptance:** index repo lớn không chiếm request; status query được. **Effort:** 2 ngày.

### R5. Context window — summarize hoặc bỏ 🟡 (finding #2/#13, Coding)
**Vấn đề:** `summarize.py` viết nhưng **không wire**; hội thoại dài chỉ truncate → mất context.
**Fix:** wire `should_summarize` + `truncate_with_summary` vào `_to_openai_messages` **an toàn tool-pairing** (chỉ summarize đoạn không chứa cặp assistant/tool dở dang). Nếu rủi ro cao → **xóa** module để hết dead code.
**Files:** `server/agent/generate.py`, `server/agent/summarize.py`.
**Acceptance:** hội thoại >threshold giữ được goal/context; không vỡ tool_call_id pairing. **Effort:** 1-2 ngày.

### R6. Circuit breaker mở rộng 🟡 (finding, Reliability)
**Fix:** áp `get_circuit_breaker` cho Qdrant (khi RAG on), Postgres (metrics/audit), Redis. Registry đã sẵn.
**Files:** `server/rag/qdrant_client.py`, `server/metrics/counter.py`, `server/redis_client.py`.
**Acceptance:** circuit state hiện trong `/health/deep` cho từng dep. **Effort:** 0.5 ngày.

### R7. Dead code 🟡 (finding #6)
**Fix:** **wire** `parallel_tools.execute_tools_parallel` vào luồng tool (nếu muốn parallel thật) **hoặc xóa** `parallel_tools.py`/`task_queue.py`. Quyết định dứt khoát.
**Acceptance:** không còn module "viết nhưng không gọi". **Effort:** 0.5-1 ngày.

### R8. RAG repo_id isolation 🟠 (finding #3) — *chỉ khi giữ RAG là option*
**Vấn đề:** bật RAG ở multi-tenant → 1 collection chung, không filter repo → privacy leak.
**Fix:** thêm `repo_id` (git remote chuẩn hoá) vào chunk_id + payload + **filter `hybrid_search`** + truyền qua `rag_search`. (Đã phân tích chi tiết trước đó.)
**Files:** `mcp_server/models.py`, `routers/index.py`, `rag/qdrant_client.py`, `agent/rag_search.py`, `routers/chat.py`.
**Acceptance:** query repo A không lôi chunk repo B (test). **Effort:** 1-2 ngày. *(Bỏ nếu quyết định không bật RAG.)*

---

## P2 — Năng lực & định vị

### R9. Multi-agent /agents — build hoặc relabel 🟠 (finding #2)
**Lựa chọn:** (a) **build** theo `docs/multi-agent-design.md` (in-process graph Planner→Researcher→Coder→Reviewer); hoặc (b) **gỡ nhãn "multi-agent"** khỏi mô tả sản phẩm cho đúng thực tế.
**Effort:** build ~3-4 ngày / relabel ~0.

### R10. Cross-session memory 🟠 (finding #4)
**Fix:** `server/agent/memory_store.py` — Postgres + embedding index, `remember/recall`, scope user/project, TTL. (Phase 15.4.)
**Effort:** 3 ngày. *(Chỉ nếu cạnh tranh về "agent nhớ".)*

### R11. RBAC + per-tool permission 🟡 (finding #8)
**Fix:** Permission/Role, scoped API key/JWT, per-tool & per-repo check. (Phase 10.4.)
**Effort:** 2-3 ngày. *(Enterprise gate.)*

### R12. State checkpointing 🟡 (finding #10)
**Fix:** LangGraph checkpointer (Redis/Postgres) → resume long task sau crash.
**Effort:** 1-2 ngày.

### R13. Prompt versioning đầy đủ 🟡 (finding, AI-eng)
**Fix:** migrate **toàn bộ** `INTENT_PROMPTS` sang `config/prompts/intents.yaml` (hiện mới 2 intent), bỏ hardcode.
**Effort:** 0.5 ngày.

---

## Lộ trình đề xuất

```
Sprint 1 (P0):  R1 Sandbox → R2 CI/CD+eval gate → R3 Audit durable
Sprint 2 (P1):  R6 Circuit → R7 Dead code → R5 Context → R4 Async jobs
Sprint 3 (P1/2): R8 RAG repo_id (nếu giữ) → R13 Prompt → R12 Checkpoint
Sprint 4 (P2):  R9 Multi-agent (build/relabel) → R10 Memory → R11 RBAC
```

**Quick wins (rẻ, làm trước được):** R6 (0.5d), R7 (0.5d), R13 (0.5d), R3 (1d).
**Blocker thật sự:** R1 (sandbox) — nên làm đầu tiên.

---

## Định vị lại (khuyến nghị từ review)
Trước khi marketing "multi-agent autonomous enterprise platform": hoàn thành R1-R4 và **quyết R9** (build hay relabel). Định vị trung thực hiện tại = *"reliable single-agent coding assistant for teams"*.

---

*Plan generated: 2026-06-06. Nguồn: Architecture Review cùng ngày.*
