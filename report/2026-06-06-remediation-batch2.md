# Remediation Batch 2 — feature-scale items

**Date:** 2026-06-06
**Branch:** `feature/new-architecture`
**Status:** ✅ DONE — 435 tests passing
**Nguồn:** `docs/remediation-plan.md` (R4/R5/R9/R10/R11)

> Mỗi mục làm thành pass riêng, commit xanh từng cái, để không nhồi ẩu.

---

## R5 — Wire `summarize` an toàn (commit a6b1e1c)
`chat._maybe_summarize`: tóm tắt hội thoại **thuần-chat** dài (>10 msg), **bỏ qua hoàn toàn** khi có bất kỳ tool message → không bao giờ vỡ cặp assistant/tool_call. Off-risk.

## R4 — Async job queue (commit a6b1e1c)
`server/jobs.py` `JobManager` (submit coroutine → job_id, background task, status TTL) + router `/jobs`, `/jobs/{id}`. Long task không block request/SSE slot. Single-instance; Redis/Celery hoá sau.

## R11 — RBAC (commit 899d2cb)
`server/auth_rbac.py`: role (viewer/developer/admin) → permission (read/write/execute/review/admin) → **lọc tool advertise** trong `generate`. Default `developer` (full) ⇒ hành vi không đổi. Config `API_KEY_ROLE` / `API_KEY_ROLES`. Vì tool chạy client-side, điểm enforce server-side là *tool nào được quảng cáo cho model*.

## R10 — Cross-session memory (commit 899d2cb)
`server/agent/memory_store.py`: memory bền vững theo scope (SQLite/Postgres, mirror audit), recall keyword+recency. Wire vào chat (recall đầu / remember cuối), **gate `ENABLE_MEMORY` off mặc định**. `init-db/003_memory_schema.sql`.

## R9 — Multi-agent `/agents` (commit này)
Bản pragmatic, trung thực: `/agents` prefix được parse/strip (`_parse_agents_directive`) → `AgentState.multi_agent` → `_route_after_verify` **luôn** qua critic/reviewer + refine loop (không chỉ task complex). **Tái dùng graph hiện có**, không message bus/agent phân tán (đó vẫn là option nặng tương lai). Định vị đúng: `/agents` = "thorough mode" (luôn review), không phải distributed multi-agent.

---

## R12 — bỏ (server stateless by design; "resume long task" thuộc R4).
## R8 — RAG repo_id: chỉ làm khi bật RAG multi-tenant (đang off) — chưa cần.

---

## Tổng kết remediation (Batch 1 + 2)

| | Done |
|---|------|
| R1 Sandbox RCE guards | ✅ |
| R2 CI/CD + eval gate | ✅ |
| R3 Audit auth failures | ✅ |
| R6 Circuit breaker (Qdrant) | ✅ |
| R7 Dead-code cleanup | ✅ |
| R13 Prompt versioning đầy đủ | ✅ |
| R5 Summarize (safe) | ✅ |
| R4 Async jobs | ✅ |
| R11 RBAC | ✅ |
| R10 Cross-session memory | ✅ |
| R9 Multi-agent /agents | ✅ |
| R8 RAG repo_id | ⏭️ chỉ khi bật RAG |
| R12 State checkpoint | ❌ bỏ (stateless) |

**11/13 done, 1 conditional, 1 dropped-by-design. 435 tests passing.**

---

*Report generated: 2026-06-06.*
