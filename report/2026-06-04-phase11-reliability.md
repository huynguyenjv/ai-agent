# Phase 11 — State Persistence & Reliability (reliability core)

**Date:** 2026-06-04
**Branch:** `feature/new-architecture`
**Status:** ✅ COMPLETED (reliability core) — 367 tests passing
**Scope:** 11.6 Retry · 11.3 Circuit Breaker · 11.4 Graceful Degradation · 11.5 Deep Health

> Phase 11 thu gọn theo quyết định **agentic-first / RAG opt-in**: bỏ **11.2** (Redis RAG/embedding cache — RAG off). **11.1** (Redis session store) hoãn tới khi chạy multi-instance.

---

## 1. Modules mới

| Mục | File | Nội dung |
|-----|------|----------|
| 11.6 Retry | `server/retry.py` | `retry_async()` exponential backoff + full jitter; `DEFAULT_RETRYABLE` (ConnectionError/TimeoutError/OSError); `compute_delay`, `is_retryable`. Gom logic retry rải rác về 1 chỗ. |
| 11.3 Circuit Breaker | `server/circuit_breaker.py` | `CircuitBreaker` (closed→open→half_open), thread-safe; `allow/record_success/record_failure/call`; registry `get_circuit_breaker(name)` (config qua `CIRCUIT_FAILURE_THRESHOLD`/`CIRCUIT_RECOVERY_TIMEOUT`); `CircuitOpenError`. |
| 11.4 Graceful Degradation | `server/agent/fallback.py` | `llm_unavailable_draft()` trả message thân thiện (VN/EN) shape giống `generate()` (`degraded=True`). |
| 11.5 Deep Health | `server/routers/health.py` | `/health/live` (liveness), `/health/ready` (vLLM reachable→503 nếu không), `/health/deep` (probe vLLM/Qdrant/Postgres + circuit states). Mỗi probe có timeout (`HEALTH_PROBE_TIMEOUT`=3s) → không treo khi backend chết. |

`AgentState` thêm field `degraded`.

## 2. Wiring

- **`generate.py`** (vLLM call):
  - Trước khi gọi: nếu circuit `vllm` **open** → trả ngay fallback (fail-fast, không retry backend đã biết hỏng).
  - Stream xong → `record_success()`; lỗi → `record_failure()`; retry chỉ khi circuit còn `allow()`.
  - Hết retry → trả **fallback graceful** (trước đây trả raw error string + sse error event).
- **`app.py`**: đăng ký `health_router`.

## 3. Tối ưu kèm theo (nối tiếp hướng A)

`server/app.py` trước đây import `Embedder` ở top → kéo `sentence_transformers`/torch (~16s) **dù RAG off**. Đã chuyển import `Embedder`/`QdrantService` **vào trong nhánh `rag_enabled`** của lifespan → RAG off **không import torch**. Khởi động nhẹ hơn; full test suite nhanh hơn (~25s → ~16s).

## 4. Tests

`tests/test_phase11_reliability.py` (15): retry (6), circuit breaker (6), fallback (1), deep health (2). Tổng 352 → **367 passed**. compileall exit 0.

## 5. Còn lại của Phase 11 (hoãn có chủ đích)

| Mục | Lý do hoãn |
|-----|-----------|
| 11.1 Redis session store + rate-limit | Chỉ cần khi chạy **nhiều instance ai-agent** sau LB. Server stateless nên không bắt buộc cho đúng đắn. |
| 11.2 Redis caches (RAG/embedding) | RAG đang off → vô nghĩa. |

Circuit breaker hiện wired cho **vLLM**. Có thể mở rộng cho Qdrant/Postgres khi cần (registry sẵn sàng).

## 6. Trace nhanh

```
server/retry.py                 + retry_async/compute_delay/is_retryable
server/circuit_breaker.py       + CircuitBreaker/registry/CircuitOpenError
server/agent/fallback.py        + llm_unavailable_draft
server/routers/health.py        + /health/live|ready|deep (probe timeouts)
server/agent/generate.py        ~ vLLM call qua circuit breaker + fallback
server/agent/state.py           + degraded field
server/app.py                   + health router; lazy import Embedder/Qdrant (RAG off → no torch)
tests/test_phase11_reliability.py  + 15 tests
```

---

*Report generated: 2026-06-04. Liên quan: improvement-plan Phase 11; agentic-first/RAG opt-in (cùng ngày).*
