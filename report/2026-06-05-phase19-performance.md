# Phase 19 — Performance Optimization

**Date:** 2026-06-05
**Branch:** `feature/new-architecture`
**Status:** ✅ COMPLETED (19.1–19.4) — 420 tests passing

> Lưu ý: phần lớn pooling/disconnect đã có sẵn (AsyncOpenAI/psycopg/redis pool, `is_disconnected` + heartbeat). Đợt này bổ sung phần còn thiếu + utility tái dùng.

---

## 1. 19.2 Connection Pooling — `server/connections.py`

- `build_vllm_client()` tạo AsyncOpenAI với **httpx pool bound** (`VLLM_MAX_CONNECTIONS`=100, keepalive=20, timeout=120) → tránh socket exhaustion khi tải cao.
- **Wire:** `app.py` lifespan dùng `build_vllm_client` thay `AsyncOpenAI(...)` trần.
- (Postgres dùng psycopg pool ở metrics; Redis dùng redis-py pool — đã pooled sẵn.)

## 2. 19.4 Streaming Optimization — `server/streaming/optimized.py`

- `make_event_queue()` → `asyncio.Queue(maxsize=SSE_QUEUE_MAXSIZE=256)` → **backpressure** (producer await khi client chậm; trước đây queue unbounded).
- `TokenCoalescer` — gộp token nhỏ thành chunk ≥ `min_chars` → giảm số SSE frame.
- **Wire:** `chat.py` dùng `make_event_queue()`. (Client-disconnect + heartbeat đã có sẵn.)

## 3. 19.1 Request Batching — `server/batch.py`

- `AsyncBatcher(process_fn, max_batch, max_wait)` — gộp các `submit()` đồng thời thành 1 lần `process_fn(items)` trong cửa sổ thời gian/size; mỗi caller vẫn nhận kết quả riêng.
- Hữu ích cho batch embedding (khi RAG on) hoặc op backend batchable. Utility, chưa wire (RAG off).

## 4. 19.3 Speculative Execution — `server/speculative.py`

- `prewarm_vllm(client, model)` — gửi completion 1 token để warm pool/model. Best-effort, không fatal.
- **Wire:** `app.py` lifespan gọi khi `SPECULATIVE_PREWARM=true` (off mặc định).

---

## 5. Tests

`tests/test_phase19_performance.py` (10): batcher (coalesce/partial-flush/error), pool limits, prewarm (success/fail/disabled), bounded queue + token coalescer. 410 → **420 passed**. compileall exit 0.

## 6. Trace nhanh
```
server/connections.py             (mới) bound vLLM httpx pool
server/streaming/optimized.py     (mới) make_event_queue + TokenCoalescer
server/batch.py                   (mới) AsyncBatcher
server/speculative.py             (mới) prewarm_vllm (gated)
server/app.py                     ~ build_vllm_client + optional prewarm
server/routers/chat.py            ~ make_event_queue (bounded)
tests/test_phase19_performance.py + 10 tests
```

---

*Report generated: 2026-06-05. Liên quan: improvement-plan Phase 19.*
