# Phase 11.1 — Redis session + rate-limit (multi-instance)

**Date:** 2026-06-05
**Branch:** `feature/new-architecture`
**Status:** ✅ COMPLETED — 375 tests passing
**Scope:** 11.1 Redis-backed session store + rate limiter (shared across instances), with in-memory fallback

> Đóng nốt mục cuối của Phase 11. (11.2 Redis cache đã hủy theo agentic-first.)

---

## 1. Vì sao

Khi chạy **nhiều instance ai-agent** sau load balancer:
- **Rate limiter** in-memory per-instance → N instance = giới hạn thực tế **N×** (sai). 🔴
- **Session** in-memory → request rơi sang instance khác mất carry-over multi-turn. 🟡

Redis dùng chung giải cả hai. Phục vụ **mọi** request (single-agent là chính).

## 2. Thay đổi

| File | Nội dung |
|------|----------|
| `server/redis_client.py` (mới) | `get_redis()` — kết nối lazy, **None** khi `REDIS_URL` unset / thiếu lib / unreachable (→ in-memory). Không bao giờ raise. `redis_configured()`, `reset_redis()`. |
| `server/rate_limit.py` | `RedisRateLimiter` (fixed-window INCR+EXPIRE, coordinated) + **fallback in-memory** mọi lỗi Redis. Factory `get_rate_limiter()` chọn Redis nếu reachable. |
| `server/session.py` | `RedisSessionStore` (SETEX + touch EXPIRE) + **fallback in-memory**. Factory `get_session_store()` chọn Redis nếu reachable. |
| `server/routers/health.py` | `/health/deep` thêm check `redis` (disabled nếu chưa cấu hình). |
| `requirements.txt` | `redis>=5`. |
| `docker-compose.yml` | service `redis` (profile `scale`); env mẫu `REDIS_URL` (comment). |

**Nguyên tắc:** drop-in — chữ ký `allow()/retry_after()/get()/set()/delete()` giữ nguyên, call site (`chat.py`) **không đổi**. Redis off (mặc định) → hành vi y như cũ.

## 3. Bật multi-instance

```bash
docker compose --profile scale up        # khởi động redis
# trong .env / compose: REDIS_URL=redis://redis:6379/0
# rồi scale: docker compose up --scale ai-agent=N (sau LB)
```
Không set `REDIS_URL` → tự dùng in-memory (1 instance), không cần Redis.

## 4. Degradation
Mọi thao tác Redis lỗi lúc runtime → tự rơi về in-memory (per-instance) thay vì fail request. Redis down không làm sập API.

## 5. Tests
`tests/test_phase11_redis.py` (8): RedisRateLimiter (allow/block, client độc lập, retry_after, fallback), RedisSessionStore (roundtrip, fallback), factory chọn in-memory khi không có Redis. Dùng `FakeRedis`/`BrokenRedis` (không cần lib thật). 367 → **375 passed**.

## 6. Phase 11 — tổng kết

| Mục | |
|-----|---|
| 11.3 Circuit Breaker / 11.4 Degradation / 11.5 Deep Health / 11.6 Retry | ✅ (commit 338b4ff) |
| **11.1 Redis session + rate-limit** | ✅ (commit này) |
| 11.2 Redis caches | ❌ hủy (RAG off) |

→ **Phase 11 hoàn chỉnh** (11.2 không tính theo quyết định agentic-first).

## 7. Trace nhanh
```
server/redis_client.py     (mới) kết nối Redis chịu lỗi
server/rate_limit.py       + RedisRateLimiter + factory chọn backend + reset_rate_limiter
server/session.py          + RedisSessionStore + factory chọn backend
server/routers/health.py   + _check_redis trong /health/deep
docker-compose.yml         + redis (profile scale) + REDIS_URL mẫu
requirements.txt           + redis>=5
tests/test_phase11_redis.py + 8 tests
```

---

*Report generated: 2026-06-05. Liên quan: `report/2026-06-04-phase11-reliability.md`.*
