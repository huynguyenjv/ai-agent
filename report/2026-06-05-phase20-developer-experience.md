# Phase 20 — Developer Experience

**Date:** 2026-06-05
**Branch:** `feature/new-architecture`
**Status:** ✅ COMPLETED (20.1–20.4) — 429 tests passing

---

## 1. 20.2 Local Dev Mode — `server/dev_mode.py`

- `DEV_MODE=true` → app dùng **MockVLLMClient** (drop-in cho AsyncOpenAI: streaming + non-streaming + `models.list`). Chạy/test full pipeline **không cần model server**. RAG đã off nên local run không cần gì thêm.
- **Wire:** `app.py` lifespan dùng mock khi `is_dev_mode()`.

## 2. 20.3 Integration Tests — `tests/integration/`

- `test_chat_e2e.py`: drive **full LangGraph** (classify→route→planner→generate→verify→post_process) qua FastAPI app thật + mock vLLM (DEV_MODE). Khẳng định: stream ra mock reply, auth bắt buộc (403), input validation (422). Không cần dịch vụ ngoài.

## 3. 20.4 CLI — `cli.py`

- Đã có: `health`, `chat`, `review`, `index`. Thêm:
  - `config` — validate cấu hình CLI + server reachable.
  - `health --deep` — gọi `/health/deep`.
  - `chat --agents` — chèn prefix `/agents` (multi-agent opt-in, xem design doc).

## 4. 20.1 Developer Docs — `docs/DEVELOPMENT.md`

- Setup, **dev mode**, CLI, bảng env, test (unit + integration), docker profiles (rag/scale/observability), troubleshooting.

---

## 5. Tests

`tests/integration/test_chat_e2e.py` (3) + `tests/test_phase20_dx.py` (6: dev mode mock streaming/non-stream/models, CLI config + --agents). 420 → **429 passed**. compileall exit 0.

## 6. Trace nhanh
```
server/dev_mode.py                       (mới) mock vLLM client
tests/integration/test_chat_e2e.py       (mới) e2e full pipeline
tests/test_phase20_dx.py                 (mới) dev mode + cli unit
docs/DEVELOPMENT.md                      (mới) dev guide
server/app.py                            ~ DEV_MODE → mock vLLM
cli.py                                   ~ config cmd, --deep, --agents
```

---

*Report generated: 2026-06-05. Liên quan: improvement-plan Phase 20; [[multi-agent-opt-in-design]].*
