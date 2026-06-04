# RAG Lazy-Init — gate Qdrant + Embedder behind ENABLE_RAG

**Date:** 2026-06-04
**Branch:** `feature/new-architecture`
**Status:** ✅ COMPLETED — 352 tests passing
**Decision:** Hướng A — keep RAG dormant (not deleted), but stop loading it when off

---

## 1. Vấn đề

Sau khi chuyển agentic-first (`ENABLE_RAG` default off), `server/app.py` lifespan **vẫn luôn** nối Qdrant + load Embedder (sentence-transformers ~80MB) lúc khởi động, dù RAG không dùng → lãng phí RAM + thời gian startup + ép phụ thuộc Qdrant.

## 2. Thay đổi

- **`server/app.py`**: lifespan chỉ init Qdrant + Embedder **khi `ENABLE_RAG` bật**. Khi off: `app.state.qdrant = None`, `app.state.embedder = None`, log "RAG disabled — skipping". Cleanup `qdrant.close()` guard None. vLLM client luôn init.
  - An toàn: `build_agent_graph(enable_rag=False)` không thêm node `rag_search` nên không đụng qdrant/embedder None. `/index` đã sẵn trả 503 khi embedder None.
- **`docker-compose.yml`**: service `qdrant` → `profiles: ["rag"]` (chỉ chạy với `docker compose --profile rag up`); gỡ `qdrant` khỏi `depends_on` của `ai-agent` (chỉ còn `postgres`).

## 3. Hệ quả

- Mặc định: **không load embedder, không cần Qdrant** → startup nhanh, nhẹ RAM, container không phụ thuộc Qdrant.
- Bật lại RAG: `ENABLE_RAG=true` + `docker compose --profile rag up` (khởi động Qdrant).

## 4. Test

`tests/test_agentic_search.py::TestRagInitGating` — RAG off → `app.state.qdrant/embedder is None`, vLLM client vẫn có. Tổng: 351 → **352 passed**.

## 5. Roadmap (theo quyết định này)

RAG giữ lại dạng tùy chọn (semantic recall cho repo lớn). **Ngừng đầu tư** vào RAG khi off:
- Bỏ/hoãn **Phase 13** (RAG hybrid/delta/AST/file-watcher).
- Bỏ **Phase 11.2** (Redis RAG/embedding cache).
- **repo_id isolation** → moot.

## 6. Trace nhanh

```
server/app.py        ~ lifespan gate Qdrant+Embedder theo ENABLE_RAG; cleanup guard None
docker-compose.yml   ~ qdrant profiles:[rag]; ai-agent depends_on chỉ postgres
tests/test_agentic_search.py  + TestRagInitGating
```

**Revert:** bỏ điều kiện `if rag_enabled` trong lifespan; gỡ `profiles` + thêm lại qdrant depends_on.

---

*Report generated: 2026-06-04. Liên quan: `report/2026-06-04-agentic-search-and-rag-optin.md`.*
