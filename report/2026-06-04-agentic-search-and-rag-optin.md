# Agentic-Search + RAG Opt-In

**Date:** 2026-06-04
**Branch:** `feature/new-architecture`
**Status:** ✅ COMPLETED — 328 tests passing
**Focus:** Make agentic search the primary context path; demote Qdrant RAG to opt-in

---

## 1. Lý do (Decision)

Bối cảnh: **1 server ai-agent dùng chung cho nhiều user**, mỗi user có repo trên máy riêng.

Vấn đề với RAG/Qdrant trong bối cảnh này:
- **Cold-start**: RAG rỗng cho tới khi có gì đó được index.
- **Isolation**: 1 collection `codebase` chung, không filter theo repo → query của user này lôi code user khác (privacy).
- **Overwrite/stale**: 2 user cùng repo tranh nhau index; code đổi phải re-index.
- **Hạ tầng**: Qdrant + embedder + pipeline `/index`.

Cách Claude Code (Anthropic) giải bài này: **không dùng vector RAG** — model dùng tool (grep/glob/read) tìm & đọc file just-in-time trên filesystem local. Toàn bộ class vấn đề trên **biến mất** vì không có store tập trung.

Hệ ai-agent **đã có sẵn** primitive agentic chạy client-side (luôn fresh, tự cô lập per-user): `search_symbol`, `read_file`, `get_project_skeleton`. Chỉ thiếu **full-text/regex grep**.

→ **Quyết định: agentic-first, RAG opt-in.** Bổ sung `vtrip_grep`, đặt `ENABLE_RAG` mặc định OFF. Giữ nguyên code RAG (Phase 6) để bật lại cho monorepo lớn khi cần.

---

## 2. Thay đổi (What changed)

### 2.1 Tool mới: `vtrip_grep` (agentic full-text search)

**File:** `mcp_server/tools.py` — hàm `grep_content()`

- Pure-Python, ripgrep-style (regex). Đọc file **fresh** từ đĩa, **không** upload Qdrant.
- An toàn: bỏ qua `SKIP_DIRS`, `SKIP_EXTENSIONS` (kể cả multi-part `.min.js`), file binary (`UnicodeDecodeError` → skip), file > 2MB (`GREP_MAX_FILE_BYTES`), chống path-traversal qua symlink (`realpath.startswith(real_repo)`), cap kết quả ≤ 200 (`GREP_MAX_RESULTS_CAP`).
- Tham số: `pattern` (regex, bắt buộc), `path_glob` (lọc file), `ignore_case`, `max_results`.
- Trả: `{matches: [{file_path, line_number, line}], total, files_scanned, truncated, pattern}`.

Helper kèm theo: `_matches_skip_extension()`.

### 2.2 Đăng ký tool (parity client ↔ server)

| File | Thay đổi |
|------|----------|
| `mcp_server/server.py` | import `grep_content`; thêm `Tool(name="vtrip_grep", ...)` vào `list_tools`; thêm nhánh routing `vtrip_grep` trong `call_tool`; cập nhật docstring |
| `server/agent/generate.py` | thêm schema `vtrip_grep` vào `MCP_TOOLS`; thêm dòng vào `TOOL_INSTRUCTIONS`; thêm nudge "explore với tool thay vì đoán"; `TOOL_NAME_MAP`: `grep/ripgrep/rg/search_text/find_in_files/content_search → vtrip_grep` (lưu ý: `grep` trước đây map nhầm sang `search_symbol`, nay sửa đúng) |

### 2.3 RAG opt-in

**File:** `server/routers/chat.py` — `_enable_rag()`

- `ENABLE_RAG` default đổi `"true"` → **`"false"`**. RAG (node `rag_search` + Qdrant) chỉ bật khi đặt `ENABLE_RAG=true`.
- Khi OFF: graph không thêm node `rag_search`; agent lấy context hoàn toàn qua tool client-side. Phần inject RAG context vào prompt (Phase 8.1) chỉ kích hoạt khi có `rag_chunks` → khi RAG off thì no-op (an toàn).

### 2.4 Tests

**File:** `tests/test_agentic_search.py` (11 test)
- `grep_content`: match cơ bản, skip SKIP_DIRS, lọc `path_glob`, case-sensitive, regex lỗi, line number.
- Parity: `vtrip_grep` có trong `MCP_TOOL_NAMES` và được route trong `server.py`; alias `grep/ripgrep → vtrip_grep`.
- RAG opt-in: `_enable_rag()` = False khi không set env, True khi `ENABLE_RAG=true`.

---

## 3. Ảnh hưởng / Migration

- **Mặc định từ nay**: deployment không cần Qdrant + embedder để hoạt động (chat + agentic tool). Nhẹ hơn, riêng tư hơn (code không rời máy user).
- **Để bật lại RAG** (cho repo lớn cần semantic recall): đặt `ENABLE_RAG=true`. **Khi đó** mới cần làm tiếp các việc đã hoãn ở §5.
- **Continue config**: client cần MCP server bản mới để có `vtrip_grep` (đã đăng ký). Các tool cũ không đổi.

---

## 4. Verify

```
python -m compileall -q server mcp_server   # exit 0
python -m pytest                            # 328 passed
```

Trước đó: 317 test. Sau: 328 (+11). Không sửa/đụng test cũ.

---

## 5. Việc đã HOÃN có chủ đích (chỉ làm nếu bật lại RAG)

Các việc này **không cần** khi agentic-first + RAG off. Chỉ kích hoạt nếu sau này `ENABLE_RAG=true` cho monorepo lớn:

1. **Cô lập `repo_id`** — thêm `repo_id` (git remote chuẩn hoá) vào chunk_id + payload Qdrant + filter trong `hybrid_search` + truyền qua `rag_search`. (Giải isolation/overwrite trên collection chung.)
2. **CLI bootstrap index** — `cli.py` lệnh `index <repo>` quét toàn repo (chạy phía client/CI).
3. **Delta indexing + file-watcher** (Phase 13) — tự re-index file đổi theo git diff.

Xem thêm: `docs/improvement-plan.md` Phase 13.

---

## 6. Trace nhanh (files chạm)

```
mcp_server/tools.py        + grep_content(), _matches_skip_extension(), GREP_* consts
mcp_server/server.py       + vtrip_grep (import, list_tools, call_tool, docstring)
server/agent/generate.py   + vtrip_grep schema, TOOL_INSTRUCTIONS, TOOL_NAME_MAP aliases, explore nudge
server/routers/chat.py     ~ _enable_rag() default false (RAG opt-in)
tests/test_agentic_search.py  + 11 tests
```

**Revert RAG về như cũ:** đổi `_enable_rag()` default lại `"true"`.
**Gỡ grep:** xoá `vtrip_grep` ở 3 nơi (tools/server/generate) + test.

---

*Report generated: 2026-06-04. Liên quan: Phase 6 (RAG), Phase 8 (context/caching wiring — 2026-06-04-... đã xong trước đó).*
