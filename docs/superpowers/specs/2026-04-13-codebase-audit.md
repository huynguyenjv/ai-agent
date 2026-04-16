# Codebase Audit — Gaps, Bugs, Production Readiness

**Date**: 2026-04-13
**Scope**: Toàn bộ codebase (A) + production readiness (C)
**Reviewer**: senior-level audit
**Status**: Draft — chờ chốt để chuyển sang implementation plan

---

## 0. TL;DR — Top blocking items trước khi lên prod

| # | Vấn đề | Mức | File |
|---|--------|-----|------|
| 1 | **Secret GitLab token đã commit vào `.env` trong repo** | **CRITICAL** | `.env:24` |
| 2 | Thiếu `.dockerignore` → build image có thể ăn `.env`, `venv/`, `.git/` | **CRITICAL** | root |
| 3 | `verify=False` trên GitLab httpx client (tools_review) → MITM risk | HIGH | `mcp_server/tools_review.py:45,79,92` |
| 4 | Không có timeout cho `agent.ainvoke` trong `/review/pr` → hang = holding connection vô thời hạn | HIGH | `server/routers/review.py` |
| 5 | Không có timeout cho LLM call trong `review_analyze` → cùng vấn đề | HIGH | `server/agent/review_analyze.py:81` |
| 6 | `plan_steps` timeout 800ms không thực tế với model 32B AWQ → plan silently fail mỗi request code_gen | HIGH | `server/agent/plan_steps.py:77` |
| 7 | `hybrid_search` dùng `SparseVector` truyền thẳng vào `query_points(query=...)` — qdrant-client API đang chuyển sang `prefetch` cho hybrid, có thể break | HIGH | `server/rag/qdrant_client.py:192` |
| 8 | Client disconnect không cancel agent task trong chat streaming → resource leak | MEDIUM | `server/routers/chat.py:183` |
| 9 | `_strip_leaked_prompt` dùng heuristic `idx < 100` — fragile, có thể mất content thật | MEDIUM | `server/agent/post_process.py:54` |
| 10 | CI pipeline luôn `allow_failure: true` → không thể fail build theo blocker | MEDIUM | `.gitlab-ci.yml` |

---

## 1. Phương pháp review

Audit chia thành 4 lớp:
- **L1 Security & Secrets** — lộ token, SSL, injection, auth bypass
- **L2 Correctness & Bugs** — logic sai, race, error path
- **L3 Robustness & Ops** — timeout, retry, cancellation, observability
- **L4 Architecture & Maintainability** — boundaries, dead code, tech debt

Mỗi finding có: severity (blocker/major/minor), location, evidence, fix đề xuất.

---

## 2. L1 — Security & Secrets

### 2.1 [CRITICAL] GitLab token commit vào `.env`
**Location**: `.env:24` → `GITLAB_TOKEN=`

**Evidence**: `.env` có giá trị thật của Personal Access Token GitLab, file đang track trong git (theo `git status` đầu conversation `M .env`).

**Impact**: Bất kỳ ai clone repo hoặc xem git history → full API access vào `gitlab.vinit.tech` dưới quyền maintainer. MCP tool có thể đọc toàn bộ source, mở/đóng MR, push comment, thậm chí modify repo tùy scope PAT.

**Fix**:
1. Revoke token hiện tại trên GitLab ngay.
2. Tạo `.env.example` chứa key names, value rỗng. Di chuyển giá trị thật sang `.env` local (không track).
3. Thêm vào `.gitignore`: `.env`, `.env.prod`, `.env.*.local`.
4. Chạy `git filter-repo` hoặc `git rm --cached .env` + commit. Nếu đã push → force-push hoặc rotate toàn bộ secrets.
5. Audit git history cho các secret khác (API_KEY cũng có trong `.env`).

### 2.2 [CRITICAL] Thiếu `.dockerignore`
**Location**: root, không có file `.dockerignore`

**Evidence**: `Dockerfile:17` → `COPY . .` — copy toàn bộ context.

**Impact**: Image prod có thể chứa `.env` (secrets), `venv/` (vài trăm MB), `.git/` (history + secrets cũ), `logs/`, `.claude/`, `tests/`.

**Fix**: Tạo `.dockerignore`:
```
.env
.env.*
venv/
.venv/
.git/
.gitignore
__pycache__/
*.pyc
.pytest_cache/
.claude/
logs/
tests/
docs/
*.md
.agent/
.vscode/
.idea/
```

### 2.3 [HIGH] SSL verification bị tắt cho GitLab
**Location**: `mcp_server/tools_review.py:45,79,92` → `httpx.AsyncClient(timeout=..., verify=False)`

**Evidence**: Mọi request tới GitLab bypass cert check.

**Impact**: MITM có thể intercept token + inject response. Đặc biệt nguy hiểm khi agent chạy qua proxy corporate.

**Fix**:
- Default `verify=True`.
- Nếu cần support self-signed (gitlab nội bộ), add env `GITLAB_CA_BUNDLE=/path/to/ca.pem`, truyền `verify=os.environ.get("GITLAB_CA_BUNDLE", True)`.

### 2.4 [MEDIUM] CORS `*` fallback trong prod
**Location**: `server/app.py:84` → `CORS_ORIGINS=*` mặc định

**Evidence**: Nếu env không set, expose mọi origin.

**Fix**: Fail-closed trong prod — raise lỗi nếu `CORS_ORIGINS` không set khi `APP_ENV=production`, hoặc default `[]` thay vì `*`.

### 2.5 [MEDIUM] `read_file` path traversal check có thể bypass bằng symlink
**Location**: `mcp_server/tools.py:36-39`

**Evidence**: Dùng `realpath` + `startswith(real_repo)`. OK với tuyệt đại đa số case, nhưng nếu `real_repo` không có trailing separator, path `/repoXXX/evil` có thể pass check vì `startswith`.

**Fix**: `if not real_file.startswith(real_repo + os.sep) and real_file != real_repo: return error`.

### 2.6 [LOW] Token log leak risk
**Location**: `server/logging_config.py` (chưa xem nội dung), `chat.py:105` log full request metadata.

**Fix**: Đảm bảo không log body có chứa Authorization / X-Api-Key. Thêm middleware redact.

---

## 3. L2 — Correctness & Bugs

### 3.1 [HIGH] `plan_steps` timeout 800ms không khả thi với Qwen 32B AWQ
**Location**: `server/agent/plan_steps.py:77` → `timeout=0.8`

**Evidence**: Model 32B AWQ trên GPU thông thường có TTFT ~500-1500ms + generate 400 tokens ≈ 3-10s. 800ms tổng → 100% timeout.

**Impact**: Mọi request `code_gen` gọi plan_steps đều timeout, log warning, tool_results không có `plan`. `generate` chạy không plan → chất lượng giảm.

**Fix**:
- Tăng timeout lên 8-15s.
- Hoặc bỏ plan_steps cho V1 (YAGNI) — generate trực tiếp đã đủ.
- Hoặc dùng model nhỏ hơn cho planning (Qwen 1.5B).

### 3.2 [HIGH] `hybrid_search` API có thể lỗi với qdrant-client mới
**Location**: `server/rag/qdrant_client.py:175-202`

**Evidence**: Dùng `query_points(query=dense_vector, using="dense")` và `query_points(query=SparseVector(...), using="sparse")`. Qdrant-client 1.10+ khuyến nghị dùng `prefetch` + `query` top-level cho hybrid, và tự RRF bằng `FusionQuery(fusion=Fusion.RRF)`.

**Impact**: Tùy version client — version cũ có thể accept, version mới warn/fail.

**Fix**:
- Pin `qdrant-client` version (hiện `>=1.12`).
- Hoặc migrate sang `query_points` với `prefetch=[...]` + `query=FusionQuery(fusion=RRF)`, loại bỏ RRF tính tay.
- Viết integration test đụng Qdrant thật.

### 3.3 [HIGH] `rag_search` `hash_verified` semantic ngược
**Location**: `server/agent/rag_search.py:96` → comment "Not verified, but usable" nhưng trả `hash_verified=False` khi force_reindex active

**Evidence**: Gate 5 design — khi `force_reindex=True` → chunks đang stale, client đang reindex parallel → `hash_verified` nên là `False` (đúng hiện tại) nhưng semantics "stale-while-revalidate" không được downstream dùng. `generate` không check field này.

**Impact**: Field dead — người đọc code nhầm tưởng có verification flow.

**Fix**:
- Hoặc xóa field + logic (YAGNI).
- Hoặc implement đúng: downstream `generate` degrade confidence khi `hash_verified=False` (thêm disclaimer vào prompt).

### 3.4 [MEDIUM] `post_process._strip_leaked_prompt` heuristic fragile
**Location**: `server/agent/post_process.py:54` → `if idx < 100`

**Evidence**: Nếu LLM chèn "## Codebase Context" ở giữa response (hợp lệ — user đang hỏi về context), logic skip nhưng check `idx < 100` có thể false negative. Nếu LLM thật sự leak prompt ở char 150 → không strip.

**Fix**:
- Strip-safe: chỉ match ở đầu buffer (`text.lstrip().startswith(marker)`).
- Hoặc remove logic — dùng prompt system đủ tốt để LLM không leak, không band-aid ở post-process.

### 3.5 [MEDIUM] `chat.py` Turn 2 mất `tool_calls` trên AIMessage
**Location**: `server/routers/chat.py:121-131`

**Evidence**: Convert mọi non-user/non-tool msg thành `AIMessage(content=msg.content or "")`. Continue Turn 2 gửi `[user, assistant (tool_calls=[...]), tool (result)]`. Message assistant với `tool_calls` bị mất field `additional_kwargs={"tool_calls": ...}`.

**Impact**: Nếu downstream node nào cần match tool_call_id ↔ tool_calls (để biết tool nào gọi tool nào), sẽ thất bại. Hiện tại `review_analyze` không dùng nên chưa lộ bug. Sẽ lộ khi add multi-tool chain.

**Fix**:
```python
elif msg.role == "assistant":
    extra = {"tool_calls": msg.tool_calls} if msg.tool_calls else {}
    messages.append(AIMessage(content=msg.content or "", additional_kwargs=extra))
```

### 3.6 [MEDIUM] `review_analyze` không handle Turn 2 đúng khi LLM fail
**Location**: `server/agent/review_analyze.py:113-120`

**Evidence**: Nếu 2 lần LLM đều trả JSON sai, `findings = []`. Agent vẫn chạy tiếp, render "No issues found" → post lên MR. User không biết LLM fail.

**Fix**:
- Trả thêm `review_error` field trong state.
- `review_format` render error banner nếu `review_error` set.

### 3.7 [MEDIUM] `classify_intent` `code_review` ưu tiên cao có thể cannibalize intent khác
**Location**: `server/agent/classify_intent.py:18-22`

**Evidence**: Pattern `review\s*(?:file|pr|mr|code|diff)` match cả "write a review for code generation feature" — không phải ý người dùng.

**Fix**:
- Thu hẹp pattern: require URL MR hoặc keyword "audit" trực tiếp.
- Hoặc đặt code_review xuống thấp hơn unit_test/refine.

### 3.8 [LOW] `classify_intent` Vietnamese không có diacritics fallback
**Evidence**: `"review lại"` dùng diacritic. User gõ "review lai" không match → falls back code_gen.

**Fix**: Thêm variants không dấu, hoặc normalize text trước khi match (`unicodedata.normalize('NFKD')`).

### 3.9 [LOW] `_parse_previous_reviews` regex brittle
**Location**: `server/agent/review_format.py:41`

**Evidence**: Nếu user tay tự chỉnh comment (thêm dòng, đổi format), parse fail → mất history mãi mãi.

**Fix**: Store history as fenced JSON block trong comment body: `<!-- HISTORY: [...] -->`. Parse JSON, không regex.

### 3.10 [LOW] `review_format` field `pr_context.previous_reviews` ghi nhưng không đọc lại
**Location**: `review_format.py:110` ghi `next_history`, nhưng request sau là fresh state → field chết.

**Fix**: Xoá `next_history` logic. Chỉ parse từ MR body là đủ.

---

## 4. L3 — Robustness & Ops

### 4.1 [HIGH] Không có timeout cho `/review/pr` và `review_analyze`
**Location**: `server/routers/review.py`, `server/agent/review_analyze.py:81`

**Evidence**: Nếu vLLM hang hoặc GitLab API hang → request HTTP block đến FastAPI default (không có), uvicorn default. CI job kéo dài.

**Fix**:
- Wrap `agent.ainvoke` bằng `asyncio.wait_for(..., timeout=REVIEW_TIMEOUT_SEC=150)`.
- LLM call trong review_analyze: `asyncio.wait_for(..., timeout=60)`.
- Return HTTP 504 nếu timeout.

### 4.2 [MEDIUM] Chat streaming không cancel agent task khi client disconnect
**Location**: `server/routers/chat.py:183`

**Evidence**: `asyncio.create_task(_run_agent())` không lưu ref, không check `req.is_disconnected()`. Client đóng kết nối → agent vẫn chạy tới khi xong, tốn GPU.

**Fix**:
```python
agent_task = asyncio.create_task(_run_agent())
try:
    while True:
        if await req.is_disconnected():
            agent_task.cancel()
            break
        ...
finally:
    if not agent_task.done():
        agent_task.cancel()
```

### 4.3 [MEDIUM] `upsert_mr_comment` retry không có backoff
**Location**: `server/agent/upsert_mr_comment.py:58`

**Evidence**: Loop range(2), không sleep giữa attempts. GitLab 5xx tạm thời → retry ngay → fail lần 2.

**Fix**: `await asyncio.sleep(1)` trước attempt 2. Thêm jitter nếu nhiều worker.

### 4.4 [MEDIUM] GitLab API không handle 429 rate limit
**Location**: `mcp_server/tools_review.py`

**Evidence**: `resp.raise_for_status()` raise trên 429 → agent fail.

**Fix**: Check `Retry-After` header, sleep, retry 1x. Hoặc dùng `tenacity` với `retry_if_exception_type(httpx.HTTPStatusError)`.

### 4.5 [MEDIUM] Không có observability meaningful
**Evidence**:
- Không có Prometheus metrics dù README nói có `/metrics`.
- Correlation ID có nhưng chỉ log, không propagate vào vLLM/Qdrant call.
- Không log duration từng node LangGraph.

**Fix**:
- Add `prometheus-fastapi-instrumentator` → expose `/metrics` với request latency, intent distribution, tool_call count.
- Log duration mỗi node bằng LangGraph callback hoặc wrap thủ công.
- Metric đặc thù review: `code_review_findings_total{severity}`, `code_review_duration_seconds`.

### 4.6 [MEDIUM] `/review/pr` sync response, PR lớn có thể timeout proxy
**Evidence**: Nginx timeout default 60s, review có thể 30-120s.

**Fix**: Đã nới timeout trong `nginx.conf` route `/review/` (đã loại khỏi compose prod đơn giản). Với deploy không có nginx custom: yêu cầu proxy server của user nâng `proxy_read_timeout` cho `/review/pr`. Hoặc chuyển sang job mode: `POST /review/pr` enqueue → trả `{job_id}`; `GET /review/pr/{job_id}` poll.

### 4.7 [MEDIUM] `/review/pr` không có rate limit
**Evidence**: CI job gọi không giới hạn → spam GitLab API.

**Fix**:
- Dedupe ở server: cache `(repo, pr_id, commit_sha)` trong 60s, nếu trùng → trả 409 Conflict.
- Hoặc rate limit theo IP bằng `slowapi`.

### 4.8 [MEDIUM] Không healthcheck cho dependency (vLLM, Qdrant)
**Location**: `server/app.py:124` → `/health` chỉ trả `{"status": "ok"}`.

**Evidence**: Nếu Qdrant/vLLM down, health vẫn OK → load balancer không biết.

**Fix**:
- `/health/live` — process alive (như hiện tại).
- `/health/ready` — ping Qdrant + vLLM, trả 503 nếu chết.

### 4.9 [LOW] Log format switch nhưng JSON formatter chưa xem code
Cần verify `server/logging_config.py` emit JSON đúng format (timestamp RFC3339, level, correlation_id, message).

### 4.10 [LOW] CI `.gitlab-ci.yml` luôn `allow_failure: true`
**Fix**: Thêm check `findings_count.blocker > 0` và `exit 1` nếu cần fail pipeline cứng. Gắn sau một env flag `REVIEW_FAIL_ON_BLOCKER=1` để không ép tất cả team.

---

## 5. L4 — Architecture & Maintainability

### 5.1 [MEDIUM] `generate.py` BASE_SYSTEM_PROMPT không đa-intent
**Evidence**: 1 prompt system duy nhất cho code_gen/unit_test/explain/structural_analysis. Không tận dụng được intent đã classify.

**Fix**: Map intent → prompt template. Ví dụ `unit_test` thêm chỉ dẫn JUnit5+Mockito; `explain` khuyến khích dạy dỗ step-by-step.

### 5.2 [MEDIUM] README lệch thực tế codebase
**Evidence**: README mô tả `agent/subgraphs/unit_test.py`, `agent/graph_adapter.py`, `indexer/`, `intelligence/`, `config/agent.yaml` — không tồn tại. Luồng 7-pass validation / 3-level repair / human_review không có code.

**Fix**: Viết lại README theo thực tế (đã có `docs/CODEBASE_OVERVIEW.md`, copy content). Hoặc xoá README cũ, link sang overview.

### 5.3 [MEDIUM] Boundaries giữa `server/agent/` và `mcp_server/tools_review.py` blur
**Evidence**: `server/agent/upsert_mr_comment.py` import `mcp_server.tools_review` → server package phụ thuộc mcp_server package. MCP là client-side tool, server không nên import.

**Fix**:
- Tách `tools_review.py` thành `server/integrations/gitlab.py` (logic thuần), mcp_server expose wrapper gọi sang.
- Hoặc chấp nhận coupling (agent cần GitLab API trực tiếp cho upsert), đổi tên package `mcp_server` → `integrations` cho rõ.

### 5.4 [LOW] Dead code `hash_verifier.py`
**Evidence**: Import trong `rag_search.py:13` nhưng không dùng class.

**Fix**: Xoá import, và xem file `hash_verifier.py` có được dùng ở đâu khác không. Nếu không → xoá file.

### 5.5 [LOW] `AgentState` TypedDict gốc `total=True` (mặc định) → tôi đã chuyển sang `total=False` khi add code review fields. Code cũ có thể build state không đầy đủ field. Verify test `test_agent_state_new_fields` vẫn pass.

### 5.6 [LOW] `continue_compat.py` regex file extension hardcode
**Evidence**: Chỉ match 13 extension cụ thể. Thiếu `.kts, .scala, .rs (có nhưng partial), .php, .rb`.

**Fix**: Lấy extension từ `EXT_TO_LANG` ở `rag_search.py` để đồng bộ.

### 5.7 [LOW] Duplicate regex file pattern ở `route_context.py` và `continue_compat.py`
**Fix**: Extract thành module `server/util/file_patterns.py`.

---

## 6. Code review feature — gaps riêng

### 6.1 [MEDIUM] Prompt review không có "few-shot"
**Evidence**: `SYSTEM_PROMPT` chỉ rule + schema. LLM có thể hallucinate severity.

**Fix**: Thêm 1-2 example finding format trong prompt để calibrate.

### 6.2 [MEDIUM] PR lớn không thực sự split theo file
**Evidence**: `review_analyze.py:101` chỉ truncate char-based → cut giữa hunk, LLM nhận diff cắt cụt.

**Fix**: Parse diff theo file (GitLab API đã trả `changes[]`), loop từng file, merge findings. Nếu 1 file vẫn > threshold → skip với finding info "file too large".

### 6.3 [MEDIUM] Thiếu test cho luồng code review
**Evidence**: `test_new_arch.py` không có test review_analyze / review_format / upsert_mr_comment.

**Fix**: Thêm unit test:
- `review_format` snapshot test (fixture findings → expected markdown).
- `review_analyze` mock LLM → valid JSON / invalid JSON / retry.
- `_parse_previous_reviews` roundtrip.
- `/review/pr` integration với mock httpx cho GitLab.

### 6.4 [LOW] `severity` từ LLM không validate
**Evidence**: Nếu LLM trả `"severity": "critical"` → không rơi vào `_SEVERITY_ORDER`, rank 99, render không có emoji.

**Fix**: Whitelist severity trong `review_analyze` normalise, map unknown → "major".

---

## 7. Production deployment readiness

### 7.1 Checklist bắt buộc trước go-live

- [ ] Rotate GitLab token + remove khỏi git (item 2.1).
- [ ] `.dockerignore` (item 2.2).
- [ ] Bật SSL verify (item 2.3).
- [ ] Timeout wrapping cho `/review/pr` + LLM calls (4.1).
- [ ] Client disconnect handling (4.2).
- [ ] `/health/ready` check dependencies (4.8).
- [ ] Pin qdrant-client version + test hybrid_search (3.2).
- [ ] Fix plan_steps timeout hoặc bỏ (3.1).
- [ ] Tests cho code review flow (6.3).
- [ ] README align thực tế (5.2).

### 7.2 Nên có (không blocking)

- Metrics `/metrics` thật (4.5).
- Rate limit `/review/pr` (4.7).
- Dedupe commit_sha 60s (4.7).
- Few-shot prompt review (6.1).
- Split diff theo file (6.2).

### 7.3 Deploy recommendations

- Prod compose: đã gọn (chỉ 1 container). Cần `.dockerignore` + thay `.env.prod.example` với placeholder.
- Healthcheck Docker: đã có nhưng chỉ ping `/health` — tốt khi có `/health/ready` thì update CMD healthcheck sang endpoint đó.
- Logging: bật `LOG_FORMAT=json` (đã set trong compose). Verify output đúng JSON.
- Log rotation: đã set 50MB × 5 files, OK.

---

## 8. Đề xuất thứ tự triển khai fix (sprint-sized)

### Sprint 1 — Security & must-have trước prod (2-3 ngày)
1. Rotate GitLab token, remove `.env` khỏi git, tạo `.env.example`, update `.gitignore`.
2. Tạo `.dockerignore`.
3. Bật SSL verify (env override cho CA bundle nếu internal).
4. Timeout wrapping: `/review/pr`, `review_analyze`, `generate` stream (async_timeout).
5. `plan_steps` tăng timeout 10s hoặc remove hoàn toàn.
6. Client disconnect handling chat streaming.

### Sprint 2 — Correctness & coverage (3-5 ngày)
7. Fix `chat.py` Turn 2 preserve tool_calls trong AIMessage.
8. Viết test cho review flow (unit + integration mock).
9. Diff split theo file trong `review_analyze`.
10. Few-shot trong prompt review.
11. Align README với thực tế (hoặc delete, point về CODEBASE_OVERVIEW).

### Sprint 3 — Robustness & ops (3-5 ngày)
12. `/health/ready` ping Qdrant + vLLM.
13. Prometheus metrics.
14. Dedupe/rate limit `/review/pr`.
15. 429 + backoff retry cho GitLab API.
16. Strip dead code (`hash_verifier`, `next_history` logic).

### Sprint 4 — Nice-to-have (tùy)
17. Refactor boundaries (server/integrations/ vs mcp_server/).
18. Intent-specific prompts.
19. Async job mode cho `/review/pr` (poll-based).
20. Webhook GitLab real-time.

---

## 9. Câu hỏi mở cần chốt

1. **Token leak**: anh đã revoke token chưa? Có muốn tôi `git rm --cached .env` + commit + force-push để xoá khỏi history?
2. **plan_steps**: giữ (tăng timeout) hay xóa cho V1?
3. **Async job mode `/review/pr`**: làm V1 (đỡ timeout) hay sync tạm đủ?
4. **Prometheus metrics**: anh có Prometheus sẵn không? Nếu không thì skip V1.
5. **Test framework**: dùng pytest + mock httpx, ok?
6. **Pass/fail pipeline**: cho phép block merge khi có blocker finding, hay chỉ comment?

---

**Next step**: anh review, chọn các item muốn đưa vào plan. Tôi dùng `writing-plans` skill convert thành implementation plan có task breakdown cụ thể.
