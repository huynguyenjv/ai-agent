# GitLab Review Runner — Design

**Date:** 2026-04-16
**Status:** Approved for planning

## Problem

AI server chạy ngoài network whitelist của GitLab self-hosted (`gitlab.vinit.tech`). Gọi GitLab API trực tiếp từ AI server → `403 Forbidden`. Không thể whitelist IP của AI server (policy cố định).

## Solution

Đảo ngược hướng gọi. Thay vì AI server gọi GitLab, dùng GitLab CI Runner (IP đã whitelisted) chạy một Docker image (“agent image”) đóng vai orchestrator:

1. Fetch MR diff từ GitLab API.
2. Gửi diff lên AI server qua endpoint mới `POST /review/analyze`.
3. Nhận kết quả review.
4. Post/update comment + inline discussions trên MR.

AI server trở thành stateless: chỉ nhận diff, trả review. Không còn outbound call tới GitLab.

## Architecture

```
GitLab MR event
      │
      ▼
.gitlab-ci.yml (rule: merge_request_event)
      │
      ▼
GitLab Runner ──► Docker: gitlab-review-runner
                     │
                     │ (1) fetch diff          ──► GitLab API
                     │ (2) POST /review/analyze──► AI Server
                     │ (3) review result       ◄── AI Server
                     │ (4) resolve old threads ──► GitLab API
                     │ (5) post summary + inline─► GitLab API
                     ▼
                  exit 0/1
```

**2 deliverables:**

| Component | Repo location | Chức năng |
|---|---|---|
| `POST /review/analyze` endpoint | `server/routers/review.py` (sửa) hoặc file mới | Nhận diff, chạy agent graph, trả markdown + findings + inline_comments |
| Agent image | `gitlab-review-runner/` (folder mới) | Docker image chạy trong GitLab Runner, orchestrate toàn bộ flow |

## Component 1: AI Server Endpoint

### Contract

**Request:** `POST /review/analyze`

```
Headers:
  X-API-Key: <API_KEY>
  Content-Type: application/json

Body:
{
  "repo": "group/project",
  "pr_id": 190,
  "title": "feat: add user auth",
  "source_branch": "feature/auth",
  "target_branch": "main",
  "author": "huynmb",
  "commit_sha": "abc123",
  "base_sha": "def456",
  "head_sha": "abc123",
  "start_sha": "def456",
  "diff": "--- a/file.py\n+++ b/file.py\n@@ ...",
  "files": [
    {"path": "src/auth.py", "status": "modified"}
  ]
}
```

**Response:** `200 OK`

```json
{
  "markdown": "## Code Review Summary\n...",
  "findings": [
    {
      "severity": "high",
      "file": "src/auth.py",
      "line": 42,
      "message": "SQL injection risk",
      "suggestion": "Use parameterized query"
    }
  ],
  "inline_comments": [
    {
      "new_path": "src/auth.py",
      "new_line": 42,
      "old_path": "src/auth.py",
      "body": "⚠️ **High**: SQL injection risk..."
    }
  ],
  "findings_count": {"critical": 0, "high": 1, "medium": 0, "low": 0}
}
```

**Error responses:** `400` (invalid body), `401` (bad API key), `500` (agent failure), `504` (timeout).

### Implementation

- File: `server/routers/review.py` — thêm hàm `analyze_review` bên cạnh `review_pr` hiện tại. Giữ `/review/pr` để backward compat (có thể deprecate sau).
- Pydantic model `ReviewAnalyzeRequest` với các field trên.
- Logic:
  1. Build agent graph (reuse từ `_run_review`).
  2. Inject diff + metadata trực tiếp vào state → bỏ turn fetch diff (turn 1).
  3. Set `auto_post=False` để agent không cố gắng gọi tool `upsert_mr_comment`.
  4. Sau khi agent hoàn tất, extract `draft` (markdown), `review_findings`, và inline comments đã generate từ state.
- Timeout reuse `REVIEW_TIMEOUT_SECS`.
- AI server **không** import `mcp_server.tools_review` cho endpoint này.

### Thay đổi cần thiết trong agent graph

- Node fetch diff: skip nếu state có `pr_context.diff` sẵn.
- Node post comment (`upsert_mr_comment`): skip nếu `auto_post=False`, thay vào đó serialize inline findings vào state field mới `inline_comments`.

## Component 2: Agent Image (`gitlab-review-runner`)

### Folder structure

```
gitlab-review-runner/
├── Dockerfile
├── requirements.txt
├── main.py                    # Orchestrator entrypoint
├── gitlab_client.py           # GitLab API wrapper
├── ai_client.py               # AI server wrapper
├── config.py                  # Load env vars
├── .gitlab-ci.example.yml     # Template cho repo consumer
└── README.md
```

### `config.py` — env vars

Từ GitLab CI auto-inject:
- `CI_PROJECT_PATH` → repo (ví dụ `group/project`)
- `CI_MERGE_REQUEST_IID` → pr_id
- `CI_PIPELINE_SOURCE` → verify = `merge_request_event`

User-provided qua GitLab CI/CD Variables (masked):
- `GITLAB_TOKEN` — PAT với scope `api`
- `AI_SERVER_URL` — ví dụ `https://ai-server.internal`
- `AI_SERVER_API_KEY` — match với `API_KEY` của AI server

Optional:
- `GITLAB_URL` (default = `CI_SERVER_URL`)
- `GITLAB_CA_BUNDLE` — custom CA nếu GitLab dùng self-signed
- `AI_REVIEWER_MARKER` (default `AI_REVIEW_MARKER:v1`)
- `AI_REVIEWER_INLINE_MARKER` (default `AI_REVIEW_INLINE:v1`)
- `HTTP_TIMEOUT` (default 30)

### `gitlab_client.py` — API wrappers

Port từ `mcp_server/tools_review.py`, giữ nguyên signature + retry behavior:

| Function | HTTP |
|---|---|
| `fetch_diff(repo, pr_id)` | GET `/projects/:id/merge_requests/:iid`<br>GET `/projects/:id/merge_requests/:iid/changes?access_raw_diffs=true` |
| `fetch_existing_note(repo, pr_id, marker)` | GET `/projects/:id/merge_requests/:iid/notes` |
| `upsert_comment(repo, pr_id, body, note_id=None)` | POST (create) / PUT (update) `/notes` |
| `list_discussions(repo, pr_id)` | GET `/discussions` (paginated) |
| `resolve_discussion(repo, pr_id, discussion_id)` | PUT `/discussions/:id?resolved=true` |
| `create_inline_discussion(repo, pr_id, body, base_sha, head_sha, start_sha, new_path, new_line, old_path=None)` | POST `/discussions` với `position[*]` params |

Simple retry: 3 attempts, exponential backoff (1s, 2s, 4s) cho 429/5xx + transient errors. Fail-fast trên 4xx khác (403, 404).

### `ai_client.py`

1 function: `analyze(payload: dict) -> dict`
- POST `{AI_SERVER_URL}/review/analyze` với header `X-API-Key`.
- Timeout = `REVIEW_TIMEOUT_SECS + 30` (buffer trên server timeout).
- Raise nếu non-200.

### `main.py` — flow

```python
def main():
    cfg = load_config()
    gl = GitLabClient(cfg)
    ai = AIClient(cfg)

    # 1. Fetch diff
    diff_payload = gl.fetch_diff(cfg.repo, cfg.pr_id)
    if not diff_payload["diff"].strip():
        log.info("empty diff, exiting")
        return 0

    # 2. Call AI server
    result = ai.analyze({
        "repo": cfg.repo,
        "pr_id": cfg.pr_id,
        **diff_payload,
    })

    # 3. Upsert summary comment (with marker embedded in markdown)
    body = f"<!-- {cfg.summary_marker} -->\n{result['markdown']}"
    existing = gl.fetch_existing_note(cfg.repo, cfg.pr_id, cfg.summary_marker)
    gl.upsert_comment(
        cfg.repo, cfg.pr_id, body,
        note_id=existing["note_id"] if existing else None,
    )

    # 4. Resolve old AI inline discussions
    for d in gl.list_discussions(cfg.repo, cfg.pr_id):
        first_note = (d.get("notes") or [{}])[0]
        if cfg.inline_marker in (first_note.get("body") or "") and not first_note.get("resolved"):
            gl.resolve_discussion(cfg.repo, cfg.pr_id, d["id"])

    # 5. Create new inline discussions
    for ic in result.get("inline_comments", []):
        body = f"<!-- {cfg.inline_marker} -->\n{ic['body']}"
        gl.create_inline_discussion(
            cfg.repo, cfg.pr_id, body,
            base_sha=diff_payload["base_sha"],
            head_sha=diff_payload["head_sha"],
            start_sha=diff_payload["start_sha"],
            new_path=ic["new_path"],
            new_line=ic["new_line"],
            old_path=ic.get("old_path"),
        )

    return 0

if __name__ == "__main__":
    sys.exit(main())
```

### Error handling

- Bất kỳ exception nào → log stacktrace, `sys.exit(1)` → GitLab job đỏ → dev thấy trong MR pipeline.
- Optional: trước khi exit 1, post 1 comment ngắn `⚠️ AI review failed: <reason>` để dev biết (không fail silent).

### `Dockerfile`

```dockerfile
FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "main.py"]
```

`requirements.txt`:
```
httpx>=0.27
```

Build & publish: `registry.vinit.tech/ai/gitlab-review-runner:<tag>` qua CI của repo này (separate pipeline, ngoài scope design).

### `.gitlab-ci.example.yml`

```yaml
ai-review:
  stage: review
  image: registry.vinit.tech/ai/gitlab-review-runner:latest
  rules:
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"
  variables:
    AI_SERVER_URL: "https://ai-server.internal"
  script:
    - python /app/main.py
  allow_failure: true
```

`GITLAB_TOKEN` và `AI_SERVER_API_KEY` set qua project CI/CD Variables (type: Variable, Masked, Protected).

## Data Flow Summary

1. Dev push MR → GitLab fires `merge_request_event`.
2. Runner picks up job → pulls `gitlab-review-runner:latest`.
3. Container starts → `main.py` reads `CI_*` vars + user secrets.
4. Fetch MR metadata + diff from GitLab (IP whitelisted).
5. POST payload to AI server `/review/analyze`.
6. AI server runs agent graph with diff injected → returns markdown + findings + inline_comments.
7. Container posts/updates summary note on MR.
8. Container resolves outdated AI inline discussions.
9. Container creates new inline discussions for each finding.
10. Exit 0 → job green.

## Testing Strategy

**AI server:**
- Unit test `POST /review/analyze` với mock agent graph, verify request/response schema.
- Integration test với real vLLM + sample diff.

**Agent image:**
- Unit test `gitlab_client.py` với `httpx.MockTransport`.
- Unit test `ai_client.py` tương tự.
- E2E: chạy container local với sandbox GitLab project + staging AI server.

## Rollout

1. Deploy AI server với endpoint `/review/analyze` (giữ `/review/pr` cùng lúc để không break).
2. Build + push image `gitlab-review-runner:v0.1.0`.
3. Thử trên 1 project test → verify flow end-to-end.
4. Document cách add job vào `.gitlab-ci.yml` cho team.
5. Sau khi ổn định, deprecate `/review/pr` (remove sau 1 release cycle).

## Out of Scope

- Self-hosting GitLab Runner (user nói runner đã sẵn có).
- Queue/worker cho high-volume MRs (1 MR = 1 job ephemeral là đủ).
- Streaming review kết quả (batch response OK cho V1).
- Multi-provider (GitHub/Bitbucket) — V1 chỉ GitLab.
