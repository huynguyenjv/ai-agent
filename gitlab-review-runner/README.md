# gitlab-review-runner

Docker image chạy trong GitLab CI Runner, orchestrate AI code review cho Merge Request.

## Flow

```
GitLab MR event → GitLab Runner → this image
    1. Fetch MR diff (GitLab API)
    2. POST diff → AI Server /review/analyze
    3. Upsert summary note trên MR
    4. Resolve các inline discussions cũ của AI
    5. Post inline discussions mới
```

Runner chạy trong network đã whitelist với GitLab. AI server không cần truy cập GitLab.

## Build

```bash
docker build -t registry.vinit.tech/ai/gitlab-review-runner:latest .
docker push registry.vinit.tech/ai/gitlab-review-runner:latest
```

## Env vars

**Auto-injected by GitLab CI:**
- `CI_PROJECT_PATH` — repo path (e.g. `group/project`)
- `CI_MERGE_REQUEST_IID` — MR IID
- `CI_SERVER_URL` — GitLab base URL (fallback cho `GITLAB_URL`)

**Required (set qua Project → Settings → CI/CD → Variables):**
- `GITLAB_TOKEN` — PAT, scope `api`, user phải là member của project
- `AI_SERVER_URL` — ví dụ `https://ai-server.internal`
- `AI_SERVER_API_KEY` — khớp với `API_KEY` của AI server

**Optional:**
- `GITLAB_URL` — override `CI_SERVER_URL`
- `GITLAB_CA_BUNDLE` — path tới CA bundle custom (self-signed cert)
- `AI_REVIEWER_MARKER` — default `AI_REVIEW_MARKER:v1`
- `AI_REVIEWER_INLINE_MARKER` — default `AI_REVIEW_INLINE:v1`
- `HTTP_TIMEOUT` — default `60` giây

## Usage

Xem [.gitlab-ci.example.yml](.gitlab-ci.example.yml). Copy vào `.gitlab-ci.yml` của repo muốn bật AI review.

## Local test

```bash
docker run --rm \
  -e CI_PROJECT_PATH=group/project \
  -e CI_MERGE_REQUEST_IID=190 \
  -e GITLAB_URL=https://gitlab.vinit.tech \
  -e GITLAB_TOKEN=glpat-xxx \
  -e AI_SERVER_URL=https://ai-server.internal \
  -e AI_SERVER_API_KEY=xxx \
  registry.vinit.tech/ai/gitlab-review-runner:latest
```
