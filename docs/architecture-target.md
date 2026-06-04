# Kiến trúc đích (Target Architecture) — AI Coding Agent

> **Phiên bản đích:** 3.0.0 (TO-BE) — sau khi hoàn thành toàn bộ `docs/improvement-plan.md`
> **Cập nhật:** 2026-06-04
> **Điểm mục tiêu:** 9.5/10 (hiện tại ~7.5/10)

> ⚠️ **Đây là tài liệu THIẾT KẾ ĐÍCH, không phải trạng thái hiện tại.**
> Tài liệu mô tả hệ thống *sau khi* triển khai xong Phase 10 → 20. Trạng thái đang chạy hiện tại xem `docs/architecture.md`.
> Các thành phần được gắn nhãn `[Pxx]` để truy vết về phase tương ứng trong improvement-plan.

---

## 1. Tổng quan

AI Coding Agent v3.0 là một **self-hosted multi-agent coding platform** chạy phân tán, expose API tương thích OpenAI, tích hợp IDE (Continue/Tabby) và CI (GitLab). So với v2.0 (single-instance, single-agent, in-memory), kiến trúc đích bổ sung:

- **Multi-agent collaboration** `[P15]` — coordinator điều phối các agent chuyên biệt (Researcher / Coder / Reviewer / Planner / Executor) qua message bus.
- **State persistence phân tán** `[P11]` — Redis cho session/cache, circuit breaker, graceful degradation.
- **Horizontal scaling** `[P12, optional]` — Kubernetes + HPA + Celery worker + Qdrant cluster.
- **Hybrid RAG** `[P13]` — semantic + BM25 + delta indexing + cached AST + file watcher.
- **Full observability** `[P14]` — OpenTelemetry + Jaeger + Sentry + SLO dashboards.
- **AI engineering loop** `[P16]` — LLM-as-Judge, A/B testing, trace→dataset, prompt versioning.
- **Tool nâng cao** `[P17]` — multi-file atomic edits, call graph, LSP, tool-result validation.
- **Enterprise** `[P18]` — token budget, multi-tenant, SSO/OIDC, backup, GitOps, CI/CD.
- **Security hardening** `[P10]` — sandbox, input guard, secret scanning, audit log, input validation.

### Nguyên tắc giữ nguyên từ v2.0
1. **Stateless request handling** — client gửi full history; state chia sẻ qua Redis (không sticky session).
2. **Native tool-call** — model tự quyết định gọi tool, có lớp tolerance name/arg.
3. **LangGraph orchestration** — state machine có conditional routing + retry loop.
4. **Stateless code review** — server không gọi GitLab; `gitlab-review-runner` lo I/O.
5. **Always-SSE streaming**.

---

## 2. Sơ đồ kiến trúc tổng thể (TO-BE)

```
                       ┌────────────────────────────────────────────────┐
                       │  Clients                                        │
                       │   Continue / Tabby IDE   │  GitLab CI Runner    │
                       │   + MCP server (stdio, client-side)             │
                       └───────────────┬─────────────────┬───────────────┘
                                       │ HTTPS/SSE        │ /review/analyze
                                       ▼                  ▼
                       ┌────────────────────────────────────────────────┐
                       │  Ingress / LoadBalancer  [P12]                  │
                       │   TLS · sticky(SSE) · rate-limit · backpressure │
                       └───────────────┬─────────────────────────────────┘
                                       ▼
        ┌───────────────────────────────────────────────────────────────────────┐
        │  AI-Agent API Pods (HPA 2..10)  [P12]                                   │
        │  ┌──────────────────────────────────────────────────────────────────┐ │
        │  │ Middleware: OIDC/SSO · tenant ctx · correlation-id · OTel · audit  │ │
        │  │             input-validation · input-guard · rate-limit · budget   │ │
        │  └──────────────────────────────────────────────────────────────────┘ │
        │  Routers: /v1/chat · /index · /review/analyze · /metrics · /feedback   │
        │           /health(deep) · /experiments · /admin                        │
        │                              │                                          │
        │                              ▼                                          │
        │  ┌──────────────────── Agent Coordinator [P15] ───────────────────┐   │
        │  │  Planner ─▶ Researcher ─▶ Coder ─▶ Reviewer ─▶ Executor          │   │
        │  │  (mỗi agent là một LangGraph subgraph; giao tiếp qua Message Bus)│   │
        │  └──────────────────────────────────────────────────────────────────┘ │
        └───┬───────────────┬──────────────┬───────────────┬──────────────┬──────┘
            │ async jobs     │ LLM           │ RAG            │ state        │ tools
            ▼                ▼               ▼                ▼              ▼
   ┌─────────────┐   ┌─────────────┐ ┌───────────────┐ ┌──────────┐ ┌─────────────┐
   │ Celery      │   │ vLLM pool   │ │ Hybrid RAG    │ │ Redis    │ │ Sandboxed   │
   │ workers[P12]│   │ +CircuitBrk │ │ Qdrant+BM25   │ │ session/ │ │ tool exec   │
   │ index/batch │   │ [P11/P19]   │ │ +AST cache    │ │ cache    │ │ +validator  │
   └─────────────┘   └─────────────┘ │ [P13]         │ │ [P11]    │ │ [P10/P17]   │
                                     └───────┬───────┘ └────┬─────┘ └─────────────┘
                                             ▼              ▼
                                     ┌───────────────┐ ┌──────────────────────────┐
                                     │ Qdrant cluster│ │ Postgres (metrics/audit/ │
                                     │ 3 replicas    │ │ budget/memory/dataset)   │
                                     │ [P12]         │ │ [P10/P16/P18]            │
                                     └───────────────┘ └──────────────────────────┘

   Observability [P14]: OpenTelemetry → Jaeger · Prometheus → Grafana(SLO) · Sentry · Loki
   GitOps [P18]: ArgoCD ← Helm ← CI/CD pipeline (build · test · Trivy · deploy)
```

---

## 3. Topology triển khai `[P12, P18]`

| Lớp | Thành phần | Ghi chú |
|-----|-----------|---------|
| Edge | Ingress + TLS, LoadBalancer | sticky session cho SSE, rate-limit ở LB |
| App | Deployment `ai-agent` (HPA 2→10) | scale theo CPU/mem + custom metric `ai_agent_active_requests`; PodDisruptionBudget minAvailable=1; pre-stop hook graceful shutdown |
| Worker | Celery workers | long task: full-repo index, batch codegen, delta reindex; priority queue; Flower monitoring |
| LLM | vLLM (pool) | bọc Circuit Breaker + connection pool |
| Vector | Qdrant StatefulSet ×3 | replication factor, shard, backup CronJob |
| State | Redis | session store + cache + message bus + Celery broker |
| RDBMS | Postgres | metrics, audit log, token budget, agent memory, dataset, experiments |
| GitOps | ArgoCD + Helm | environment overlays, sync policy, rollback tự động |

> **Lưu ý phạm vi:** Trong `improvement-plan.md`, Phase 12 (K8s/HPA/Celery/Qdrant cluster) và một số mục Phase 18 được đánh dấu *optional / chưa cam kết triển khai*. Tài liệu đích vẫn mô tả để hình dung trạng thái "hoàn thiện đầy đủ"; nếu giữ deployment đơn giản thì các pod/worker rút về docker-compose như v2.0 nhưng vẫn dùng Redis + Postgres.

---

## 4. Vòng đời một request (chat)

```
1. Ingress → TLS, rate-limit, route
2. Middleware: OIDC verify → tenant context → correlation-id → OTel span start
3. Input validation (Pydantic) → input-guard (prompt injection) → token-budget pre-check [P10/P18]
4. Load session state từ Redis (theo conversation_id, scoped theo tenant) [P11]
5. Agent Coordinator chọn workflow theo intent [P15]
      Planner → (Researcher: hybrid RAG) → Coder → Reviewer → Executor
6. LLM calls qua vLLM pool (circuit breaker + retry + batching) [P11/P19]
7. Tool calls: sandbox execute + tool-result validation [P10/P17]
8. Stream tokens ra SSE; ghi OTel span + Prometheus metric mỗi node [P14]
9. Lưu session + agent memory về Redis/Postgres; record token usage budget
10. Hậu kỳ: LLM-as-Judge chấm điểm (async), trace → dataset nếu có feedback [P16]
```

---

## 5. Multi-Agent Orchestration `[P15]`

Thay thế single-agent graph của v2.0 bằng **Agent Coordinator** điều phối nhiều agent chuyên biệt. Mỗi agent là một LangGraph subgraph độc lập, giao tiếp qua **message bus**.

### 5.1 Các agent chuyên biệt (`server/agents/`)

| Agent | Trách nhiệm |
|-------|-------------|
| `PlannerAgent` | Phân rã task thành các bước, xác định context cần thiết. |
| `ResearcherAgent` | Thu thập context: hybrid RAG, search symbol, đọc file, call graph. |
| `CoderAgent` | Sinh/sửa code (native tool-call, multi-file atomic edits). |
| `ReviewerAgent` | Kiểm tra chất lượng/bảo mật, chấm điểm, trả feedback. |
| `ExecutorAgent` | Thực thi tool sandboxed (test/lint/build), validate kết quả. |

Tất cả kế thừa `BaseAgent` (`agents/base.py`) với capability discovery.

### 5.2 Coordinator (`server/agent/coordinator.py`)

```
execute_workflow(task):
    plan      = PlannerAgent.plan(task)
    context   = ResearcherAgent.gather(plan.context_needs)   # có thể song song
    solution  = CoderAgent.generate(plan, context)
    review    = ReviewerAgent.review(solution)
    if not review.approved:  # loop có giới hạn
        return execute_workflow(review.feedback)
    return solution
```

- **Workflow DSL** định nghĩa pipeline, agent selection logic, chạy song song, aggregate kết quả, xử lý failure.

### 5.3 Message Bus (`server/agent/message_bus.py`)

- **Redis Pub/Sub** — `publish(channel, AgentMessage)` / `subscribe(channel)`.
- **Protocol** (`protocol.py`): `AgentMessage{sender, receiver, type(REQUEST/RESPONSE/BROADCAST/ERROR), payload, correlation_id, timestamp}`.
- Dead-letter queue cho message lỗi.

### 5.4 Cross-session Memory (`server/agent/memory_store.py`)

- Lưu memory bền vững (Postgres) + embedding index để **semantic recall**.
- `remember(key, value, ttl)` / `recall(query, top_k)`.
- Scoping theo user/project/tenant; có expiration.

---

## 6. LangGraph core (giữ + nâng cấp)

Graph trong mỗi agent vẫn theo mô hình v2.0 (`classify_intent → route_context → planner → rag_search → generate → verify → critic → post_process`), nhưng:

- **Prompt versioning** `[P16]` — system prompt không hardcode trong `generate.py` mà load từ `config/prompts/*.yaml` (version, variants, hot-reload, rollback).
- **Workflow customization** `[P20]` — pipeline cấu hình được thay vì cố định.
- **A/B variant** `[P16]` — node `generate` chọn prompt variant theo experiment assignment.

State (`AgentState`) mở rộng thêm: `tenant_id`, `experiment_variant`, `budget_remaining`, `trace_id`, `agent_role`.

---

## 7. Native tool-call + Tool Enhancements `[P17]`

Giữ cơ chế native tool-call của v2.0, bổ sung:

| Năng lực | Module | Mô tả |
|----------|--------|-------|
| Multi-file atomic edit | `mcp_server/tools_multifile.py` | Backup → apply nhiều file → verify → rollback nếu lỗi; conflict detection; dry-run. |
| Call graph | `mcp_server/call_graph.py` | Trích quan hệ gọi hàm theo ngôn ngữ; `callers()/callees()/impact_analysis()`; lưu NetworkX/Neo4j; incremental update. |
| LSP integration | `mcp_server/lsp_client.py` | go-to-definition, find references, hover, diagnostics. |
| Tool-result validation | `server/agent/tool_validator.py` | Validate kết quả tool trước khi dùng; phân loại lỗi; gợi ý retry. |
| Tool analytics | `server/metrics/tools.py` | success/failure rate, latency, độ chính xác chọn tool. |

Tool execution chạy trong **sandbox** `[P10]` (whitelist + resource limit + audit).

---

## 8. RAG subsystem — Hybrid `[P13]`

| Thành phần | Module | Mô tả |
|-----------|--------|-------|
| Hybrid search | `rag/hybrid_search.py` | Semantic (Qdrant) **+** BM25 (Elasticsearch/Meilisearch) chạy song song, hợp nhất bằng **Reciprocal Rank Fusion**; exact-match boost; code-aware tokenization. |
| Delta indexing | `rag/delta_indexer.py` | Index incremental theo git diff; xóa embedding cũ + index file thay đổi; track commit hash/repo; webhook on push. |
| File watcher | `rag/file_watcher.py` | Watchdog + debounce + ignore patterns → re-index real-time. |
| Cached AST | `mcp_server/ast_cache.py` | LRU cache AST có validate mtime; pre-warm; symbol index từ AST cache. |
| Smart chunking | `rag/smart_chunking.py` | Chunk theo function/class, giữ imports, gắn docstring, cross-reference. |
| Parent-child + HyDE + reranker | (từ v2.0) | Enrich context, mở rộng query, re-rank. |

**Qdrant chạy cluster 3 replica** `[P12]` (replication + shard + backup) thay cho single instance — loại bỏ SPOF.

---

## 9. State Persistence & Reliability `[P11]`

| Cơ chế | Module | Mô tả |
|--------|--------|-------|
| Redis session store | `server/session_redis.py` | `save_state/load_state` theo `session_id`, TTL, cleanup cron, fallback in-memory khi Redis lỗi. |
| Redis caches | `server/cache_redis.py` | EmbeddingCache / LLMResponseCache / RAGResultCache; key namespacing; invalidation; cache warming; degrade về in-memory. |
| Circuit breaker | `server/circuit_breaker.py` | states closed/open/half-open; áp dụng cho vLLM, Qdrant, external API; metric trạng thái; manual override. |
| Graceful degradation | `server/agent/fallback.py` | Fallback response khi LLM down; serve cached; partial mode; auto-recovery. |
| Deep health check | `server/routers/health.py` | `/health/deep` kiểm tra vLLM/Qdrant/Redis/Postgres/disk/mem; liveness vs readiness. |
| Retry strategy | `server/retry.py` | Exponential backoff + jitter; retry-able exception; áp dụng mọi external call. |

➡️ **Cache hit rate mục tiêu >60%** (v2.0 ~30%).

---

## 10. Security Hardening `[P10]`

| Lớp | Module | Mô tả |
|-----|--------|-------|
| Command sandbox | `mcp_server/sandbox.py` | Whitelist command (test/lint/build/git-read/...), chặn dangerous pattern, timeout/memory limit, network isolation, path-traversal guard, audit. (Tùy chọn Firejail/nsjail + cgroups trên Linux.) |
| Prompt injection guard | `server/agent/input_guard.py` | Pattern (role hijack, instruction override, delimiter, jailbreak, exfiltration) + unicode/homoglyph + perplexity anomaly; block vs warn mode; log input khả nghi. |
| Secret scanning | `server/utils/secret_scanner.py` | Regex + entropy; scan output LLM & file đọc; redaction `[REDACTED]`; alert. |
| Audit logging | `server/audit.py` + Postgres | Log mọi tool exec, auth event, security violation, admin action; retention 90 ngày; export SIEM (optional). |
| Input validation | `server/validation.py` | Pydantic mọi input; giới hạn message/file size; path traversal; schema cho tool args. |
| RBAC `[optional]` | `server/auth_rbac.py` | Permission/Role, scoped JWT, per-tool/per-repo check, token revocation. *(Đánh dấu optional trong plan.)* |

---

## 11. Observability `[P14]`

| Thành phần | Module | Mô tả |
|-----------|--------|-------|
| Distributed tracing | `server/tracing.py` | OpenTelemetry SDK, auto-instrument FastAPI/httpx, custom span cho graph node, LLM call (kèm token count), RAG search, tool exec. |
| Jaeger | docker-compose/Helm | OTLP exporter, sampling strategy, retention, Jaeger UI. |
| Correlation IDs | `server/middleware/correlation.py` | `X-Correlation-ID` middleware, propagate vào log + downstream + error response. |
| Error tracking | `server/error_tracking.py` | Sentry: environment tag, user context, PII scrubbing, release tracking. |
| SLO dashboards | `deploy/grafana/dashboards/slo.json` | Availability 99.9%, latency P99 <10s, error rate <1%, error budget + alerting. |
| Structured logging | `server/logging_config.py` | JSON log, per-module level, request context, aggregation Loki/ELK, sampling. |

---

## 12. AI Engineering Loop `[P16]`

| Năng lực | Module | Mô tả |
|----------|--------|-------|
| LLM-as-Judge | `eval/llm_judge.py` | Chấm điểm response đa chiều (correctness/completeness/quality/clarity); batch eval; correlate với người. |
| A/B testing | `server/experiment.py` | Feature flag + assignment xác định theo hash(user); track metric per variant; significance; dashboard. |
| Trace → Dataset | `eval/trace_collector.py` | Thu thập example tích cực/tiêu cực (thumbs up/down), PII removal, export JSONL, dataset versioning → phục vụ fine-tune. |
| Prompt versioning | `config/prompts/*.yaml` | YAML version + variants + hot-reload + diff + rollback. |
| Online quality metrics | `server/metrics/quality.py` | satisfaction rate, task completion, retry rate, code acceptance, response length. |
| Model comparison | `eval/model_comparison.py` | Multi-model eval runner, cost/quality trade-off, latency, capability matrix, recommendation. |
| Offline benchmark | `eval/benchmark.py` (từ v2.0) | Regression detection vs baseline. |
| Feedback analyzer | `server/feedback_analyzer.py` (từ v2.0) | Pattern detection → prompt improvement suggestion. |

---

## 13. Performance Optimization `[P19]`

| Cơ chế | Module | Mô tả |
|--------|--------|-------|
| Request batching | `server/batch.py` | Batch embedding & LLM inference; batch window + size limit. |
| Connection pooling | `server/connections.py` | Pool httpx (vLLM), asyncpg (Postgres), aioredis (Redis); health monitor → tránh socket exhaustion. |
| Speculative execution | `server/speculative.py` | Pre-warm prompt phổ biến, predictive context loading, background pre-compute. |
| Streaming optimization | `server/streaming/optimized.py` | Chunked transfer, buffer management, backpressure, client-disconnect detection. |

---

## 14. Enterprise Features `[P18]`

| Tính năng | Module | Mô tả |
|-----------|--------|-------|
| Token budget | `server/budget.py` + Postgres | Budget per user/project/org; pre-flight check; record usage; alert; cost allocation dashboard. |
| Multi-tenant isolation | `server/tenant.py` | Tenant context middleware; session/RAG-index/rate-limit scoped theo tenant; data isolation verification. |
| SSO (OIDC) | `server/auth_oidc.py` | OIDC discovery, JWT validation, user provisioning, group→role mapping. |
| Backup & restore | `deploy/backup/` | Postgres backup + Qdrant snapshot CronJob → S3/GCS; restore procedure; verification. |
| GitOps | `deploy/argocd/` | ArgoCD Application, env overlays, sync policy, rollback, promotion workflow. |
| CI/CD | `.gitlab-ci.yml` / `.github/workflows/` | build & test → image build → Trivy scan → staging → prod → smoke test. |

---

## 15. Developer Experience `[P20]`

| Mục | Đường dẫn | Mô tả |
|-----|-----------|-------|
| Developer docs | `docs/` | architecture (file này + hiện tại), API doc, setup guide, contributing, troubleshooting. |
| Local dev mode | `server/dev_mode.py` | Mock vLLM, in-memory Qdrant, SQLite thay Postgres, hot-reload, debug endpoints. |
| Integration tests | `tests/integration/` | E2E graph, API integration, external mocks, testcontainers. |
| CLI tool | `cli/main.py` | lệnh index / query / health / config validation. |

---

## 16. Cấu trúc thư mục đích (rút gọn)

```
ai-agent/
├── main.py · cli/main.py                         # entry + CLI [P20]
├── Dockerfile · docker-compose.yml
│
├── server/
│   ├── app.py                                    # middleware: OIDC, tenant, OTel, audit, budget
│   ├── auth.py · auth_oidc.py · auth_rbac.py     # [P18][P10]
│   ├── session_redis.py · cache_redis.py         # [P11]
│   ├── circuit_breaker.py · retry.py             # [P11]
│   ├── validation.py · audit.py                  # [P10]
│   ├── budget.py · tenant.py · experiment.py     # [P18][P16]
│   ├── batch.py · connections.py · speculative.py# [P19]
│   ├── tracing.py · error_tracking.py            # [P14]
│   ├── middleware/correlation.py                 # [P14]
│   ├── routers/  chat · index · review · metrics · feedback · health · experiments · admin
│   ├── agents/                                   # multi-agent [P15]
│   │   ├── base.py · planner.py · researcher.py · coder.py · reviewer.py · executor.py
│   ├── agent/
│   │   ├── coordinator.py · message_bus.py · protocol.py · memory_store.py  # [P15]
│   │   ├── graph.py · state.py · classify_intent.py · route_context.py
│   │   ├── generate.py · verify_result.py · critic.py · post_process.py
│   │   ├── tool_validator.py · fallback.py       # [P17][P11]
│   │   ├── input_guard.py                        # [P10]
│   │   └── prompts/  (+ config/prompts/*.yaml versioned [P16])
│   ├── rag/
│   │   ├── hybrid_search.py · delta_indexer.py · file_watcher.py · smart_chunking.py  # [P13]
│   │   ├── embedder.py · qdrant_client.py · context_retrieval.py · hyde.py · reranker.py
│   ├── metrics/  counter · prometheus · models · quality · tools   # [P14][P16][P17]
│   ├── streaming/  sse.py · optimized.py
│   └── utils/  sanitize · secret_scanner · content · json_parser · async_io  # [P10]
│
├── mcp_server/
│   ├── server.py · tools.py · tools_indexer.py · tools_review.py · tools_refactor.py
│   ├── tools_multifile.py · call_graph.py · lsp_client.py · ast_cache.py   # [P17][P13]
│   ├── sandbox.py                                                          # [P10]
│   └── plugins/  java · go · python · typescript · csharp · hcl · fallback
│
├── eval/  benchmark.py · llm_judge.py · trace_collector.py · model_comparison.py  # [P16]
├── gitlab-review-runner/                         # CI review service (riêng)
├── deploy/  helm/ · argocd/ · backup/ · nginx · prometheus · grafana(slo)         # [P12][P14][P18]
├── config/  rules.yaml · prompts/*.yaml          # [P16]
├── init-db/  schema (metrics · audit · budget · memory · dataset)                 # [P10][P16][P18]
└── tests/  unit/ · integration/                  # [P20]
```

---

## 17. Cấu hình / Environment (bổ sung so với v2.0)

| Biến | Mục đích | Phase |
|------|----------|-------|
| `REDIS_URL` | session/cache/message-bus/Celery broker | P11/P15 |
| `BM25_URL` | Elasticsearch/Meilisearch cho hybrid search | P13 |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | Jaeger/OTLP | P14 |
| `SENTRY_DSN` | error tracking | P14 |
| `OIDC_ISSUER` / `OIDC_CLIENT_ID` | SSO | P18 |
| `TOKEN_BUDGET_DEFAULT` | budget mặc định | P18 |
| `PROMPT_CONFIG_DIR` | thư mục prompt versioned | P16 |
| `CIRCUIT_FAILURE_THRESHOLD` / `CIRCUIT_RECOVERY_TIMEOUT` | circuit breaker | P11 |
| `ENABLE_MULTI_AGENT` | bật coordinator multi-agent | P15 |
| (kế thừa) `QDRANT_URL`, `VLLM_BASE_URL`, `VLLM_MODEL`, `ENABLE_RAG`, `MAX_TOOL_TURNS`, ... | | v2.0 |

---

## 18. Lộ trình & tiến triển điểm

| Phase | Nội dung | Ưu tiên | Score |
|-------|----------|---------|-------|
| 10 | Security Hardening | P0 | 6→8.5 |
| 11 | State Persistence & Reliability | P0 | 5→8 |
| 13 | RAG Improvements (hybrid, delta, AST) | P1 | 8→9.5 |
| 14 | Observability (OTel/Jaeger/Sentry/SLO) | P1 | 8→9.5 |
| 15 | Multi-Agent Architecture | P2 | 7→9 |
| 16 | AI Engineering Maturity | P2 | 7→9 |
| 17 | Tool Enhancements | P2 | 6→8.5 |
| 18 | Enterprise Features | P2 | — |
| 19 | Performance Optimization | P2 | 7→9 |
| 20 | Developer Experience | P2 | 7→9 |
| 12 | Horizontal Scaling *(optional)* | P1 | 5→8.5 |

```
Current:  7.5/10
After P0: 8.0/10   (Security + Reliability)
After P1: 8.8/10   (RAG + Observability [+ Scaling])
After P2: 9.5/10   (Multi-agent + AI eng + Tools + Enterprise + Perf + DX)
```

**Ước tính:** ~16–22 tuần (4–5.5 tháng), ~200 task, giải quyết 47 điểm yếu.

---

## 19. Tóm tắt thay đổi cốt lõi v2.0 → v3.0

| Khía cạnh | v2.0 (hiện tại) | v3.0 (đích) |
|-----------|-----------------|-------------|
| Agent | Single LangGraph agent | Multi-agent coordinator + message bus `[P15]` |
| State | In-memory session/cache | Redis distributed + fallback `[P11]` |
| RAG | Semantic-only Qdrant đơn | Hybrid (semantic+BM25) + delta + AST cache + Qdrant cluster `[P13]` |
| Scale | Single instance | K8s + HPA + Celery workers `[P12]` |
| Observability | Prometheus + JSON log | + OTel/Jaeger + Sentry + SLO `[P14]` |
| Security | sanitize + (untracked) guard/sandbox | Đầy đủ: sandbox + guard + secret scan + audit + validation `[P10]` |
| Tools | read/search/edit cơ bản | + multi-file atomic + call graph + LSP + validator `[P17]` |
| AI eng | offline benchmark + feedback | + LLM-judge + A/B + dataset + prompt versioning `[P16]` |
| Enterprise | API key | + OIDC SSO + multi-tenant + budget + backup + GitOps `[P18]` |
| Reliability | retry vLLM cơ bản | + circuit breaker + graceful degradation + deep health `[P11]` |

---

*Tài liệu thiết kế đích, tổng hợp từ `docs/architecture.md` (trạng thái hiện tại) + `docs/improvement-plan.md` (Phase 10–20). Sinh ngày 2026-06-04.*
```
