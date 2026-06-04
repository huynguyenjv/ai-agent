# AI Agent Improvement Plan (Complete)

**Date:** 2026-05-30  
**Based on:** Architecture Review  
**Current Score:** 7.5/10  
**Target Score:** 9.5/10

---

## Overview - All Identified Weaknesses

### 1. AI-Agent Architecture (Score: 7/10)
- [ ] Single Agent Instance - không có multi-agent collaboration thực sự
- [ ] Missing Agent Memory Persistence - state chỉ trong conversation context
- [ ] Limited Task Decomposition - execution vẫn sequential
- [ ] No Agent Communication Protocol - thiếu message bus/event system

### 2. Coding Capability (Score: 7/10)
- [ ] Context Window Efficiency - không intelligent chunking cho large files
- [ ] No Incremental Code Analysis - mỗi request re-analyze, thiếu cached AST
- [ ] Limited Multi-file Operations - tools chỉ operate single file
- [ ] Dependency Graph Missing - không track call graph

### 3. RAG & Memory (Score: 8/10)
- [ ] In-Memory Only - tất cả caches là in-process
- [ ] No Embedding Update Strategy - thiếu incremental re-indexing
- [ ] Single Qdrant Instance - no replication, SPOF
- [ ] Missing Hybrid Search - chỉ semantic, không BM25

### 4. Tool Calling (Score: 6/10)
- [ ] No True Sandboxing - run_command không có resource limits
- [ ] Missing Tool Timeouts - commands có thể hang vô hạn
- [ ] No Tool Result Validation - trust LLM output hoàn toàn
- [ ] Limited IDE Integration - chỉ Continue compat, không LSP
- [ ] Command Injection Risk - LLM control command execution

### 5. Scalability (Score: 5/10)
- [ ] Single Instance Design - không horizontal scaling
- [ ] No Queue System - requests processed synchronously
- [ ] Session State In-Memory - cannot load-balance
- [ ] Missing Async Processing - long tasks block
- [ ] No Cost Optimization - không budget controls
- [ ] No Backpressure - no request throttling under load

### 6. Performance (Score: 7/10)
- [ ] No Request Batching - mỗi request = separate LLM call
- [ ] Sequential Planning - Planner → Generate blocking
- [ ] Missing Connection Pooling - potential socket exhaustion
- [ ] No Speculative Decoding - không pre-warm prompts
- [ ] Cache Hit Rate Low - ~30%, target >60%

### 7. Security (Score: 6/10)
- [ ] No Prompt Injection Defense - LLM input không sanitized
- [ ] Weak Sandboxing - run_command executes arbitrary commands
- [ ] No Secret Scanning - có thể expose .env content
- [ ] No Audit Logging - security events không tracked
- [ ] No Input Validation - thiếu schema validation

<!-- Don't implemented -->
<!-- ### 8. DevOps & Infrastructure (Score: 8/10)
- [ ] No Kubernetes Manifests - chỉ docker-compose
- [ ] Missing CI/CD Pipeline - no automated deployment
- [ ] No Blue/Green Deployment - downtime during updates
- [ ] Missing Distributed Tracing - no Jaeger/Zipkin
- [ ] No Secrets Management - env vars only
- [ ] No Backup Strategy - data loss risk
- [ ] No Disaster Recovery plan -->

### 9. Developer Experience (Score: 7/10)
- [ ] No Prompt Versioning - prompts hardcoded
- [ ] Limited Workflow Customization - fixed LangGraph structure
- [ ] Missing Developer Docs - no architecture docs
- [ ] No Local Development Mode - requires full stack
- [ ] No Integration Tests - only unit tests

### 10. AI Engineering (Score: 7/10)
- [ ] No Online Evaluation - production metrics không tied to quality
- [ ] Missing LLM-as-Judge - no automated response scoring
- [ ] No A/B Testing - single model path
- [ ] No Dataset Collection - cannot fine-tune from production
- [ ] No Model Comparison - cannot compare model performance
- [ ] No Prompt Optimization Loop - manual prompt tuning only

---

## Phase 10: Security Hardening (P0 - Critical)

**Timeline:** 1-2 tuần  
**Current:** 6/10 → **Target:** 8.5/10

### 10.1 Command Sandboxing
**File:** `mcp_server/sandbox.py`, `mcp_server/tools.py`

```python
# Whitelist approach
ALLOWED_COMMANDS = {
    "pytest", "python -m pytest",
    "npm test", "npm run test", "npx jest",
    "go test", "cargo test",
    "gradle test", "mvn test",
    "ruff check", "eslint", "gofmt",
    "git status", "git diff", "git log",
}

# Resource limits
SANDBOX_CONFIG = {
    "timeout_seconds": 300,
    "memory_limit_mb": 512,
    "cpu_limit": 1.0,
    "network": False,  # No network access
    "filesystem": "readonly",  # Except temp dir
}
```

**Tasks:**
- [ ] 10.1.1 Create command whitelist registry
- [ ] 10.1.2 Implement command parser & validator
- [ ] 10.1.3 Add Firejail/nsjail wrapper for Linux
- [ ] 10.1.4 Add resource cgroups limits
- [ ] 10.1.5 Implement timeout enforcement
- [ ] 10.1.6 Add network isolation option
- [ ] 10.1.7 Tests for sandbox escape attempts

### 10.2 Prompt Injection Detection
**File:** `server/agent/input_guard.py`

```python
class InputGuard:
    """Detect and block prompt injection attempts."""
    
    INJECTION_PATTERNS = [
        r"ignore.*previous.*instructions",
        r"disregard.*above",
        r"you are now",
        r"new instructions:",
        r"system prompt:",
        r"</?(system|assistant|user)>",
    ]
    
    def check(self, text: str) -> GuardResult:
        # Pattern matching
        # Perplexity-based detection
        # Character encoding attacks
        pass
```

**Tasks:**
- [ ] 10.2.1 Regex pattern detector
- [ ] 10.2.2 Unicode/encoding attack detection
- [ ] 10.2.3 Jailbreak phrase database
- [ ] 10.2.4 Perplexity-based anomaly detection
- [ ] 10.2.5 Input sanitization layer
- [ ] 10.2.6 Logging suspicious inputs
- [ ] 10.2.7 Block vs warn modes

### 10.3 Secret Scanning
**File:** `server/utils/secret_scanner.py`

**Tasks:**
- [ ] 10.3.1 Regex patterns for API keys, tokens, passwords
- [ ] 10.3.2 Entropy-based secret detection
- [ ] 10.3.3 Scan LLM outputs before returning
- [ ] 10.3.4 Scan file reads for .env, credentials
- [ ] 10.3.5 Redaction with [REDACTED] placeholder
- [ ] 10.3.6 Alert on detected secrets

<!-- ### 10.4 RBAC (Role-Based Access Control)
**File:** `server/auth_rbac.py`, `server/models/permission.py`

```python
class Permission(Enum):
    READ_FILE = "read_file"
    WRITE_FILE = "write_file"
    RUN_COMMAND = "run_command"
    ADMIN = "admin"

class Role:
    VIEWER = [Permission.READ_FILE]
    DEVELOPER = [Permission.READ_FILE, Permission.WRITE_FILE, Permission.RUN_COMMAND]
    ADMIN = [Permission.ADMIN]
```

**Tasks:**
- [ ] 10.4.1 Permission enum & Role definitions
- [ ] 10.4.2 Scoped JWT tokens with claims
- [ ] 10.4.3 Per-tool permission checks
- [ ] 10.4.4 Per-repository access control
- [ ] 10.4.5 Token revocation list
- [ ] 10.4.6 Admin endpoints for user management -->

### 10.5 Audit Logging
**File:** `server/audit.py`, `server/models/audit_log.py`

**Tasks:**
- [ ] 10.5.1 AuditLog model (Postgres)
- [ ] 10.5.2 Log all tool executions
- [ ] 10.5.3 Log authentication events
- [ ] 10.5.4 Log security violations
- [ ] 10.5.5 Log admin actions
- [ ] 10.5.6 Retention policy (90 days)
- [ ] 10.5.7 Export to SIEM (optional)

### 10.6 Input Validation
**File:** `server/validation.py`

**Tasks:**
- [ ] 10.6.1 Pydantic models for all inputs
- [ ] 10.6.2 Max message length limits
- [ ] 10.6.3 Max file size limits
- [ ] 10.6.4 Path traversal validation
- [ ] 10.6.5 Schema validation for tool arguments

---

## Phase 11: State Persistence & Reliability (P0)

**Timeline:** 1-2 tuần  
**Current:** 5/10 → **Target:** 8/10

### 11.1 Redis Session Store
**File:** `server/session_redis.py`

```python
class RedisSessionStore:
    def __init__(self, redis_url: str):
        self.redis = aioredis.from_url(redis_url)
    
    async def save_state(self, session_id: str, state: AgentState):
        await self.redis.setex(
            f"session:{session_id}",
            ttl=3600,
            value=json.dumps(state)
        )
    
    async def load_state(self, session_id: str) -> AgentState | None:
        data = await self.redis.get(f"session:{session_id}")
        return json.loads(data) if data else None
```

**Tasks:**
- [ ] 11.1.1 Redis connection pool
- [ ] 11.1.2 Session serialization/deserialization
- [ ] 11.1.3 Session TTL management
- [ ] 11.1.4 Session cleanup cron
- [ ] 11.1.5 Fallback to in-memory on Redis failure
- [ ] 11.1.6 Session migration (in-memory → Redis)

### 11.2 Redis-Backed Caches
**File:** `server/cache_redis.py`

**Tasks:**
- [ ] 11.2.1 RedisEmbeddingCache
- [ ] 11.2.2 RedisLLMResponseCache
- [ ] 11.2.3 RedisRAGResultCache
- [ ] 11.2.4 Cache key namespacing
- [ ] 11.2.5 Cache invalidation patterns
- [ ] 11.2.6 Cache warming on startup
- [ ] 11.2.7 Graceful degradation to in-memory

### 11.3 Circuit Breaker
**File:** `server/circuit_breaker.py`

```python
class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60,
        half_open_requests: int = 3,
    ):
        self.state = "closed"  # closed, open, half-open
        self.failures = 0
        
    async def call(self, func, *args, **kwargs):
        if self.state == "open":
            if time.time() > self.open_until:
                self.state = "half-open"
            else:
                raise CircuitOpenError()
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
```

**Tasks:**
- [ ] 11.3.1 CircuitBreaker class with states
- [ ] 11.3.2 Apply to vLLM client
- [ ] 11.3.3 Apply to Qdrant client
- [ ] 11.3.4 Apply to external APIs
- [ ] 11.3.5 Metrics for circuit state
- [ ] 11.3.6 Manual override endpoints

### 11.4 Graceful Degradation
**File:** `server/agent/fallback.py`

**Tasks:**
- [ ] 11.4.1 Fallback responses when LLM unavailable
- [ ] 11.4.2 Cached response serving
- [ ] 11.4.3 Partial functionality mode
- [ ] 11.4.4 User notification of degraded state
- [ ] 11.4.5 Auto-recovery detection

### 11.5 Health Checks (Deep)
**File:** `server/routers/health.py`

```python
@router.get("/health/deep")
async def deep_health_check():
    checks = {
        "vllm": await check_vllm(),
        "qdrant": await check_qdrant(),
        "redis": await check_redis(),
        "postgres": await check_postgres(),
    }
    
    all_healthy = all(c["status"] == "ok" for c in checks.values())
    return {"status": "ok" if all_healthy else "degraded", "checks": checks}
```

**Tasks:**
- [ ] 11.5.1 vLLM connectivity check
- [ ] 11.5.2 Qdrant read/write check
- [ ] 11.5.3 Redis ping check
- [ ] 11.5.4 Postgres query check
- [ ] 11.5.5 Disk space check
- [ ] 11.5.6 Memory usage check
- [ ] 11.5.7 Liveness vs Readiness probes

### 11.6 Retry Strategy
**File:** `server/retry.py`

**Tasks:**
- [ ] 11.6.1 Exponential backoff helper
- [ ] 11.6.2 Jitter for thundering herd prevention
- [ ] 11.6.3 Max retry configuration
- [ ] 11.6.4 Retry-able exception detection
- [ ] 11.6.5 Apply to all external calls

---

<!-- Don't implemented -->

<!-- ## Phase 12: Horizontal Scaling (P1)

**Timeline:** 2-3 tuần  
**Current:** 5/10 → **Target:** 8.5/10

### 12.1 Kubernetes Helm Chart
**Directory:** `deploy/helm/ai-agent/`

```
deploy/helm/ai-agent/
├── Chart.yaml
├── values.yaml
├── templates/
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── hpa.yaml
│   ├── pdb.yaml
│   ├── configmap.yaml
│   ├── secret.yaml
│   ├── ingress.yaml
│   └── serviceaccount.yaml
```

**Tasks:**
- [ ] 12.1.1 Deployment template
- [ ] 12.1.2 Service (ClusterIP + LoadBalancer)
- [ ] 12.1.3 ConfigMap for non-secret config
- [ ] 12.1.4 Secret template với external-secrets
- [ ] 12.1.5 Ingress with TLS
- [ ] 12.1.6 ServiceAccount + RBAC
- [ ] 12.1.7 values.yaml với environments

### 12.2 HPA (Horizontal Pod Autoscaler)
**File:** `deploy/helm/ai-agent/templates/hpa.yaml`

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ai-agent
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ai-agent
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Pods
    pods:
      metric:
        name: ai_agent_active_requests
      target:
        type: AverageValue
        averageValue: 10
```

**Tasks:**
- [ ] 12.2.1 CPU-based scaling
- [ ] 12.2.2 Memory-based scaling
- [ ] 12.2.3 Custom metrics (active requests)
- [ ] 12.2.4 Scale down stabilization
- [ ] 12.2.5 Load testing & tuning

### 12.3 Task Queue (Celery)
**Directory:** `server/task_queue/`

```python
# server/task_queue/worker.py
from celery import Celery

app = Celery('ai-agent', broker='redis://redis:6379/1')

@app.task(bind=True, max_retries=3)
def process_long_task(self, task_id: str, payload: dict):
    # Long-running operations
    # - Large file indexing
    # - Full repository analysis
    # - Batch code generation
    pass
```

**Tasks:**
- [ ] 12.3.1 Celery app setup
- [ ] 12.3.2 Task definitions
- [ ] 12.3.3 Task result backend (Redis)
- [ ] 12.3.4 Task status API endpoints
- [ ] 12.3.5 Worker deployment template
- [ ] 12.3.6 Flower monitoring dashboard
- [ ] 12.3.7 Priority queues

### 12.4 Qdrant Cluster Mode
**File:** `deploy/qdrant-cluster.yml`

**Tasks:**
- [ ] 12.4.1 Qdrant StatefulSet (3 replicas)
- [ ] 12.4.2 Persistent volume claims
- [ ] 12.4.3 Collection replication factor
- [ ] 12.4.4 Shard configuration
- [ ] 12.4.5 Backup CronJob

### 12.5 Load Balancer Configuration
**File:** `deploy/nginx.conf`, `deploy/helm/templates/ingress.yaml`

**Tasks:**
- [ ] 12.5.1 Sticky sessions (IP hash)
- [ ] 12.5.2 WebSocket support for SSE
- [ ] 12.5.3 Rate limiting at LB level
- [ ] 12.5.4 SSL termination
- [ ] 12.5.5 Health check endpoints

### 12.6 PodDisruptionBudget
**File:** `deploy/helm/ai-agent/templates/pdb.yaml`

**Tasks:**
- [ ] 12.6.1 minAvailable: 1 for HA
- [ ] 12.6.2 Rolling update strategy
- [ ] 12.6.3 Pre-stop hooks for graceful shutdown

### 12.7 Backpressure & Throttling
**File:** `server/throttle.py`

**Tasks:**
- [ ] 12.7.1 Request queue with max depth
- [ ] 12.7.2 429 responses when overloaded
- [ ] 12.7.3 Priority queue for premium users
- [ ] 12.7.4 Adaptive rate limiting

--- -->

## Phase 13: RAG Improvements (P1)

**Timeline:** 1-2 tuần  
**Current:** 8/10 → **Target:** 9.5/10

### 13.1 Delta Indexing
**File:** `server/rag/delta_indexer.py`

```python
class DeltaIndexer:
    """Incremental indexing based on git changes."""
    
    async def index_changes(self, repo_path: str, since_commit: str):
        # Get changed files
        changed = await self._get_git_diff(repo_path, since_commit)
        
        # Delete old embeddings for changed files
        await self._delete_embeddings(changed.modified + changed.deleted)
        
        # Re-index only changed/added files
        await self._index_files(changed.added + changed.modified)
```

**Tasks:**
- [ ] 13.1.1 Git diff parsing
- [ ] 13.1.2 Selective embedding deletion
- [ ] 13.1.3 Incremental embedding creation
- [ ] 13.1.4 Commit hash tracking per repo
- [ ] 13.1.5 Webhook trigger on push
- [ ] 13.1.6 Background indexing job

### 13.2 Hybrid Search (BM25 + Semantic)
**File:** `server/rag/hybrid_search.py`

```python
async def hybrid_search(query: str, top_k: int = 10) -> list[dict]:
    # Parallel search
    semantic_results, bm25_results = await asyncio.gather(
        semantic_search(query, top_k * 2),
        bm25_search(query, top_k * 2),
    )
    
    # Reciprocal Rank Fusion
    return reciprocal_rank_fusion(semantic_results, bm25_results, k=60)
```

**Tasks:**
- [ ] 13.2.1 BM25 index (Elasticsearch/Meilisearch)
- [ ] 13.2.2 Parallel search execution
- [ ] 13.2.3 Reciprocal Rank Fusion
- [ ] 13.2.4 Score normalization
- [ ] 13.2.5 Exact match boost
- [ ] 13.2.6 Code-specific tokenization

### 13.3 File Watcher
**File:** `server/rag/file_watcher.py`

**Tasks:**
- [ ] 13.3.1 Watchdog integration
- [ ] 13.3.2 Debouncing file changes
- [ ] 13.3.3 Ignore patterns (.git, node_modules)
- [ ] 13.3.4 Queue changed files for re-indexing
- [ ] 13.3.5 Real-time index updates

### 13.4 Cached AST
**File:** `mcp_server/ast_cache.py`

```python
class ASTCache:
    """LRU cache for parsed ASTs."""
    
    def __init__(self, max_size: int = 1000):
        self._cache: dict[str, tuple[float, Any]] = {}
        
    def get_ast(self, file_path: str) -> Any | None:
        entry = self._cache.get(file_path)
        if entry:
            mtime, ast = entry
            if os.path.getmtime(file_path) == mtime:
                return ast
        return None
```

**Tasks:**
- [ ] 13.4.1 AST cache with mtime validation
- [ ] 13.4.2 Pre-warm on startup
- [ ] 13.4.3 Invalidation on file change
- [ ] 13.4.4 LRU eviction
- [ ] 13.4.5 Symbol index from cached AST

### 13.5 Smart Chunking
**File:** `server/rag/smart_chunking.py`

**Tasks:**
- [ ] 13.5.1 Function-level chunking
- [ ] 13.5.2 Class-level chunking
- [ ] 13.5.3 Preserve imports in each chunk
- [ ] 13.5.4 Docstring association
- [ ] 13.5.5 Cross-reference linking

---

## Phase 14: Observability (P1)

**Timeline:** 1 tuần  
**Current:** 8/10 → **Target:** 9.5/10

### 14.1 OpenTelemetry Tracing
**File:** `server/tracing.py`

```python
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

def setup_tracing(app: FastAPI):
    tracer_provider = TracerProvider(
        resource=Resource.create({"service.name": "ai-agent"})
    )
    tracer_provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter())
    )
    trace.set_tracer_provider(tracer_provider)
    FastAPIInstrumentor.instrument_app(app)
```

**Tasks:**
- [ ] 14.1.1 OpenTelemetry SDK setup
- [ ] 14.1.2 Auto-instrumentation (FastAPI, httpx)
- [ ] 14.1.3 Custom spans for graph nodes
- [ ] 14.1.4 LLM call tracing with token counts
- [ ] 14.1.5 RAG search tracing
- [ ] 14.1.6 Tool execution tracing

### 14.2 Jaeger Integration
**File:** `docker-compose.yml`

**Tasks:**
- [ ] 14.2.1 Jaeger container in compose
- [ ] 14.2.2 OTLP exporter configuration
- [ ] 14.2.3 Sampling strategy
- [ ] 14.2.4 Retention configuration
- [ ] 14.2.5 Jaeger UI access

### 14.3 Correlation IDs
**File:** `server/middleware/correlation.py`

```python
@app.middleware("http")
async def add_correlation_id(request: Request, call_next):
    correlation_id = request.headers.get("X-Correlation-ID") or str(uuid.uuid4())
    
    # Set in context for logging
    correlation_id_ctx.set(correlation_id)
    
    response = await call_next(request)
    response.headers["X-Correlation-ID"] = correlation_id
    return response
```

**Tasks:**
- [ ] 14.3.1 Correlation ID middleware
- [ ] 14.3.2 Propagate to all logs
- [ ] 14.3.3 Propagate to downstream calls
- [ ] 14.3.4 Include in error responses

### 14.4 Error Tracking (Sentry)
**File:** `server/error_tracking.py`

**Tasks:**
- [ ] 14.4.1 Sentry SDK integration
- [ ] 14.4.2 Environment tagging
- [ ] 14.4.3 User context attachment
- [ ] 14.4.4 PII scrubbing
- [ ] 14.4.5 Release tracking

### 14.5 SLO Dashboards
**File:** `deploy/grafana/dashboards/slo.json`

**Tasks:**
- [ ] 14.5.1 Availability SLO (99.9%)
- [ ] 14.5.2 Latency P99 SLO (<10s)
- [ ] 14.5.3 Error rate SLO (<1%)
- [ ] 14.5.4 Error budget tracking
- [ ] 14.5.5 Alerting rules

### 14.6 Structured Logging
**File:** `server/logging_config.py`

**Tasks:**
- [ ] 14.6.1 JSON log format
- [ ] 14.6.2 Log levels per module
- [ ] 14.6.3 Request context in logs
- [ ] 14.6.4 Log aggregation (Loki/ELK)
- [ ] 14.6.5 Log sampling for high volume

---

## Phase 15: Multi-Agent Architecture (P2)

**Timeline:** 2-3 tuần  
**Current:** 7/10 → **Target:** 9/10

### 15.1 Agent Message Bus
**File:** `server/agent/message_bus.py`

```python
class AgentMessageBus:
    """Redis Pub/Sub based message bus."""
    
    async def publish(self, channel: str, message: AgentMessage):
        await self.redis.publish(channel, message.json())
    
    async def subscribe(self, channel: str) -> AsyncIterator[AgentMessage]:
        pubsub = self.redis.pubsub()
        await pubsub.subscribe(channel)
        async for msg in pubsub.listen():
            yield AgentMessage.parse_raw(msg["data"])
```

**Tasks:**
- [ ] 15.1.1 Message bus interface
- [ ] 15.1.2 Redis Pub/Sub implementation
- [ ] 15.1.3 Message serialization (Protobuf?)
- [ ] 15.1.4 Message routing
- [ ] 15.1.5 Dead letter queue

### 15.2 Specialized Agents
**Directory:** `server/agents/`

```
server/agents/
├── base.py           # BaseAgent class
├── researcher.py     # Code search, RAG queries
├── coder.py          # Code generation, edits
├── reviewer.py       # Code review, quality checks
├── planner.py        # Task decomposition
└── executor.py       # Tool execution
```

**Tasks:**
- [ ] 15.2.1 BaseAgent abstract class
- [ ] 15.2.2 ResearcherAgent - context gathering
- [ ] 15.2.3 CoderAgent - code generation
- [ ] 15.2.4 ReviewerAgent - quality checks
- [ ] 15.2.5 ExecutorAgent - tool execution
- [ ] 15.2.6 Agent capability discovery

### 15.3 Agent Coordinator
**File:** `server/agent/coordinator.py`

```python
class AgentCoordinator:
    """Orchestrates multi-agent workflows."""
    
    async def execute_workflow(self, task: Task) -> Result:
        # 1. Planner decomposes task
        plan = await self.planner.plan(task)
        
        # 2. Researcher gathers context
        context = await self.researcher.gather(plan.context_needs)
        
        # 3. Coder generates solution
        solution = await self.coder.generate(plan, context)
        
        # 4. Reviewer validates
        review = await self.reviewer.review(solution)
        
        # 5. Iterate if needed
        if not review.approved:
            return await self.execute_workflow(review.feedback)
        
        return solution
```

**Tasks:**
- [ ] 15.3.1 Workflow definition DSL
- [ ] 15.3.2 Agent selection logic
- [ ] 15.3.3 Parallel agent execution
- [ ] 15.3.4 Result aggregation
- [ ] 15.3.5 Failure handling

### 15.4 Cross-Session Memory
**File:** `server/agent/memory_store.py`

```python
class AgentMemoryStore:
    """Persistent memory across sessions."""
    
    async def remember(self, key: str, value: Any, ttl: int = None):
        # Store in Postgres/Redis
        pass
    
    async def recall(self, query: str, top_k: int = 5) -> list[Memory]:
        # Semantic search over memories
        pass
```

**Tasks:**
- [ ] 15.4.1 Memory schema (Postgres)
- [ ] 15.4.2 Memory embedding index
- [ ] 15.4.3 Memory retrieval
- [ ] 15.4.4 Memory expiration
- [ ] 15.4.5 User/project scoping

### 15.5 Agent Communication Protocol
**File:** `server/agent/protocol.py`

```python
class AgentMessage(BaseModel):
    sender: str
    receiver: str
    type: MessageType  # REQUEST, RESPONSE, BROADCAST, ERROR
    payload: dict
    correlation_id: str
    timestamp: datetime
```

**Tasks:**
- [ ] 15.5.1 Message types definition
- [ ] 15.5.2 Request-response pattern
- [ ] 15.5.3 Broadcast pattern
- [ ] 15.5.4 Error handling
- [ ] 15.5.5 Message validation

---

## Phase 16: AI Engineering Maturity (P2)

**Timeline:** 2 tuần  
**Current:** 7/10 → **Target:** 9/10

### 16.1 LLM-as-Judge
**File:** `eval/llm_judge.py`

```python
JUDGE_PROMPT = """
Rate the following AI response on a scale of 1-10 for:
1. Correctness: Does it solve the task?
2. Completeness: Is anything missing?
3. Code Quality: Is the code clean and idiomatic?
4. Clarity: Is the explanation clear?

Task: {task}
Response: {response}

Output JSON: {"correctness": X, "completeness": X, "quality": X, "clarity": X, "overall": X}
"""

async def judge_response(task: str, response: str) -> JudgeResult:
    # Use a capable model (GPT-4, Claude) as judge
    result = await judge_llm.complete(JUDGE_PROMPT.format(...))
    return JudgeResult.parse(result)
```

**Tasks:**
- [ ] 16.1.1 Judge prompt engineering
- [ ] 16.1.2 Multi-dimension scoring
- [ ] 16.1.3 Batch evaluation
- [ ] 16.1.4 Human correlation validation
- [ ] 16.1.5 Judge model selection

### 16.2 A/B Testing Framework
**File:** `server/experiment.py`

```python
class ExperimentManager:
    """Feature flag and A/B testing."""
    
    def get_variant(self, experiment: str, user_id: str) -> str:
        # Deterministic assignment
        hash_val = hash(f"{experiment}:{user_id}") % 100
        
        exp = self.experiments[experiment]
        cumulative = 0
        for variant, weight in exp.variants.items():
            cumulative += weight
            if hash_val < cumulative:
                return variant
        return exp.default
```

**Tasks:**
- [ ] 16.2.1 Experiment definition schema
- [ ] 16.2.2 User/session assignment
- [ ] 16.2.3 Variant metrics tracking
- [ ] 16.2.4 Statistical significance calculation
- [ ] 16.2.5 Experiment dashboard

### 16.3 Trace → Dataset Pipeline
**File:** `eval/trace_collector.py`

```python
class TraceCollector:
    """Collect production traces for fine-tuning."""
    
    async def collect_positive(self, trace: Trace):
        # User gave positive feedback
        await self.dataset.add({
            "messages": trace.messages,
            "response": trace.response,
            "feedback": "positive",
            "metadata": trace.metadata,
        })
```

**Tasks:**
- [ ] 16.3.1 Trace schema definition
- [ ] 16.3.2 Positive example collection (thumbs up)
- [ ] 16.3.3 Negative example collection
- [ ] 16.3.4 PII removal
- [ ] 16.3.5 Export to training format (JSONL)
- [ ] 16.3.6 Dataset versioning

### 16.4 Prompt Versioning
**Directory:** `config/prompts/`

```yaml
# config/prompts/code_gen.yaml
version: "2.1.0"
metadata:
  author: team
  updated: 2026-05-30
  
prompts:
  system: |
    You are an expert coding assistant...
    
  user_template: |
    Task: {task}
    Context: {context}
    
variants:
  concise:
    system: |
      You are a concise coding assistant...
```

**Tasks:**
- [ ] 16.4.1 YAML prompt format
- [ ] 16.4.2 Prompt loader with hot-reload
- [ ] 16.4.3 Version tracking
- [ ] 16.4.4 Prompt variants for A/B testing
- [ ] 16.4.5 Prompt diff visualization
- [ ] 16.4.6 Rollback capability

### 16.5 Online Quality Metrics
**File:** `server/metrics/quality.py`

**Tasks:**
- [ ] 16.5.1 User satisfaction rate
- [ ] 16.5.2 Task completion rate
- [ ] 16.5.3 Retry rate per intent
- [ ] 16.5.4 Code acceptance rate
- [ ] 16.5.5 Response length analysis

### 16.6 Model Comparison Dashboard
**File:** `eval/model_comparison.py`

**Tasks:**
- [ ] 16.6.1 Multi-model evaluation runner
- [ ] 16.6.2 Cost/quality trade-off analysis
- [ ] 16.6.3 Latency comparison
- [ ] 16.6.4 Capability matrix
- [ ] 16.6.5 Recommendation engine

---

## Phase 17: Tool Enhancements (P2)

**Timeline:** 1-2 tuần  
**Current:** 6/10 → **Target:** 8.5/10

### 17.1 Multi-file Atomic Edits
**File:** `mcp_server/tools_multifile.py`

```python
async def apply_multi_file_edits(
    repo_path: str,
    edits: list[FileEdit],
    dry_run: bool = True,
) -> MultiEditResult:
    # Create backup
    backup = await create_backup(repo_path, edits)
    
    try:
        for edit in edits:
            await apply_edit(repo_path, edit)
        
        # Verify all files
        for edit in edits:
            if not await verify_edit(edit):
                raise EditVerificationError(edit)
        
        return MultiEditResult(success=True, files=len(edits))
        
    except Exception as e:
        # Rollback
        await restore_backup(backup)
        return MultiEditResult(success=False, error=str(e))
```

**Tasks:**
- [ ] 17.1.1 Backup creation
- [ ] 17.1.2 Atomic transaction
- [ ] 17.1.3 Rollback on failure
- [ ] 17.1.4 Verify edits
- [ ] 17.1.5 Conflict detection

### 17.2 Call Graph Database
**File:** `mcp_server/call_graph.py`

```python
class CallGraph:
    """Track function call relationships."""
    
    def build(self, repo_path: str):
        # Parse all files
        # Extract function definitions
        # Extract function calls
        # Build graph edges
        pass
    
    def callers(self, function: str) -> list[str]:
        """Who calls this function?"""
        pass
    
    def callees(self, function: str) -> list[str]:
        """What does this function call?"""
        pass
    
    def impact_analysis(self, function: str) -> ImpactReport:
        """What breaks if we change this?"""
        pass
```

**Tasks:**
- [ ] 17.2.1 Call extraction per language
- [ ] 17.2.2 Graph storage (NetworkX/Neo4j)
- [ ] 17.2.3 Incremental updates
- [ ] 17.2.4 Impact analysis
- [ ] 17.2.5 Visualization

### 17.3 LSP Integration
**File:** `mcp_server/lsp_client.py`

**Tasks:**
- [ ] 17.3.1 LSP client implementation
- [ ] 17.3.2 Go to definition
- [ ] 17.3.3 Find references
- [ ] 17.3.4 Hover information
- [ ] 17.3.5 Diagnostics

### 17.4 Tool Result Validation
**File:** `server/agent/tool_validator.py`

```python
class ToolValidator:
    """Validate tool results before using."""
    
    def validate_read_file(self, result: dict) -> ValidationResult:
        if "error" in result:
            return ValidationResult(valid=False, reason=result["error"])
        if not result.get("content"):
            return ValidationResult(valid=False, reason="Empty content")
        return ValidationResult(valid=True)
```

**Tasks:**
- [ ] 17.4.1 Per-tool validators
- [ ] 17.4.2 Schema validation
- [ ] 17.4.3 Error categorization
- [ ] 17.4.4 Retry suggestions
- [ ] 17.4.5 Logging invalid results

### 17.5 Tool Usage Analytics
**File:** `server/metrics/tools.py`

**Tasks:**
- [ ] 17.5.1 Tool success/failure rates
- [ ] 17.5.2 Tool latency distribution
- [ ] 17.5.3 Tool selection accuracy
- [ ] 17.5.4 Tool combination patterns

---

## Phase 18: Enterprise Features (P2)

**Timeline:** 2-3 tuần  
**Impact:** Enterprise readiness

### 18.1 Token Budget Management
**File:** `server/budget.py`

```python
class TokenBudget:
    """Per-user/project token budget."""
    
    async def check_budget(self, user_id: str, estimated_tokens: int) -> bool:
        usage = await self.get_usage(user_id)
        budget = await self.get_budget(user_id)
        return usage + estimated_tokens <= budget
    
    async def record_usage(self, user_id: str, tokens: int):
        await self.db.increment_usage(user_id, tokens)
```

**Tasks:**
- [ ] 18.1.1 Budget schema (per user, project, org)
- [ ] 18.1.2 Pre-flight budget check
- [ ] 18.1.3 Usage recording
- [ ] 18.1.4 Budget alerts
- [ ] 18.1.5 Usage dashboard
- [ ] 18.1.6 Cost allocation

### 18.2 Multi-tenant Isolation
**File:** `server/tenant.py`

**Tasks:**
- [ ] 18.2.1 Tenant context middleware
- [ ] 18.2.2 Tenant-scoped sessions
- [ ] 18.2.3 Tenant-scoped RAG index
- [ ] 18.2.4 Tenant-scoped rate limits
- [ ] 18.2.5 Data isolation verification

### 18.3 SSO Integration (OIDC)
**File:** `server/auth_oidc.py`

**Tasks:**
- [ ] 18.3.1 OIDC discovery
- [ ] 18.3.2 JWT validation
- [ ] 18.3.3 User provisioning
- [ ] 18.3.4 Group mapping to roles
- [ ] 18.3.5 Session management

### 18.4 Backup & Restore
**Directory:** `deploy/backup/`

**Tasks:**
- [ ] 18.4.1 Postgres backup CronJob
- [ ] 18.4.2 Qdrant snapshot CronJob
- [ ] 18.4.3 S3/GCS backup storage
- [ ] 18.4.4 Restore procedures
- [ ] 18.4.5 Backup verification

### 18.5 GitOps Deployment
**Directory:** `deploy/argocd/`

**Tasks:**
- [ ] 18.5.1 ArgoCD Application manifest
- [ ] 18.5.2 Environment overlays
- [ ] 18.5.3 Sync policies
- [ ] 18.5.4 Rollback automation
- [ ] 18.5.5 Promotion workflow

### 18.6 CI/CD Pipeline
**File:** `.github/workflows/` or `.gitlab-ci.yml`

**Tasks:**
- [ ] 18.6.1 Build & test pipeline
- [ ] 18.6.2 Container image build
- [ ] 18.6.3 Security scanning (Trivy)
- [ ] 18.6.4 Staging deployment
- [ ] 18.6.5 Production deployment
- [ ] 18.6.6 Smoke tests

---

## Phase 19: Performance Optimization (P2)

**Timeline:** 1-2 tuần  
**Current:** 7/10 → **Target:** 9/10

### 19.1 Request Batching
**File:** `server/batch.py`

**Tasks:**
- [ ] 19.1.1 Embedding batch requests
- [ ] 19.1.2 LLM batch inference
- [ ] 19.1.3 Batch window configuration
- [ ] 19.1.4 Batch size limits

### 19.2 Connection Pooling
**File:** `server/connections.py`

**Tasks:**
- [ ] 19.2.1 httpx connection pool for vLLM
- [ ] 19.2.2 asyncpg pool for Postgres
- [ ] 19.2.3 aioredis pool for Redis
- [ ] 19.2.4 Connection health monitoring

### 19.3 Speculative Execution
**File:** `server/speculative.py`

**Tasks:**
- [ ] 19.3.1 Pre-warm common prompts
- [ ] 19.3.2 Predictive context loading
- [ ] 19.3.3 Background pre-computation

### 19.4 Response Streaming Optimization
**File:** `server/streaming/optimized.py`

**Tasks:**
- [ ] 19.4.1 Chunked transfer encoding
- [ ] 19.4.2 Buffer management
- [ ] 19.4.3 Backpressure handling
- [ ] 19.4.4 Client disconnect detection

---

## Phase 20: Developer Experience (P2)

**Timeline:** 1 tuần  
**Current:** 7/10 → **Target:** 9/10

### 20.1 Developer Documentation
**Directory:** `docs/`

**Tasks:**
- [ ] 20.1.1 Architecture overview
- [ ] 20.1.2 API documentation
- [ ] 20.1.3 Development setup guide
- [ ] 20.1.4 Contributing guide
- [ ] 20.1.5 Troubleshooting guide

### 20.2 Local Development Mode
**File:** `server/dev_mode.py`

**Tasks:**
- [ ] 20.2.1 Mock vLLM responses
- [ ] 20.2.2 In-memory Qdrant
- [ ] 20.2.3 SQLite instead of Postgres
- [ ] 20.2.4 Hot-reload configuration
- [ ] 20.2.5 Debug endpoints

### 20.3 Integration Tests
**Directory:** `tests/integration/`

**Tasks:**
- [ ] 20.3.1 End-to-end graph tests
- [ ] 20.3.2 API integration tests
- [ ] 20.3.3 External service mocks
- [ ] 20.3.4 Test containers setup

### 20.4 CLI Tool
**File:** `cli/main.py`

**Tasks:**
- [ ] 20.4.1 Index command
- [ ] 20.4.2 Query command
- [ ] 20.4.3 Health check command
- [ ] 20.4.4 Config validation

---

## Summary

### Total Tasks: ~200 tasks across 11 phases

### Timeline Estimate

| Phase | Timeline | Priority |
|-------|----------|----------|
| 10 - Security | 1-2 tuần | P0 |
| 11 - Reliability | 1-2 tuần | P0 |
| 12 - Scaling | 2-3 tuần | P1 |
| 13 - RAG | 1-2 tuần | P1 |
| 14 - Observability | 1 tuần | P1 |
| 15 - Multi-Agent | 2-3 tuần | P2 |
| 16 - AI Engineering | 2 tuần | P2 |
| 17 - Tools | 1-2 tuần | P2 |
| 18 - Enterprise | 2-3 tuần | P2 |
| 19 - Performance | 1-2 tuần | P2 |
| 20 - DX | 1 tuần | P2 |

**Total: ~16-22 tuần (4-5.5 months)**

### Score Progression

```
Current:  7.5/10
After P0: 8.0/10 (+0.5)
After P1: 8.8/10 (+0.8)
After P2: 9.5/10 (+0.7)
```

---

*Plan generated: 2026-05-30*
*Total weaknesses addressed: 47*
*Total tasks: ~200*
