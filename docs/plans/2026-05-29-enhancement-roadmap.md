# AI Agent Enhancement Roadmap

**Based on:** Principal Architect Review (2026-05-29)  
**Current Score:** 7.0/10  
**Target Score:** 9.0/10  
**Scope:** Code & Feature enhancements only (no infra/deployment)

---

## Phase 4: Security Hardening (Week 1)

### 4.1 Fix shell=True Vulnerability
**Priority:** CRITICAL  
**Effort:** 2 hours  
**File:** `mcp_server/tools.py`

```python
# Before (UNSAFE)
subprocess.run(command, shell=True, ...)

# After (SAFE)
subprocess.run(shlex.split(command), shell=False, ...)
```

**Tasks:**
- [ ] Replace `shell=True` with `shell=False` + `shlex.split()`
- [ ] Update git tools to use list arguments
- [ ] Add tests for command injection attempts

### 4.2 Prompt Injection Defense
**Priority:** CRITICAL  
**Effort:** 4 hours  
**File:** `server/utils/sanitize.py` (NEW)

```python
def sanitize_user_input(text: str) -> str:
    """Remove prompt injection attempts."""
    # Remove system prompt markers
    markers = ["<|system|>", "<|assistant|>", "<<SYS>>", "[INST]"]
    for m in markers:
        text = text.replace(m, "")
    
    # Escape special sequences
    text = text.replace("```system", "```text")
    
    # Limit length
    return text[:MAX_USER_INPUT_LENGTH]

def detect_jailbreak(text: str) -> bool:
    """Detect common jailbreak patterns."""
    patterns = [
        r"ignore.*previous.*instructions",
        r"pretend.*you.*are",
        r"act.*as.*if",
        r"bypass.*safety",
    ]
    return any(re.search(p, text, re.I) for p in patterns)
```

**Tasks:**
- [ ] Create `server/utils/sanitize.py`
- [ ] Wire sanitization to `classify_intent` input
- [ ] Wire sanitization to `generate` prompt building
- [ ] Add jailbreak detection with logging
- [ ] Add tests for injection patterns

### 4.3 Tool Access Control
**Priority:** HIGH  
**Effort:** 3 hours  
**File:** `server/agent/generate.py`

```python
# Role-based tool permissions
TOOL_PERMISSIONS = {
    "readonly": ["vtrip_read_file", "vtrip_search_symbol", "vtrip_git_status", "vtrip_git_log"],
    "developer": ["readonly"] + ["vtrip_run_command", "vtrip_diff_preview"],
    "admin": ["developer"] + ["vtrip_apply_edits", "vtrip_git_commit"],
}

def filter_tools_by_role(tools: list, role: str) -> list:
    allowed = set()
    for r in TOOL_PERMISSIONS.get(role, []):
        if r in TOOL_PERMISSIONS:
            allowed.update(TOOL_PERMISSIONS[r])
        else:
            allowed.add(r)
    return [t for t in tools if t["function"]["name"] in allowed]
```

**Tasks:**
- [ ] Add `role` field to ChatRequest
- [ ] Implement `filter_tools_by_role()`
- [ ] Add audit logging for tool usage
- [ ] Add tests for permission filtering

---

## Phase 5: Multi-Agent Architecture (Week 2-3)

### 5.1 Planner Agent
**Priority:** HIGH  
**Effort:** 8 hours  
**File:** `server/agent/planner.py` (NEW)

```python
async def plan_task(
    state: AgentState,
    vllm_client,
    model: str,
) -> dict:
    """Decompose complex task into steps."""
    
    PLANNER_PROMPT = """You are a task planner for a coding assistant.
    
Given the user request, break it down into atomic steps.
Each step should be:
- Self-contained
- Verifiable
- Has clear inputs/outputs

Output JSON:
{
    "complexity": "simple|medium|complex",
    "steps": [
        {"id": 1, "action": "read_file", "target": "...", "reason": "..."},
        {"id": 2, "action": "generate_code", "depends_on": [1], "reason": "..."},
        ...
    ],
    "estimated_tools": ["vtrip_read_file", "vtrip_apply_edits"]
}
"""
    # LLM call to generate plan
    ...
    
    return {
        "task_plan": plan,
        "plan_steps": steps,
        "current_step": 0,
    }
```

**Tasks:**
- [ ] Create `server/agent/planner.py`
- [ ] Add planning prompt with structured output
- [ ] Add plan validation (cycle detection, dependency check)
- [ ] Wire to graph for complex tasks (>3 tools estimated)
- [ ] Add step tracking in AgentState

### 5.2 Critic Agent
**Priority:** HIGH  
**Effort:** 6 hours  
**File:** `server/agent/critic.py` (NEW)

```python
async def critique_output(
    state: AgentState,
    vllm_client,
    model: str,
) -> dict:
    """Review generated output for quality issues."""
    
    CRITIC_PROMPT = """Review this code/response for:
1. Correctness - Does it solve the problem?
2. Completeness - Are all requirements addressed?
3. Quality - Is it well-structured and maintainable?
4. Safety - Any security issues?

Output JSON:
{
    "passed": true/false,
    "issues": [
        {"severity": "high|medium|low", "category": "...", "description": "...", "suggestion": "..."}
    ],
    "score": 0-10,
    "retry_with_feedback": "specific instructions if retry needed"
}
"""
    ...
    
    return {
        "critic_passed": passed,
        "critic_issues": issues,
        "critic_feedback": feedback,
    }
```

**Tasks:**
- [ ] Create `server/agent/critic.py`
- [ ] Add critic prompt with scoring rubric
- [ ] Wire after `verify_result` for code_gen/unit_test intents
- [ ] Pass critic feedback to retry loop
- [ ] Add critic metrics to Prometheus

### 5.3 Update Graph for Multi-Agent
**Priority:** HIGH  
**Effort:** 4 hours  
**File:** `server/agent/graph.py`

```
New Flow:
classify_intent
    ├─ simple task → generate → verify → post_process → END
    └─ complex task → planner → loop {
                                   execute_step → verify_step
                                 } → critic → post_process → END
```

**Tasks:**
- [ ] Add `planner` node
- [ ] Add `critic` node  
- [ ] Add complexity routing after classify_intent
- [ ] Add step execution loop
- [ ] Update AgentState with plan fields

### 5.4 Task Queue System
**Priority:** HIGH  
**Effort:** 6 hours  
**File:** `server/agent/task_queue.py` (NEW)

```python
from dataclasses import dataclass
from enum import Enum
from typing import Any
import asyncio

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class Task:
    id: str
    action: str
    params: dict[str, Any]
    depends_on: list[str]
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: str | None = None

class TaskQueue:
    """Manages task execution with dependency resolution."""
    
    def __init__(self):
        self._tasks: dict[str, Task] = {}
        self._results: dict[str, Any] = {}
    
    def add_task(self, task: Task) -> None:
        self._tasks[task.id] = task
    
    def get_ready_tasks(self) -> list[Task]:
        """Get tasks whose dependencies are satisfied."""
        ready = []
        for task in self._tasks.values():
            if task.status != TaskStatus.PENDING:
                continue
            deps_satisfied = all(
                self._tasks[dep].status == TaskStatus.COMPLETED
                for dep in task.depends_on
            )
            if deps_satisfied:
                ready.append(task)
        return ready
    
    async def execute_all(self, executor) -> dict[str, Any]:
        """Execute all tasks respecting dependencies."""
        while True:
            ready = self.get_ready_tasks()
            if not ready:
                break
            
            # Execute ready tasks in parallel
            results = await asyncio.gather(
                *[executor(task) for task in ready],
                return_exceptions=True
            )
            
            for task, result in zip(ready, results):
                if isinstance(result, Exception):
                    task.status = TaskStatus.FAILED
                    task.error = str(result)
                else:
                    task.status = TaskStatus.COMPLETED
                    task.result = result
                    self._results[task.id] = result
        
        return self._results
```

**Tasks:**
- [ ] Create `server/agent/task_queue.py`
- [ ] Add dependency resolution
- [ ] Add parallel execution for independent tasks
- [ ] Wire to Planner output
- [ ] Add task status tracking in AgentState

### 5.5 Parallel Tool Execution
**Priority:** HIGH  
**Effort:** 4 hours  
**File:** `server/agent/generate.py`

```python
async def execute_tools_parallel(
    tool_calls: list[dict],
    repo_path: str,
    registry: PluginRegistry,
) -> list[dict]:
    """Execute independent tools in parallel."""
    
    # Group by dependency
    independent = []
    dependent = []
    
    for tc in tool_calls:
        # Tools that read are independent
        # Tools that write depend on reads
        if tc["function"]["name"] in READ_ONLY_TOOLS:
            independent.append(tc)
        else:
            dependent.append(tc)
    
    # Execute independent tools in parallel
    results = []
    if independent:
        tasks = [_execute_single_tool(tc, repo_path, registry) for tc in independent]
        results.extend(await asyncio.gather(*tasks))
    
    # Execute dependent tools sequentially
    for tc in dependent:
        result = await _execute_single_tool(tc, repo_path, registry)
        results.append(result)
    
    return results

READ_ONLY_TOOLS = {
    "vtrip_read_file",
    "vtrip_search_symbol", 
    "vtrip_get_project_skeleton",
    "vtrip_git_status",
    "vtrip_git_diff",
    "vtrip_git_log",
}
```

**Tasks:**
- [ ] Add `execute_tools_parallel()` function
- [ ] Classify tools as read-only vs write
- [ ] Execute read-only tools in parallel
- [ ] Execute write tools sequentially
- [ ] Add metrics for parallel execution

---

## Phase 6: Advanced RAG (Week 3-4)

### 6.1 Wire RAG to Graph
**Priority:** HIGH  
**Effort:** 4 hours  
**File:** `server/agent/graph.py`

Currently RAG nodes exist but edges are not connected.

**Tasks:**
- [ ] Add edge: `route_context → rag_search` when RAG enabled
- [ ] Add edge: `rag_search → generate`
- [ ] Pass RAG chunks to generate prompt
- [ ] Add RAG bypass for simple queries

### 6.2 Re-ranking Layer
**Priority:** MEDIUM  
**Effort:** 6 hours  
**File:** `server/rag/reranker.py` (NEW)

```python
from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)
    
    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        """Re-rank documents by relevance to query."""
        pairs = [(query, doc["body"]) for doc in documents]
        scores = self.model.predict(pairs)
        
        ranked = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True
        )
        return [doc for doc, _ in ranked[:top_k]]
```

**Tasks:**
- [ ] Create `server/rag/reranker.py`
- [ ] Add CrossEncoder model loading
- [ ] Wire to `hybrid_search` results
- [ ] Add reranker to Prometheus metrics
- [ ] Make reranker optional via env var

### 6.3 Query Expansion
**Priority:** MEDIUM  
**Effort:** 4 hours  
**File:** `server/rag/query_expand.py` (NEW)

```python
async def expand_query(
    query: str,
    vllm_client,
    model: str,
) -> list[str]:
    """Generate query variants for better retrieval."""
    
    EXPAND_PROMPT = """Generate 3 alternative phrasings of this code search query.
Keep the same intent, vary the terminology.

Query: {query}

Output as JSON array: ["variant1", "variant2", "variant3"]
"""
    # LLM generates variants
    variants = await _call_llm(...)
    return [query] + variants  # Original + variants
```

**Tasks:**
- [ ] Create `server/rag/query_expand.py`
- [ ] Generate 2-3 query variants via LLM
- [ ] Merge results from all variants (union + dedup)
- [ ] Add caching for query expansions
- [ ] Add toggle via env var

### 6.4 Parent-Child Retrieval
**Priority:** MEDIUM  
**Effort:** 6 hours  
**File:** `server/rag/qdrant_client.py`

```python
async def retrieve_with_context(
    self,
    chunk_ids: list[str],
    context_lines: int = 50,
) -> list[dict]:
    """Retrieve chunks with surrounding context."""
    results = []
    for chunk in chunks:
        # Get parent (file-level) context
        file_content = await self._get_file_content(chunk["file_path"])
        
        # Extract window around chunk
        start = max(0, chunk["start_line"] - context_lines)
        end = chunk["end_line"] + context_lines
        
        results.append({
            **chunk,
            "context_before": file_content[start:chunk["start_line"]],
            "context_after": file_content[chunk["end_line"]:end],
        })
    return results
```

**Tasks:**
- [ ] Add `retrieve_with_context()` method
- [ ] Store file_path → content mapping in cache
- [ ] Include context in RAG prompt
- [ ] Make context_lines configurable

### 6.5 HyDE (Hypothetical Document Embedding)
**Priority:** MEDIUM  
**Effort:** 4 hours  
**File:** `server/rag/hyde.py` (NEW)

```python
async def hyde_search(
    query: str,
    vllm_client,
    model: str,
    embedder,
    qdrant: QdrantService,
) -> list[dict]:
    """Generate hypothetical document, then search with its embedding."""
    
    HYDE_PROMPT = """Given this code search query, write a hypothetical code snippet 
that would perfectly answer the query. Write actual code, not description.

Query: {query}

Hypothetical code:
```
"""
    
    # Generate hypothetical document
    hypothetical = await _call_llm(vllm_client, model, HYDE_PROMPT.format(query=query))
    
    # Embed the hypothetical document (not the query)
    hyde_embedding = await embedder.embed(hypothetical)
    
    # Search with hypothetical embedding
    results = await qdrant.hybrid_search(
        dense_vector=hyde_embedding,
        sparse_vector=compute_sparse(hypothetical),
    )
    
    return results
```

**Tasks:**
- [ ] Create `server/rag/hyde.py`
- [ ] Generate hypothetical code via LLM
- [ ] Embed hypothetical instead of query
- [ ] Combine with normal search (ensemble)
- [ ] Add toggle via env var

### 6.6 Chunk Overlap Strategy
**Priority:** MEDIUM  
**Effort:** 4 hours  
**File:** `mcp_server/tools_indexer.py`

```python
def chunk_with_overlap(
    content: str,
    chunk_size: int = 100,  # lines
    overlap: int = 20,      # 20% overlap
) -> list[dict]:
    """Split content into overlapping chunks."""
    
    lines = content.split('\n')
    chunks = []
    
    start = 0
    while start < len(lines):
        end = min(start + chunk_size, len(lines))
        
        chunk = {
            "content": '\n'.join(lines[start:end]),
            "start_line": start + 1,
            "end_line": end,
            "has_overlap": start > 0,
        }
        chunks.append(chunk)
        
        # Move start with overlap
        start = end - overlap
        if start >= len(lines) - overlap:
            break
    
    return chunks
```

**Tasks:**
- [ ] Add `chunk_with_overlap()` function
- [ ] Configure overlap percentage via env var
- [ ] Update indexer to use overlapping chunks
- [ ] Handle overlap dedup in search results
- [ ] Add migration script for re-indexing

### 6.7 Hallucination Mitigation
**Priority:** HIGH  
**Effort:** 6 hours  
**File:** `server/agent/verify_sources.py` (NEW)

```python
def verify_rag_sources(
    response: str,
    rag_chunks: list[dict],
) -> dict:
    """Verify that response is grounded in retrieved sources."""
    
    # Extract code blocks from response
    code_blocks = extract_code_blocks(response)
    
    # Check each block against RAG chunks
    verifications = []
    for block in code_blocks:
        best_match = find_best_match(block, rag_chunks)
        verifications.append({
            "block": block[:100],
            "source": best_match["file_path"] if best_match else None,
            "similarity": best_match["score"] if best_match else 0,
            "grounded": best_match["score"] > 0.7 if best_match else False,
        })
    
    grounded_count = sum(1 for v in verifications if v["grounded"])
    
    return {
        "total_blocks": len(code_blocks),
        "grounded_blocks": grounded_count,
        "grounding_rate": grounded_count / len(code_blocks) if code_blocks else 1.0,
        "verifications": verifications,
        "potentially_hallucinated": [v for v in verifications if not v["grounded"]],
    }

def add_citations(
    response: str,
    rag_chunks: list[dict],
) -> str:
    """Add source citations to response."""
    
    # Find which chunks were used
    used_chunks = match_response_to_chunks(response, rag_chunks)
    
    # Add citations
    citations = "\n\n---\n**Sources:**\n"
    for i, chunk in enumerate(used_chunks, 1):
        citations += f"[{i}] `{chunk['file_path']}` (lines {chunk['start_line']}-{chunk['end_line']})\n"
    
    return response + citations
```

**Tasks:**
- [ ] Create `server/agent/verify_sources.py`
- [ ] Implement source verification
- [ ] Add citation generation
- [ ] Wire to post_process node
- [ ] Add grounding_rate to metrics

---

## Phase 7: Enhanced Code Intelligence (Week 4-5)

### 7.1 AST-Aware Refactoring
**Priority:** MEDIUM  
**Effort:** 8 hours  
**File:** `mcp_server/tools_refactor.py` (NEW)

```python
def rename_symbol(
    repo_path: str,
    registry: PluginRegistry,
    old_name: str,
    new_name: str,
    scope: str = "file",  # file | project
) -> dict:
    """Rename symbol across codebase using AST."""
    
    # Find all occurrences via tree-sitter
    occurrences = search_symbol(repo_path, registry, old_name)
    
    edits = []
    for occ in occurrences:
        # Parse file, find exact node, replace
        plugin = registry.get_plugin(occ["file_path"])
        tree = plugin.parse(source)
        
        # Find node at position and rename
        ...
        edits.append({
            "file_path": occ["file_path"],
            "search": old_name,
            "replace": new_name,
        })
    
    return apply_edits(repo_path, edits, dry_run=True)
```

**Tasks:**
- [ ] Create `mcp_server/tools_refactor.py`
- [ ] Add `rename_symbol` tool
- [ ] Add `extract_function` tool
- [ ] Add `inline_variable` tool
- [ ] Wire to MCP_TOOLS

### 7.2 Test Execution Integration
**Priority:** HIGH  
**Effort:** 6 hours  
**File:** `mcp_server/tools.py`

```python
def run_tests(
    repo_path: str,
    test_file: str | None = None,
    test_name: str | None = None,
    framework: str = "auto",  # pytest | jest | junit | go
) -> dict:
    """Run tests and parse results."""
    
    # Auto-detect framework
    if framework == "auto":
        framework = _detect_test_framework(repo_path)
    
    # Build command
    cmd = TEST_COMMANDS[framework]
    if test_file:
        cmd += f" {test_file}"
    if test_name:
        cmd += f" -k {test_name}" if framework == "pytest" else ...
    
    result = run_command(repo_path, cmd)
    
    # Parse test output
    parsed = _parse_test_output(result["stdout"], framework)
    
    return {
        **result,
        "tests_run": parsed["total"],
        "tests_passed": parsed["passed"],
        "tests_failed": parsed["failed"],
        "failures": parsed["failure_details"],
    }
```

**Tasks:**
- [ ] Add `run_tests` tool with framework detection
- [ ] Add test output parsing for pytest/jest/junit
- [ ] Return structured failure information
- [ ] Wire failures back to retry loop

### 7.3 Linting Integration
**Priority:** MEDIUM  
**Effort:** 4 hours  
**File:** `mcp_server/tools.py`

```python
def lint_code(
    repo_path: str,
    file_path: str | None = None,
    fix: bool = False,
) -> dict:
    """Run linter and return issues."""
    
    # Detect linter
    linter = _detect_linter(repo_path)  # ruff, eslint, golint, etc.
    
    cmd = LINT_COMMANDS[linter]
    if fix:
        cmd += " --fix"
    
    result = run_command(repo_path, cmd)
    issues = _parse_lint_output(result["stdout"], linter)
    
    return {
        "linter": linter,
        "issues": issues,
        "fixed": fix,
    }
```

**Tasks:**
- [ ] Add `lint_code` tool
- [ ] Support ruff, eslint, golint
- [ ] Parse lint output to structured format
- [ ] Optionally auto-fix in verify loop

---

## Phase 8: Context & Caching Optimization (Week 5-6)

### 8.1 Smart Context Selection
**Priority:** HIGH  
**Effort:** 6 hours  
**File:** `server/agent/context_builder.py` (NEW)

```python
def build_optimal_context(
    state: AgentState,
    token_budget: int = 8000,
) -> str:
    """Build context within token budget, prioritizing relevance."""
    
    context_parts = []
    used_tokens = 0
    
    # Priority 1: Active file (always include)
    if state.get("active_file"):
        content = read_active_file(state)
        context_parts.append(("active_file", content, priority=10))
    
    # Priority 2: RAG chunks (sorted by relevance)
    for chunk in state.get("rag_chunks", []):
        context_parts.append(("rag", chunk["body"], priority=chunk["rrf_score"]))
    
    # Priority 3: Mentioned files
    for f in state.get("mentioned_files", []):
        context_parts.append(("mentioned", read_file(f), priority=5))
    
    # Sort by priority and fit within budget
    context_parts.sort(key=lambda x: x[2], reverse=True)
    
    final_context = []
    for name, content, _ in context_parts:
        tokens = estimate_tokens(content)
        if used_tokens + tokens <= token_budget:
            final_context.append(f"### {name}\n{content}")
            used_tokens += tokens
    
    return "\n\n".join(final_context)
```

**Tasks:**
- [ ] Create `server/agent/context_builder.py`
- [ ] Implement priority-based context selection
- [ ] Add token counting (tiktoken or estimate)
- [ ] Wire to generate node
- [ ] Add context utilization metrics

### 8.2 Conversation Summarization
**Priority:** MEDIUM  
**Effort:** 4 hours  
**File:** `server/agent/summarize.py` (NEW)

```python
async def summarize_conversation(
    messages: list,
    vllm_client,
    model: str,
    max_tokens: int = 500,
) -> str:
    """Summarize long conversation to fit context window."""
    
    SUMMARY_PROMPT = """Summarize this conversation, preserving:
- User's main goal
- Key decisions made
- Important code/files mentioned
- Current state of the task

Keep under {max_tokens} tokens.
"""
    ...
```

**Tasks:**
- [ ] Create `server/agent/summarize.py`
- [ ] Trigger summarization when messages > threshold
- [ ] Store summary in session
- [ ] Prepend summary to truncated context

### 8.3 Wire Embedding Cache to RAG
**Priority:** HIGH  
**Effort:** 2 hours  
**File:** `server/rag/embedder.py`

```python
from server.cache import get_embedding_cache

class Embedder:
    async def embed(self, text: str) -> list[float]:
        cache = get_embedding_cache()
        
        # Check cache first
        cached = cache.get(text)
        if cached:
            return cached
        
        # Compute embedding
        embedding = self._model.encode(text).tolist()
        
        # Cache it
        cache.set(text, embedding)
        return embedding
```

**Tasks:**
- [ ] Wire `EmbeddingCache` to `Embedder.embed()`
- [ ] Add cache hit/miss metrics
- [ ] Add batch embedding with cache

### 8.4 LLM Response Caching
**Priority:** HIGH  
**Effort:** 4 hours  
**File:** `server/cache.py`

```python
import hashlib
import json

class LLMResponseCache:
    """Cache LLM responses for repeated/similar queries."""
    
    def __init__(self, max_size: int = 1000, ttl_hours: float = 1):
        self._cache = LRUCache(max_size=max_size, default_ttl=ttl_hours * 3600)
    
    def _make_key(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        intent: str,
    ) -> str:
        """Create cache key from request params."""
        # Only use last 3 messages for key (conversation-specific)
        recent = messages[-3:] if len(messages) > 3 else messages
        
        key_data = {
            "messages": [{"role": m["role"], "content": m.get("content", "")[:500]} for m in recent],
            "tools": [t["function"]["name"] for t in (tools or [])],
            "intent": intent,
        }
        return hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()[:32]
    
    def get(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        intent: str,
    ) -> dict | None:
        """Get cached response if exists."""
        key = self._make_key(messages, tools, intent)
        return self._cache.get(key)
    
    def set(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        intent: str,
        response: dict,
    ) -> None:
        """Cache response."""
        key = self._make_key(messages, tools, intent)
        self._cache.set(key, response)

# Singleton
_llm_cache: LLMResponseCache | None = None

def get_llm_cache() -> LLMResponseCache:
    global _llm_cache
    if _llm_cache is None:
        _llm_cache = LLMResponseCache()
    return _llm_cache
```

**Wire to generate.py:**
```python
async def generate(...) -> dict:
    cache = get_llm_cache()
    
    # Check cache for non-tool-result turns
    if not state.get("is_tool_result_turn"):
        cached = cache.get(messages, tools, intent)
        if cached:
            logger.info("LLM cache hit")
            return cached
    
    # ... generate response ...
    
    # Cache successful responses (no tool calls = final answer)
    if not final_tool_calls and draft:
        cache.set(messages, tools, intent, result)
    
    return result
```

**Tasks:**
- [ ] Add `LLMResponseCache` class to cache.py
- [ ] Create cache key from messages + tools + intent
- [ ] Wire to generate node
- [ ] Only cache final responses (not tool-calling turns)
- [ ] Add cache hit/miss metrics
- [ ] Make TTL configurable

### 8.5 RAG Result Caching
**Priority:** HIGH  
**Effort:** 3 hours  
**File:** `server/rag/qdrant_client.py`

```python
from server.cache import LRUCache

class QdrantService:
    def __init__(self, ...):
        ...
        self._search_cache = LRUCache(max_size=500, default_ttl=300)  # 5 min TTL
    
    async def hybrid_search(
        self,
        dense_vector: list[float],
        sparse_vector: dict[int, float],
        lang_filter: str | None = None,
        top_k: int = 8,
        use_cache: bool = True,
    ) -> list[dict]:
        # Create cache key
        if use_cache:
            cache_key = self._make_search_key(dense_vector, sparse_vector, lang_filter, top_k)
            cached = self._search_cache.get(cache_key)
            if cached:
                logger.info("RAG search cache hit")
                return cached
        
        # ... perform search ...
        
        # Cache results
        if use_cache:
            self._search_cache.set(cache_key, results)
        
        return results
```

**Tasks:**
- [ ] Add search result caching to QdrantService
- [ ] Use vector hash for cache key
- [ ] Short TTL (5 min) to handle code changes
- [ ] Add bypass option for force-fresh search
- [ ] Add cache metrics

### 8.6 Async File I/O
**Priority:** MEDIUM  
**Effort:** 6 hours  
**File:** `mcp_server/tools.py`, `mcp_server/tools_indexer.py`

```python
import aiofiles
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Thread pool for blocking I/O
_executor = ThreadPoolExecutor(max_workers=4)

async def read_file_async(
    repo_path: str,
    file_path: str,
    start_line: int = 1,
    end_line: int = 150,
) -> dict:
    """Async version of read_file."""
    abs_path = os.path.join(repo_path, file_path)
    
    # Validation (sync is fine)
    if not os.path.isfile(abs_path):
        return {"error": f"File not found: {file_path}"}
    
    # Async file read
    async with aiofiles.open(abs_path, "r", encoding="utf-8", errors="replace") as f:
        all_lines = await f.readlines()
    
    # ... rest of logic ...
    return result

async def walk_directory_async(repo_path: str) -> list[str]:
    """Async directory walk using thread pool."""
    loop = asyncio.get_event_loop()
    
    def _walk_sync():
        files = []
        for root, dirs, filenames in os.walk(repo_path):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for f in filenames:
                files.append(os.path.join(root, f))
        return files
    
    return await loop.run_in_executor(_executor, _walk_sync)
```

**Tasks:**
- [ ] Add `aiofiles` to requirements.txt
- [ ] Create async versions of file read functions
- [ ] Use thread pool for os.walk
- [ ] Update tools to use async I/O
- [ ] Benchmark improvement

---

## Phase 9: Evaluation & Feedback Loop (Week 6-7)

### 9.1 Offline Evaluation Suite
**Priority:** MEDIUM  
**Effort:** 8 hours  
**File:** `eval/benchmark.py` (NEW)

```python
BENCHMARK_CASES = [
    {
        "id": "code_gen_simple",
        "input": "Write a Python function to calculate factorial",
        "expected_intent": "code_gen",
        "expected_contains": ["def factorial", "return"],
        "expected_not_contains": ["import os"],
    },
    {
        "id": "unit_test_java",
        "input": "Write unit tests for UserService.createUser()",
        "expected_intent": "unit_test",
        "expected_contains": ["@Test", "assert"],
    },
    ...
]

async def run_benchmark(agent, cases: list) -> dict:
    results = []
    for case in cases:
        output = await agent.invoke(case["input"])
        
        score = evaluate_output(output, case)
        results.append({
            "case_id": case["id"],
            "passed": score >= 0.8,
            "score": score,
            "output_preview": output[:200],
        })
    
    return {
        "total": len(results),
        "passed": sum(1 for r in results if r["passed"]),
        "avg_score": sum(r["score"] for r in results) / len(results),
        "details": results,
    }
```

**Tasks:**
- [ ] Create `eval/` directory
- [ ] Define 20+ benchmark cases
- [ ] Implement scoring functions
- [ ] Add regression detection (compare to baseline)
- [ ] Create CLI for running benchmarks

### 9.2 Feedback → Prompt Refinement
**Priority:** MEDIUM  
**Effort:** 6 hours  
**File:** `server/feedback_analyzer.py` (NEW)

```python
def analyze_feedback_patterns(days: int = 7) -> dict:
    """Analyze feedback to identify improvement areas."""
    
    feedback = load_feedback(days)
    
    # Group by intent
    by_intent = group_by(feedback, "intent")
    
    # Find low-rated intents
    problem_intents = [
        intent for intent, items in by_intent.items()
        if avg_rating(items) < 3.5
    ]
    
    # Extract common complaint keywords
    complaints = extract_keywords(
        [f["comment"] for f in feedback if f["rating"] <= 2]
    )
    
    return {
        "problem_intents": problem_intents,
        "common_complaints": complaints,
        "suggested_prompt_changes": generate_suggestions(complaints),
    }
```

**Tasks:**
- [ ] Create feedback analysis pipeline
- [ ] Identify low-performing intents
- [ ] Extract complaint patterns
- [ ] Generate prompt improvement suggestions
- [ ] Create weekly report

---

## Summary

| Phase | Focus | Items | Effort | Score Impact |
|-------|-------|-------|--------|--------------|
| 4 | Security Hardening | 4.1-4.3 | 1 week | +0.5 |
| 5 | Multi-Agent + Parallel | 5.1-5.5 | 2 weeks | +1.0 |
| 6 | Advanced RAG | 6.1-6.7 | 2 weeks | +0.7 |
| 7 | Code Intelligence | 7.1-7.3 | 2 weeks | +0.5 |
| 8 | Context & Caching | 8.1-8.6 | 2 weeks | +0.5 |
| 9 | Evaluation | 9.1-9.2 | 1 week | +0.3 |

**Total Timeline:** 10 weeks  
**Expected Final Score:** 9.5/10

---

## Coverage Analysis

### Issues Covered by Plan

| Category | Total Issues | Covered | Coverage |
|----------|--------------|---------|----------|
| Security | 5 | 5 | **100%** ✅ |
| Architecture | 6 | 6 | **100%** ✅ |
| RAG | 7 | 7 | **100%** ✅ |
| Coding | 6 | 4 | 67% |
| Performance | 5 | 4 | **80%** ✅ |
| AI Engineering | 6 | 4 | 67% |
| **TOTAL** | **35** | **30** | **86%** ✅ |

### Items NOT Covered (Infrastructure/Complex)

| Issue | Reason |
|-------|--------|
| LSP integration | Cần project riêng, complex |
| Dependency graph | Cần static analysis engine |
| A/B testing | Cần infrastructure riêng |
| Fine-tuning pipeline | Cần ML platform |
| Container isolation | Infrastructure scope |

---

## New Items Added (vs Previous Version)

| Phase | New Item | Addresses |
|-------|----------|-----------|
| 5.4 | Task Queue System | A-W4 (task queue) |
| 5.5 | Parallel Tool Execution | P-W1 (serial execution) |
| 6.5 | HyDE | R-W2 (HyDE) |
| 6.6 | Chunk Overlap | R-W3 (chunk overlap) |
| 6.7 | Hallucination Mitigation | Hallucination issues |
| 8.4 | LLM Response Caching | P-W5 (LLM caching) |
| 8.5 | RAG Result Caching | P-W3 (RAG caching) |
| 8.6 | Async File I/O | P-W2 (blocking I/O) |

---

## Priority Matrix

```
                    HIGH IMPACT
                        │
    ┌───────────────────┼───────────────────┐
    │                   │                   │
    │  Multi-Agent      │  Security         │
    │  Task Queue       │  Hardening        │
    │  Parallel Tools   │                   │
    │                   │                   │
LOW ├───────────────────┼───────────────────┤ HIGH
EFFORT                  │                   EFFORT
    │                   │                   │
    │  Cache Wiring     │  HyDE             │
    │  LLM Caching      │  Evaluation       │
    │  Async I/O        │  Hallucination    │
    │                   │                   │
    └───────────────────┼───────────────────┘
                        │
                    LOW IMPACT
```

---

## Implementation Order

**Week 1:** Phase 4 (Security) - CRITICAL
- 4.1 Fix shell=True
- 4.2 Prompt injection defense
- 4.3 Role-based tool access

**Week 2-3:** Phase 5 (Multi-Agent)
- 5.1 Planner agent
- 5.2 Critic agent
- 5.3 Update graph
- 5.4 Task queue
- 5.5 Parallel tools

**Week 3-4:** Phase 6 (RAG)
- 6.1 Wire RAG to graph
- 6.2 Re-ranking
- 6.5 HyDE
- 6.7 Hallucination mitigation

**Week 4-5:** Phase 7 (Code Intelligence)
- 7.2 Test runner (HIGH priority)
- 7.1 AST refactoring
- 7.3 Linting

**Week 5-6:** Phase 8 (Caching)
- 8.3 Embedding cache
- 8.4 LLM response cache
- 8.5 RAG result cache
- 8.6 Async I/O

**Week 6-7:** Phase 9 (Evaluation)
- 9.1 Benchmark suite
- 9.2 Feedback analysis

---

*Plan created: 2026-05-29*  
*Updated: 2026-05-29 (Full coverage)*  
*Based on: Principal Architect Review + Baseline Assessment*
