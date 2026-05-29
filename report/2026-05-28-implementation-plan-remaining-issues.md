# Implementation Plan — Remaining Issues

**Date:** 2026-05-28  
**Status:** ✅ COMPLETED  
**Total:** 13 issues → 4 remaining (architectural decisions)  
**Actual effort:** ~2 hours

---

## Phase 1: Quick Wins (30 min)

### 1.1 Fix Performance Issue #9
**File:** `server/agent/generate.py`  
**Issue:** `_get_already_called_tools` scans history multiple times

```python
# Current: O(n²) - scans messages twice, parses JSON repeatedly
def _get_already_called_tools(state):
    for msg in messages:
        for tc in tool_calls:
            args = _parse_json_safe(fn.get("arguments"))  # JSON parse per call

# Fix: Cache parsed results, single pass
def _get_already_called_tools(state) -> set[str]:
    already_called: set[str] = set()
    for msg in state.get("messages", []):
        if getattr(msg, "type", None) != "ai":
            continue
        tc_list = (getattr(msg, "additional_kwargs", {}) or {}).get("tool_calls", [])
        for tc in tc_list:
            fn = tc.get("function", {})
            name = fn.get("name", "")
            # Cache normalized args string directly instead of re-parsing
            args_str = fn.get("arguments", "{}")
            call_key = f"{name}:{args_str}"
            already_called.add(call_key)
    return already_called
```

**Test:** Run existing tests + benchmark with large conversation history

---

### 1.2 Use Shared Utilities (#26-27)
**Files:** `chat.py`, `classify_intent.py`, `generate.py`, `review_analyze.py`

```python
# Replace local functions with shared utilities
from server.utils.content import normalize_content
from server.utils.json_parser import parse_json_safe, extract_json_object

# chat.py: Replace _flatten_content
# Before
def _flatten_content(content) -> str: ...

# After
from server.utils.content import normalize_content as _flatten_content
```

**Checklist:**
- [ ] `chat.py:94` - replace `_flatten_content` → `normalize_content`
- [ ] `classify_intent.py:312` - replace inline normalization
- [ ] `generate.py:384` - replace `_parse_json_safe` → `parse_json_safe`
- [ ] `generate.py:419` - replace content normalization
- [ ] `review_analyze.py:133` - replace `_parse_json_object` → `extract_json_object`

---

## Phase 2: Plugin Refactor (1-2 hours)

### 2.1 Extract `_extract_by_mode()` to BasePlugin (#24)

**Current state:** Each plugin has nearly identical `_extract_by_mode()`:
```python
# java_plugin.py, go_plugin.py, ts_plugin.py, etc.
@staticmethod
def _extract_by_mode(node, source: bytes, mode: str) -> str:
    if mode == "signature":
        # ... same logic
    elif mode == "full":
        # ... same logic
```

**Fix:** Move to `BasePlugin`:

```python
# mcp_server/plugins/base.py
class BasePlugin(ABC):
    @staticmethod
    def _extract_by_mode(node, source: bytes, mode: str) -> str:
        """Extract code from node based on mode (signature/full/outline)."""
        if mode == "signature":
            # First line only
            start = node.start_byte
            end = source.find(b"\n", start)
            if end == -1:
                end = node.end_byte
            return source[start:end].decode("utf-8", errors="replace").strip()
        elif mode == "full":
            return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
        else:  # outline
            # First line + "..."
            start = node.start_byte
            end = source.find(b"\n", start)
            if end == -1:
                end = node.end_byte
            first_line = source[start:end].decode("utf-8", errors="replace").strip()
            return f"{first_line} ..."
```

**Plugins to update:**
- [ ] `java_plugin.py` - remove `_extract_by_mode`, use inherited
- [ ] `go_plugin.py` - remove `_extract_by_mode`, use inherited
- [ ] `typescript_plugin.py` - remove `_extract_by_mode`, use inherited
- [ ] `python_plugin.py` - remove `_extract_by_mode`, use inherited
- [ ] `csharp_plugin.py` - remove `_extract_by_mode`, use inherited
- [ ] `hcl_plugin.py` - remove `_extract_by_mode`, use inherited

---

### 2.2 Extract Tree Walk Helper (#25)

**Current state:** `java_plugin.py` and `go_plugin.py` have identical `_walk()`:
```python
def _walk(node):
    yield node
    for child in node.children:
        yield from _walk(child)
```

**Fix:** Add to `BasePlugin`:

```python
# mcp_server/plugins/base.py
class BasePlugin(ABC):
    @staticmethod
    def _walk(node):
        """Recursively yield all nodes in tree."""
        yield node
        for child in node.children:
            yield from BasePlugin._walk(child)
```

---

## Phase 3: Architectural Features (2-3 hours)

### 3.1 Lightweight Validation (Optional)

**Goal:** Add essential validations without full 7-pass pipeline

```python
# server/agent/post_process.py - enhance existing
def post_process(state: AgentState) -> dict:
    draft = state.get("draft", "")
    
    # Quick validations (non-blocking, just warnings)
    warnings = []
    
    # Check for common issues in generated code
    if "```java" in draft:
        code = extract_code_block(draft, "java")
        if "@SpringBootTest" in code and "@Mock" not in code:
            warnings.append("Test uses @SpringBootTest but no @Mock - may need mocking")
        if "assert" not in code.lower() and "verify" not in code.lower():
            warnings.append("Test has no assertions")
    
    if warnings:
        logger.warning("post_process warnings: %s", warnings)
        # Optionally append warnings to draft
        # draft += f"\n\n⚠️ Warnings:\n" + "\n".join(f"- {w}" for w in warnings)
    
    return {"draft": draft, "validation_warnings": warnings}
```

---

### 3.2 Simple Metrics Tracking

**Goal:** Track key metrics without full MetricsCollector

Already partially implemented in `server/metrics/`. Enhance:

```python
# server/metrics/models.py - add fields
@dataclass
class RequestMetrics:
    # ... existing fields
    validation_warnings: list[str] = field(default_factory=list)
    tool_calls_made: list[str] = field(default_factory=list)
    rag_chunks_used: int = 0
```

---

### 3.3 Session Memory (Optional)

**Goal:** Simple in-memory session for multi-turn refinement

```python
# server/session.py
from collections import defaultdict
from datetime import datetime, timedelta

class SessionStore:
    def __init__(self, ttl_minutes: int = 30):
        self._store: dict[str, dict] = {}
        self._ttl = timedelta(minutes=ttl_minutes)
    
    def get(self, session_id: str) -> dict | None:
        entry = self._store.get(session_id)
        if entry and datetime.now() - entry["ts"] < self._ttl:
            return entry["data"]
        return None
    
    def set(self, session_id: str, data: dict) -> None:
        self._store[session_id] = {"data": data, "ts": datetime.now()}
    
    def cleanup(self) -> int:
        now = datetime.now()
        expired = [k for k, v in self._store.items() if now - v["ts"] >= self._ttl]
        for k in expired:
            del self._store[k]
        return len(expired)

# Singleton
_session_store = SessionStore()
def get_session_store() -> SessionStore:
    return _session_store
```

---

## Implementation Order

```
Priority | Phase | Task                          | Time   | Deps
---------|-------|-------------------------------|--------|------
1        | 1.1   | Fix _get_already_called_tools | 15 min | None
2        | 1.2   | Use shared utilities          | 15 min | None
3        | 2.1   | Extract _extract_by_mode      | 45 min | None
4        | 2.2   | Extract _walk helper          | 15 min | 2.1
5        | 3.1   | Lightweight validation        | 30 min | None
6        | 3.2   | Enhanced metrics              | 30 min | None
7        | 3.3   | Session memory (optional)     | 45 min | None
```

---

## Acceptance Criteria

### Phase 1
- [ ] All existing tests pass
- [ ] No duplicate utility functions
- [ ] Performance: O(n) for history scanning

### Phase 2
- [ ] All 6 plugins use inherited `_extract_by_mode()`
- [ ] `_walk()` defined once in BasePlugin
- [ ] Plugin tests pass

### Phase 3
- [ ] Validation warnings logged for test generation
- [ ] Metrics include new fields
- [ ] Session store works for /refine-test endpoint

---

## Rollback Plan

All changes are additive. To rollback:
1. Revert shared utilities → restore local functions
2. Revert BasePlugin → restore per-plugin methods
3. Revert validation/metrics → just remove new code

---

*Plan created: 2026-05-28*
