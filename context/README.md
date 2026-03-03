# context/ — Smart Context Assembly

Replaces brute-force "dump all RAG chunks" with a structured pipeline: intelligence analysis → snippet selection → token optimization.

## Pipeline

```
ContextBuilder.build_context(class_name)
        │
        ├── Intelligence Layer (optional)
        │     DependencyAnalyzer → TestContext
        │     (mocks needed, related types, layer info)
        │
        ├── RAG Search
        │     search_by_class(include_dependencies=True)
        │
        ├── SnippetSelector
        │     Priority-based selection:
        │     P1: target source → P2: mockable deps
        │     → P3: domain types → P4: interfaces
        │     → P5: transitive
        │
        └── TokenOptimizer
              Budget-aware truncation
              (~4 chars/token heuristic)
              Keep high-priority, trim/drop low-priority
```

## Files

| File | Description |
|------|-------------|
| `context_builder.py` | `ContextBuilder` — main entry point. Coordinates intelligence + snippet selection + token optimization. Produces `ContextResult`. |
| `snippet_selector.py` | `SnippetSelector` — priority-based snippet selection with 5 priority tiers. |
| `token_optimizer.py` | `TokenOptimizer` — budget-aware truncation. Keeps high-priority snippets, truncates/drops lower-priority ones. |

## Public API

```python
from context import ContextBuilder

builder = ContextBuilder(rag_client, intelligence=dependency_analyzer)
result = builder.build_context(
    class_name="AuthUseCaseService",
    file_path="AuthUseCaseService.java",
    max_tokens=8000,
)
# result.snippets → list[Snippet] (priority-ordered)
# result.rag_chunks → list[CodeChunk] (original RAG data)
# result.token_count → int
```

## Dependencies

- `rag/` — vector search
- `intelligence/` — optional structural intelligence (graceful degradation if unavailable)
