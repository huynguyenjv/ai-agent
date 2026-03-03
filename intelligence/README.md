# intelligence/ — Repo Structural Intelligence

Graph-based and symbol-level intelligence on Java codebases. Goes beyond vector search to provide structural answers: "what mocks does X need?", "what types are related?", "what is the impact of changing Y?".

## Files

| File | Description |
|------|-------------|
| `repo_scanner.py` | `RepoScanner` — scans a repo with `JavaParser`, produces `RepoSnapshot` with O(1) lookups by name, FQN, or file path. |
| `file_graph.py` | `FileGraph` — file-level directed graph based on import relationships. Finds dependencies, dependents, transitive closures. |
| `symbol_map.py` | `SymbolMap` — global symbol table. O(1) lookup: class→methods/fields, method→classes, field_type→injectors, annotation→classes. |
| `dependency_analyzer.py` | `DependencyAnalyzer` — merges FileGraph + SymbolMap. Produces `TestContext` (what mocks, what types) and `ImpactReport` (what breaks if X changes). |

## Key Queries

```python
from intelligence import DependencyAnalyzer

analyzer = DependencyAnalyzer(repo_scanner, file_graph, symbol_map)

# What mocks does this service need?
ctx = analyzer.test_context_for("AuthUseCaseService")
# ctx.mocks → ["OpenAPIRepository", "UserQueryService", ...]
# ctx.domain_types → ["UserProfile", "JwtToken", ...]
# ctx.layer → "service"

# What breaks if we change this class?
report = analyzer.impact_of("UserProfile")
# report.direct_dependents → ["AuthUseCaseService", "UserUseCase"]
# report.transitive_dependents → [...]
```

## Dependencies

- `indexer/` — uses `JavaParser` and `ClassInfo` for AST extraction
