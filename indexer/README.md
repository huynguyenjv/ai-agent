# indexer/ — Java Code Indexing

Parses Java source code using tree-sitter, generates semantic summaries, and upserts vectors into Qdrant.

## Pipeline

```
Java repo
    │
    ▼
JavaParser (tree-sitter)
    │  Extracts ClassInfo: methods, fields, imports, Lombok,
    │  record components, enum constants, type references
    ▼
CodeSummarizer
    │  Generates 150-250 token summaries with usage hints
    │  (constructor, builder, factory, record instantiation)
    ▼
IndexBuilder
    │  Resolves FQNs, builds dependency graph,
    │  generates embeddings, upserts to Qdrant
    ▼
Qdrant (java_codebase collection)
```

## Files

| File | Description |
|------|-------------|
| `parse_java.py` | `JavaParser` — tree-sitter Java parser. Extracts `ClassInfo` with methods, fields, imports, Lombok annotations, record components, enum constants, type references, inner classes. |
| `summarize.py` | `CodeSummarizer` — generates 150-250 token summaries for RAG indexing. Includes usage hints (how to instantiate: constructor, builder, factory, record). |
| `build_index.py` | `IndexBuilder` — scans a Java repo, builds dependency graph, resolves FQNs, generates embeddings, upserts to Qdrant. Creates rich payloads with Lombok flags, fields, method signatures. |

## Key Features

- **Lombok detection** — `@Builder`, `@Data`, `@Value`, `@Getter`, `@Setter`, `@NoArgsConstructor`, `@AllArgsConstructor`
- **Record support** — extracts record components as typed fields
- **DDD layer detection** — classifies by package path (`service`, `repository`, `domain`, `controller`, `config`, etc.) and by annotations
- **Model class detection** — package-based (`domain/model/dto/vo/entity/payload/event`) + annotation-based (`@Entity`, `@Document`) + name-based patterns
- **Dependency graph** — resolves fully-qualified names, tracks imports and type references
- **Rich payloads** — Qdrant payload includes `has_builder`, `java_type`, `fields`, `record_components`, `method_signatures`, `layer`, `package`

## Public API

```python
from indexer import IndexBuilder

builder = IndexBuilder(qdrant_client, embedding_model)
stats = builder.build_index("/path/to/java/repo", recreate=False)
# stats → {"total_classes": 150, "indexed": 148, "errors": 2}
```

## Dependencies

- `tree-sitter` + `tree-sitter-java` — Java AST parsing
- `sentence-transformers` — embedding generation
- `qdrant-client` — vector database
