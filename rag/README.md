# rag/ — Retrieval Augmented Generation

Manages the Qdrant vector database and semantic search. Converts code into embeddings using sentence-transformers and retrieves relevant chunks for context-aware test generation.

## Files

| File | Description |
|------|-------------|
| `client.py` | `RAGClient` — search, search_by_class (with dependency resolution + unfound type tracking), search_by_method, metadata filtering, index stats. 3-tier search strategy. |
| `schema.py` | Pydantic models: `CodeChunk` (30+ fields), `SearchQuery`, `SearchResult`, `MetadataFilter`, `IndexStats`, `FieldSchema`. |

## Search Strategy (3-tier)

When searching by class name, the client uses a cascading strategy:

1. **Semantic + metadata filter** — embed the class name and filter by `class_name` field
2. **Scroll by class name** — direct Qdrant scroll on metadata (guaranteed find if indexed)
3. **Broad semantic** — fall back to pure semantic search without filters

## Key Features

- **Dependency resolution** — `search_by_class(include_dependencies=True)` fetches the main class + all its imported dependencies + used domain types
- **Unfound type tracking** — attaches `unfound_types` to the main chunk when dependencies can't be found in the index
- **No truncation** — returns ALL relevant chunks (prompt builder handles token budgeting)
- **Rich metadata** — `CodeChunk` includes Lombok flags, `has_builder`, `java_type`, `record_components`, `fields`, method signatures

## CodeChunk Schema (key fields)

```python
class CodeChunk(BaseModel):
    class_name: str
    package: str
    java_type: str              # "class", "record", "enum", "interface"
    has_builder: bool           # Lombok @Builder detected
    fields: list[FieldSchema]   # Class fields
    record_components: list[FieldSchema]  # Record components (for records)
    dependencies: list[str]     # FQN of imported types
    used_types: list[str]       # Domain types referenced in method bodies
    unfound_types: list[str]    # Types not found in the index
    summary: str                # 150-250 token context summary
    layer: str                  # "service", "repository", "domain", "controller"
```

## Public API

```python
from rag import RAGClient, SearchQuery

client = RAGClient()
result = client.search(SearchQuery(query="UserService", top_k=10))
# result.chunks → list[CodeChunk]

chunks = client.search_by_class("AuthUseCaseService", include_dependencies=True)
# chunks[0].unfound_types → ["LoginPassword", "OpenAPIToken"]
```

## Dependencies

- `qdrant-client` — Qdrant vector database client
- `sentence-transformers` — embedding model (`all-MiniLM-L6-v2`)
