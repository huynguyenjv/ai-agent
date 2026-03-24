# AI Coding Agent — Implementation Plan

> **For Claude Opus 4.6:** This document is the complete, authoritative design specification. Every architectural decision has a deliberate rationale. Do not substitute simpler alternatives without understanding the trade-offs documented in Section 13. Implement phases in sequence. Each phase must pass its acceptance criteria before proceeding.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture Principles](#2-architecture-principles)
3. [Component Map](#3-component-map)
4. [Data Models](#4-data-models)
5. [MCP Server — Client Machine](#5-mcp-server--client-machine)
6. [Language Plugin System](#6-language-plugin-system)
7. [Cloud VM — FastAPI Server](#7-cloud-vm--fastapi-server)
8. [LangGraph Agent](#8-langgraph-agent)
9. [RAG and Cache Strategy](#9-rag-and-cache-strategy)
10. [SSE Streaming and UX](#10-sse-streaming-and-ux)
11. [Context Decision Flow (5-Gate)](#11-context-decision-flow-5-gate)
12. [Integration Points and API Contracts](#12-integration-points-and-api-contracts)
13. [Design Decisions and Rationale](#13-design-decisions-and-rationale)
14. [Error Handling and Edge Cases](#14-error-handling-and-edge-cases)
15. [Implementation Phases](#15-implementation-phases)
16. [Acceptance Criteria per Phase](#16-acceptance-criteria-per-phase)
17. [Configuration and Environment Variables](#17-configuration-and-environment-variables)
18. [Dependency List](#18-dependency-list)

---

## 1. System Overview

### Purpose

A language-agnostic AI coding agent that integrates with the Continue IDE plugin to provide context-aware code assistance — generation, unit test writing, explanation, structural analysis, and refactoring — across any programming language or infrastructure-as-code dialect.

### What This System Is Not

- Not a general-purpose chatbot. Every response is grounded in the actual codebase.
- Not a static analysis tool. It uses LLM generation, not rule-based inspection.
- Not a hosted SaaS. It runs on a self-managed Cloud VM with a client-side MCP server on each developer machine.

### High-Level Flow

A developer types a query in Continue. Continue routes the request to the Cloud VM agent via HTTPS. The agent classifies intent, decides what context to retrieve (from Qdrant cache or live from the client filesystem via MCP tools), assembles a prompt, and streams a response token-by-token back to Continue. While processing, the agent streams structured step events so the developer always sees what is happening.

### Supported Languages

Java, C# (.NET), Go, Python, TypeScript/JavaScript, Terraform (HCL). A fallback plugin handles any other file type via regex-based extraction. Adding a new language requires only one new plugin file and one registry entry — no other component changes.

---

## 2. Architecture Principles

### P1 — Language Detection Is Isolated to One Layer

Only the Language Plugin layer knows what programming language a file is written in. Every layer below (chunking, embedding, vector storage, retrieval, generation) works exclusively with a universal CodeChunk schema. This means the RAG system, the agent graph, and Qdrant are fully language-agnostic.

### P2 — Memory-First, Tool on Miss

Qdrant is the L1 cache. Before calling any MCP tool, the agent always checks Qdrant first. A tool call is triggered only when the cache misses, the cached data fails hash verification, or a freshness override is active. This minimizes round-trips to the client machine.

### P3 — Explicit File Mention Overrides Everything

When a user's query explicitly references a specific file (by name, @mention, or deictic reference to the currently open file), this is treated as a hard signal that the developer is working on that file right now. The agent bypasses all cache and forces a reindex. This is Gate 1 in the context decision flow and has the highest priority.

### P4 — Streaming Begins Immediately

The first SSE event must be emitted within 50ms of receiving a request. The developer must never see a blank or frozen interface. Three layers of events provide continuous feedback: thinking (agent reasoning), tool steps (what is being fetched or indexed), and content (the actual LLM output).

### P5 — No Local Tmp Files

The system does not use .agent/indexing-tmp or any temporary file cache on the client machine. Tree-sitter parses a file in approximately 5ms. SQLite hash lookup takes approximately 1ms. There is no compute cost worth caching. Temporary files would introduce consistency problems without providing meaningful performance gain.

### P6 — Fail-Safe Hash Store

The SQLite hash store on the client is only updated after a successful upload to the server. If the server upload fails, the hash is not persisted, so the next request will re-attempt the full index. This guarantees that Qdrant and the hash store never go out of sync.

### P7 — Single Qdrant Collection

All languages share one Qdrant collection named codebase. Language filtering is done at query time using a lang payload field. This simplifies operations, supports cross-language search when needed, and avoids collection proliferation.

---

## 3. Component Map

```
DEVELOPER MACHINE
└── Continue IDE Plugin
    └── MCP Server (Python, stdio transport)
        ├── Tool: get_project_skeleton
        ├── Tool: index_with_deps
        ├── Tool: read_file
        └── Tool: search_symbol
            └── Language Plugin Registry
                ├── JavaPlugin
                ├── CSharpPlugin
                ├── GoPlugin
                ├── PythonPlugin
                ├── TypeScriptPlugin
                ├── HCLPlugin
                └── FallbackPlugin
            └── Core Services
                ├── HashStore (SQLite)
                ├── DepClassifier
                ├── TokenBudget
                └── Uploader (HTTPS to VM)

CLOUD VM
└── Nginx (TLS, rate limiting, no SSE buffering)
    └── FastAPI Application
        ├── POST /v1/chat/completions  → SSE stream
        └── POST /index               → chunk ingestion
            └── LangGraph Agent
                ├── Node: classify_intent
                ├── Node: route_context  (5-gate decision)
                ├── Node: rag_search
                ├── Node: plan_steps     (code_gen only)
                ├── Node: generate       (vLLM stream)
                └── Node: post_process
            └── RAG Layer
                ├── Embedder (all-MiniLM-L6-v2 dense + BM25 sparse)
                ├── QdrantClient (hybrid search + RRF fusion)
                └── HashVerifier
            └── Infrastructure
                ├── Qdrant (1 collection: "codebase")
                └── vLLM (Qwen2.5-Coder-7B-Instruct)
```

---

## 4. Data Models

### 4.1 CodeChunk

The universal unit of indexed knowledge. Every language plugin produces CodeChunk objects. No downstream component sees any language-specific structure.

| Field | Type | Description |
|---|---|---|
| chunk_id | string | Deterministic hash of repo_root + relative_path + symbol_name. Stable across reindexes of the same symbol. |
| chunk_type | enum | One of: callable, grouping, dependency, config_block |
| symbol_name | string | Human-readable identifier: class name, function name, or resource address (e.g., aws_s3_bucket.main) |
| embed_text | string | The text used to create the vector: signature + docstring. Never the full body. |
| body | string | Full source text of the symbol. Stored in Qdrant payload, never embedded. |
| file_path | string | Path relative to repo root. |
| lang | string | Language identifier: java, go, python, csharp, typescript, hcl |
| start_line | integer | 1-based line number of the symbol opening. |
| end_line | integer | 1-based line number of the symbol closing. |
| deps | list of strings | Relative paths of project-local dependencies resolved from imports. |
| file_hash | string | MD5 of the containing file at index time. Used for cache invalidation. |
| raw_imports | list of strings | Raw import strings parsed from the file. Used by DepClassifier. Not stored in Qdrant. |

### 4.2 chunk_type Semantics

| Value | Meaning | Java example | Go example | HCL example |
|---|---|---|---|---|
| callable | A unit of executable logic | method, constructor | func, method | provisioner block |
| grouping | A container of callables | class, interface | struct, interface | module block |
| dependency | An import or module reference | import statement | import spec | source attribute |
| config_block | A declarative configuration unit | — | — | resource, data, variable, output |

### 4.3 AgentState

The mutable state object passed between LangGraph nodes.

| Field | Type | Description |
|---|---|---|
| messages | list | Full conversation history in OpenAI message format |
| intent | enum | Classified intent: code_gen, unit_test, explain, structural_analysis, search, refine |
| active_file | string or null | Path of the file currently open in the editor, injected by Continue |
| mentioned_files | list of strings | Files explicitly referenced in the user query |
| freshness_signal | boolean | True if query contains temporal keywords like "hiện tại", "vừa sửa" |
| force_reindex | boolean | True if Gate 1 or Gate 2 was triggered |
| rag_chunks | list of dicts | Retrieved chunks from Qdrant after optional hash verification |
| rag_hit | boolean | Whether Qdrant returned usable results |
| hash_verified | boolean | Whether retrieved chunks passed hash verification |
| tool_results | list of dicts | Results from MCP tool calls during this request |
| context_assembled | string | Final assembled context string passed to the LLM |
| draft | string | LLM output before post-processing |
| emitted_steps | list of strings | Tracker to prevent duplicate SSE step events |

### 4.4 IndexRequest (Client to Server)

Sent by the MCP Uploader to POST /index.

| Field | Type | Description |
|---|---|---|
| chunks | list of ChunkPayload | All chunks to upsert |
| deleted_ids | list of strings | Chunk IDs that must be deleted before upsert (stale data) |

Each ChunkPayload contains all CodeChunk fields except raw_imports.

### 4.5 SSE Event Schema

All events share a type field. Continue uses this field to route rendering.

| type | Additional fields | Continue renders |
|---|---|---|
| thinking | content: string | Small, muted italic line. Auto-collapses when content begins. |
| tool_start | tool, label, detail | Spinner card with label and detail text |
| tool_progress | step, detail, pct | Updates the spinner card inline. pct=-1 means indeterminate. |
| tool_done | tool, summary, ms | Replaces spinner with checkmark. Shows summary and elapsed time. |
| tool_error | tool, error | Red card with error message |
| content | OpenAI delta format | Streamed into the chat bubble token by token |

---

## 5. MCP Server — Client Machine

### Overview

The MCP server runs as a local stdio process on the developer's machine, spawned by Continue when the IDE starts. It is the only component that directly accesses the local filesystem. It exposes four tools to the LLM via the Model Context Protocol.

### Responsibilities

- Receive tool calls from Continue (forwarded from the cloud agent's LLM)
- Detect the appropriate language plugin for any given file
- Parse file ASTs, extract chunks, resolve dependencies
- Check the local SQLite hash store to avoid redundant work
- Upload changed chunks to the cloud server's /index endpoint
- Respond to read and search requests for live file content

### Environment

The MCP server reads its configuration from environment variables injected by Continue: SERVER_URL, API_KEY, REPO_PATH, TOKEN_BUDGET, DEPTH_DEFAULT.

---

### Tool: get_project_skeleton

**Purpose:** Return a compact structural overview of the entire repository in a single call. Used for wide structural queries like "analyze the architecture" or "are there circular dependencies?"

**Input:** include_methods (boolean, default true) — whether to include public method names but not bodies.

**Processing:**
1. Walk the entire REPO_PATH recursively.
2. Skip all directories in the hardcoded blocklist: .git, node_modules, __pycache__, .terraform, vendor, target, build, dist, .venv, venv, .idea, .vscode.
3. Skip files with blocked extensions: .min.js, .lock, .jar, .class, .pyc, .map.
4. For each remaining file, detect the language plugin.
5. Call the plugin's extract_skeleton method with the include_methods flag.
6. Group results by package or namespace path derived from the file's relative location.
7. Accumulate statistics: total class count, total package count.

**Output:** A JSON object with a packages map (package path to list of skeleton entries) and a stats block including a hint field telling the LLM what tools to call next.

**Token budget:** The output must fit within approximately 2000 tokens. Method names are included by default but bodies are never included.

**Important:** This tool does NOT upload anything to the server and does NOT use the hash store. It is a read-only local operation.

---

### Tool: index_with_deps

**Purpose:** Parse a specific file and its project-local dependencies up to a given depth, then upload all changed chunks to the server for embedding and storage.

**Input:** file_path (required string), depth (integer, default 2, maximum 3).

**Processing:**
1. Resolve the absolute path from REPO_PATH + file_path.
2. Detect the language plugin for the target file.
3. Run a BFS traversal starting from the target file. At each node:
   a. Read the file bytes.
   b. Compute MD5 hash and check against the SQLite hash store.
   c. If hash matches the stored value, skip this file entirely.
   d. If hash differs or is absent, call plugin.extract_chunks(path, source, mode) where mode depends on depth: depth 0 uses full_body, depth 1 uses signatures, depth 2 uses names_only.
   e. After extracting chunks, call DepClassifier.classify for each raw import.
   f. If a dep is classified as project and resolves to a local file, add it to the BFS queue at current_depth + 1.
   g. If a dep is classified as stdlib or third_party, discard it.
4. Collect all chunks from visited files not skipped by hash check.
5. If no chunks were collected, return immediately with an "all cached" result.
6. Call the Uploader to POST all collected chunks to SERVER_URL/index.
7. Only after receiving a successful HTTP response, update the SQLite hash store for all visited files.

**Output:** A dict with indexed (count of chunks uploaded), skipped (count of files whose hash matched), and files_processed (total files visited).

**Token budget enforcement:**
- Depth 0 (focal file) is allocated 2000 tokens.
- Depth 1 (direct deps) is allocated 3000 tokens.
- Depth 2 (transitive deps) is allocated 3000 tokens.
- Total maximum: 8000 tokens. If adding chunks from a given depth would exceed the budget, those chunks are truncated or omitted.

---

### Tool: read_file

**Purpose:** Read a contiguous range of lines from a file. Used when the LLM needs exact, guaranteed-fresh content to generate code accurately.

**Input:** file_path (required), start_line (integer, default 1), end_line (integer, default 150).

**Processing:** Resolve the absolute path. Read the full file text. Return lines from start_line to end_line inclusive (1-based).

**Output:** A dict with content (the selected lines as a string), start_line, end_line, total_lines, and file_path.

**Critical constraint:** This tool result must NEVER be uploaded to Qdrant. It is volatile by design — the developer may have unsaved changes. It is consumed once and discarded.

---

### Tool: search_symbol

**Purpose:** Locate a class, function, or method by name anywhere in the repository, returning the file path and line number.

**Input:** name (required string), type_filter (enum: class, function, method, any; default any).

**Processing:**
1. Walk REPO_PATH applying the same skip rules as get_project_skeleton.
2. For each file, detect the plugin and call extract_chunks in names_only mode.
3. Match chunk symbol names against the search term (case-insensitive).
4. Filter by chunk_type if type_filter is not any.

**Output:** A list of match objects, each containing symbol_name, chunk_type, file_path, start_line, lang.

---

### HashStore

The HashStore is a local SQLite database stored at ~/.ai-agent/hash_store.db. It contains one table with columns: file_path (text primary key), md5_hash (text), indexed_at (real — Unix timestamp).

**Invariant:** A record is only written after the server confirms successful ingestion. A record is never written on upload failure. This guarantees that a file whose hash is stored in SQLite is definitely present and current in Qdrant.

---

## 6. Language Plugin System

### Interface Contract

Every language plugin implements exactly three required methods and one optional method.

**extensions() returns list of strings.**
Returns the file extensions this plugin handles. Example: [".java"] for Java, [".tf", ".hcl"] for Terraform.

**extract_chunks(path, source_bytes, mode) returns list of CodeChunk.**
Parses the file using a Tree-sitter grammar. The mode parameter controls how much information is extracted:
- full_body: Extracts signature, docstring, AND complete source body.
- signatures: Extracts signature and docstring only; body field is empty.
- names_only: Extracts symbol name only; all other text fields are empty.

**resolve_dep_path(import_str, from_file, repo_root) returns Path or None.**
Given a raw import string, attempts to resolve it to a local file path relative to the repo root. Returns None if the import cannot be resolved to a local project file.

**extract_skeleton(path, source_bytes, include_methods) returns dict.**
Optional override. Default implementation calls extract_chunks in names_only mode and assembles the skeleton dict. Override when the language's AST structure requires custom logic for grouping symbols.

### Plugin Registry

The registry maintains a dictionary mapping file extensions to plugin instances. It is built at startup by iterating all installed plugins and calling extensions() on each. On get_plugin(path):
1. Look up by file extension (lowercase).
2. If not found, inspect the shebang line (first line of file) for python, node, or deno.
3. If still not found, return the FallbackPlugin.

### DepClassifier

The DepClassifier uses a two-step classification logic for each import string.

**Step 1 — Attempt resolution:** Call plugin.resolve_dep_path(import_str, from_file, repo_root). If it returns a path and the path exists on disk, classify as project and return the path.

**Step 2 — Check stdlib list:** Compare the import string against a hardcoded per-language list of known stdlib prefixes. If it matches, classify as stdlib.

**Default:** If neither step matches, classify as third_party.

Only project deps are fed back into the BFS indexing queue. stdlib and third_party are discarded entirely.

### Per-Language Resolution Logic

**Java and C#:** Map the fully-qualified class name to a file path by converting dots to directory separators. Search under src/main/java (Java) or the project root (C#). A dep is project if its package prefix matches the project's group ID from pom.xml or the root namespace from .csproj.

**Go:** Read the go.mod file to find the module path. If the import starts with that module path, the remainder is a relative path under the repo root. The first .go file in that directory is returned.

**Python:** Handle relative imports directly (.module or ..module). For absolute imports, search for a matching file or directory under src/ or the repo root. A dep is project if it resolves to an existing local file.

**TypeScript/JavaScript:** If the import string starts with ./ or ../, it is always project. Resolve relative to the importing file, trying .ts, .tsx, .js, .jsx, and index.ts extensions. If it does not start with ., it is third_party unless it starts with node:, which makes it stdlib.

**HCL (Terraform):** The source attribute within a module block is the import string. If it starts with ./ or ../, it is a local module (project). Otherwise it is third_party. Provider references (aws, google, azurerm) are treated as stdlib.

**Fallback:** No dep resolution. Chunks are extracted via regex patterns matching common function and class declaration syntax. The body is chunked by paragraph if structured extraction fails.

### Stdlib Prefix Lists

These are the authoritative stdlib prefix lists used by DepClassifier. They must not be reduced.

- **Java:** java., javax., sun., com.sun.
- **C#:** System., Microsoft., Windows.
- **Go:** fmt, os, io, net, strings, strconv, sync, context, errors, math, sort, time, encoding, crypto, path, bufio, bytes, runtime, reflect, log
- **Python:** os, sys, re, json, pathlib, typing, collections, itertools, functools, abc, datetime, math, random, hashlib, logging, asyncio, dataclasses, enum, copy, io
- **TypeScript:** node:fs, node:path, node:http, node:os, node:crypto, node:events, node:stream

### HCL-Specific chunk_type Mapping

Because Terraform does not have traditional callables or classes, the HCL plugin maps its constructs to the universal schema as follows:

| HCL construct | chunk_type | symbol_name format |
|---|---|---|
| resource "type" "name" | config_block | type.name (e.g., aws_s3_bucket.main) |
| data "type" "name" | config_block | data.type.name |
| variable "name" | config_block | var.name |
| output "name" | config_block | output.name |
| module "name" | grouping | module.name |
| provider "name" | config_block | provider.name |

---

## 7. Cloud VM — FastAPI Server

### Overview

The FastAPI server runs on the Cloud VM behind Nginx. It receives all requests from Continue and serves two endpoints: the streaming chat endpoint and the chunk ingestion endpoint.

### POST /v1/chat/completions

**Authentication:** Requires X-Api-Key header matching the server's configured API_KEY. Return HTTP 403 if missing or incorrect.

**Request body:** OpenAI-compatible chat completions request format, plus two additional custom fields: active_file (path of the file currently open in the IDE, may be null) and repo_path (the repo root path, may be null; used for context only).

**Response:** text/event-stream SSE stream. The stream field in the request body is ignored — this endpoint always streams.

**Nginx requirements for this endpoint:** proxy_buffering off, proxy_cache off, proxy_read_timeout 300s, X-Accel-Buffering: no header set in the response.

### POST /index

**Authentication:** Requires X-Api-Key header.

**Request body:** IndexRequest as defined in Section 4.4.

**Processing:**
1. Validate API key.
2. If deleted_ids is non-empty, delete those point IDs from Qdrant before any upserts.
3. For each chunk, call the Embedder to produce dense and sparse vectors from chunk.embed_text, then call Qdrant upsert with both vectors and the full chunk payload.
4. Return {"indexed": N, "deleted": M}.

**Idempotency:** Because chunk_id is deterministic, upserting the same chunk twice is safe. Qdrant will overwrite the existing point.

---

## 8. LangGraph Agent

### Graph Structure

The agent is a directed graph with conditional edges. Nodes are async Python functions that receive and return AgentState.

```
Entry → classify_intent → route_context
                              │
              ┌───────────────┼────────────────┐
              │ structural_analysis             │ all other intents
              ▼                                 ▼
           generate                        rag_search
                                               │
                              ┌────────────────┼───────────────┐
                              │ code_gen                        │ all other intents
                              ▼                                 ▼
                          plan_steps                         generate
                              │                                 │
                              └─────────────────┬───────────────┘
                                                ▼
                                           post_process → END
```

### Node: classify_intent

**Input:** Reads messages from AgentState.

**Logic:** Extracts the last user message. Applies a priority-ordered list of keyword and regex patterns. The classification order must be preserved because some patterns overlap.

**Intent priority order:**
1. unit_test — matches: "viết test", "write test", "unit test", "generate test", "tạo test"
2. structural_analysis — matches: "phân tích cấu trúc", "analyze architecture", "circular dep", "overview", "project structure", "toàn bộ project"
3. explain — matches: "giải thích", "explain", "how does", "hoạt động như thế nào", "làm gì", "what does"
4. search — matches: "tìm", "find", "search", "where is", "ở đâu"
5. refine — matches: "sửa lại", "refactor", "improve", "fix", "optimize", "cải thiện"
6. code_gen — default if none of the above match

**Output:** Sets state.intent.

### Node: route_context

**Input:** Reads messages and active_file from AgentState.

**Logic:** Implements the 5-Gate decision flow described in Section 11.

**Output:** Sets state.mentioned_files, state.force_reindex, state.freshness_signal.

### Node: rag_search

**Input:** Reads messages, active_file, freshness_signal, mentioned_files from AgentState.

**Logic:**
1. Extract the last user message as the search query.
2. Produce a dense embedding and a sparse BM25 vector from the query text.
3. Determine language filter from the active_file extension.
4. Call Qdrant hybrid search with dense vector, sparse vector, optional language filter, and top_k=8.
5. If no results, set rag_hit=False and rag_chunks=[].
6. If freshness_signal is true or mentioned_files is non-empty, perform hash verification: for each returned chunk, read the file_hash from the payload, compute MD5(current file bytes), keep only chunks where hashes match. Chunks with mismatched hashes are silently discarded.
7. Set rag_hit, rag_chunks, hash_verified accordingly.

**Output:** Sets state.rag_chunks, state.rag_hit, state.hash_verified.

### Node: plan_steps

**Input:** Reads messages, rag_chunks, active_file from AgentState.

**When used:** Only invoked for code_gen intent.

**Logic:** Calls the LLM (vLLM) with a planning-specific system prompt. The prompt instructs the LLM to output a structured JSON plan describing which files need to be read or modified, what each modification step is, and what the expected output should look like. The plan is stored in state.tool_results.

**Max LLM call budget:** 800ms. Use a low max_tokens limit (approximately 400 tokens).

**Output:** Appends the plan to state.tool_results.

### Node: generate

**Input:** Reads all state fields.

**Logic:**
1. Assemble the system prompt: base instruction, then RAG context formatted as labeled code blocks, then the structured plan for code_gen.
2. Append the conversation history as user/assistant messages.
3. Call vLLM via the OpenAI-compatible API with stream=True.
4. Emit each token as a content SSE delta event immediately.
5. Store the complete output in state.draft.

**Output:** Sets state.draft; streams tokens via SSE.

### Node: post_process

**Input:** Reads intent and draft from AgentState.

**Logic:** Applies intent-specific validation rules. For unit_test, verify the output contains a test function declaration. For code_gen, verify syntactic plausibility (presence of opening and closing blocks). For all intents, strip accidentally leaked system prompt text. If validation fails, the draft is returned as-is but flagged. No automatic retry in V1.

**Output:** Finalizes state.draft.

---

## 9. RAG and Cache Strategy

### Qdrant Collection Schema

Collection name: codebase

Vectors:
- dense: 384-dimensional float vector, cosine distance (all-MiniLM-L6-v2 output)
- sparse: Sparse vector with BM25-derived term weights

Payload fields (all stored, lang and chunk_type indexed for filtering): str_id, symbol_name, chunk_type, file_path, lang, start_line, end_line, file_hash, body, deps.

### Embedding Strategy

**Dense embedding:** Use all-MiniLM-L6-v2 from sentence-transformers. Input is embed_text (signature + docstring). Output is a 384-dimensional L2-normalized vector.

**Sparse embedding:** Use BM25 term frequency weighting on the tokenized embed_text. Output is a sparse vector with term indices and BM25 scores as values. Term indices are computed as hash(term) modulo 2^20.

**Critical distinction:** Only embed_text is embedded. The body field is stored as payload but never embedded. This keeps vectors semantically precise.

### Hybrid Search and RRF Fusion

1. Run dense vector search with the query's dense embedding — retrieve top 2*top_k results.
2. Run sparse vector search with the query's BM25 vector — retrieve top 2*top_k results.
3. Apply Reciprocal Rank Fusion with k=60: each result's RRF score is 1 / (k + rank + 1). Sum scores across both result lists for each unique chunk ID.
4. Sort by combined RRF score descending, return top top_k results.

The default top_k is 8.

### Data Volatility Classification and Cache Behavior

| Data type | Volatility | Cache strategy |
|---|---|---|
| Method signatures, docstrings | Low | RAG with hash verification on freshness signal |
| Project skeleton (class/method names) | Low-medium | RAG, hash check on mention |
| Method bodies | Medium | RAG, always hash-verify when force_reindex is active |
| Full file content | High | Never cached. Always use read_file() tool |
| Git diff, runtime logs | Real-time | Never cached. Live tool required (out of V1 scope) |
| README, docs, ADRs | Very low | RAG, long-lived, manual trigger to refresh |

### Write Policy for Tool Results

- Store: Results from index_with_deps() — these are structured chunks designed for RAG.
- Do not store: Results from read_file() — volatile by definition.
- Do not store: Results from search_symbol() — lookup results, not knowledge units.
- Do not store: Results from get_project_skeleton() — derived structural metadata, not source chunks.

### Stale-While-Revalidate for force_reindex

When force_reindex is true and the file has existing (but potentially stale) chunks in Qdrant:
1. Return the stale chunks immediately for the current request so generation is not blocked.
2. In the background (after the response is sent), trigger a fresh index_with_deps() call.
3. Future requests will hit the refreshed cache.

When force_reindex is true and the file has NO existing chunks in Qdrant:
1. Indexing must complete before generation begins — there is no fallback.
2. Stream tool_start and tool_progress events during this wait to avoid blank screen.

---

## 10. SSE Streaming and UX

### Three-Layer Event Model

**Layer 1 — Thinking events (type: "thinking"):**
Purpose is continuous proof of life. Rendering is small muted italic text at approximately 50% opacity. Auto-collapses when the first content event arrives. Emit at every significant decision point in the agent graph. Examples: "Phân tích intent...", "Kiểm tra cache...", "Tìm thấy 6 chunks trong RAG".

**Layer 2 — Tool step events (type: "tool_start", "tool_progress", "tool_done", "tool_error"):**
Purpose is to show exactly what is happening during MCP tool execution. Rendering is a spinner card for tool_start/tool_progress, replaced by a checkmark for tool_done, and a red card for tool_error. The pct field in tool_progress is 0-100 or -1 for indeterminate. When content events begin, all completed tool cards collapse to a single summary line.

**Layer 3 — Content events (type: "content"):**
Purpose is the actual LLM output. Format is OpenAI delta format. The terminal event is the raw string "data: [DONE]" signaling stream completion.

### Timing Constraints

| Constraint | Value |
|---|---|
| First thinking event | Within 50ms of request receipt |
| First tool_start event | Within 100ms of deciding to call a tool |
| First content token | Within 200ms of LLM beginning generation |
| Maximum blank screen time | 500ms (must never be exceeded) |
| SSE heartbeat (if no events) | Every 15 seconds to prevent proxy timeout |

### SSE Heartbeat

If no event has been emitted for 15 seconds (can happen during long LLM generation), emit a comment line: ": keep-alive". This prevents Nginx and proxy layers from closing the connection.

---

## 11. Context Decision Flow (5-Gate)

This flow runs in the route_context node. Gates are evaluated in strict order. The first gate that fires determines the strategy for the entire request.

### Gate 1 — Explicit File Mention (Highest Priority)

**Trigger condition:** The user's query contains a file reference detected from: direct filename patterns (XxxService.java, UserService, handler.go, main.tf), explicit @mention syntax (@UserService.java), or deictic reference to the active file when query contains "file này", "class này", "this file", "this class", "đây", "nó", "it", "here" AND active_file is set.

**Action:** Set force_reindex=True and populate mentioned_files with all detected file paths.

**Does NOT fire for:** Generic queries with no file reference.

### Gate 2 — Freshness Force Signal

**Trigger condition:** The query contains temporal keywords signaling the user is asking about the current state: "vừa thêm", "vừa sửa", "vừa commit", "just added", "just changed", "hiện tại đang có bug", "lỗi đang xảy ra", "recent change".

**Action:** Set force_reindex=True and freshness_signal=True.

### Gate 3 — Volatile Data Type

**Trigger condition:** The query requests data that is inherently real-time and never appropriate for RAG: git diff, runtime logs, live metrics, error stack traces from a running process.

**Action:** Return a "not supported" message. No live tools are implemented in V1.

### Gate 4 — Qdrant RAG Lookup

**Trigger condition:** None of gates 1-3 fired.

**Action:** Proceed to the rag_search node. If Qdrant misses, call index_with_deps() to fetch and index the relevant file, then retry the search once. Store new chunks in Qdrant before generating.

### Gate 5 — Hash Verification

**Trigger condition:** Gate 4 produced Qdrant results AND (freshness_signal is true OR mentioned_files is non-empty).

**Action:** For each returned chunk, verify the stored file_hash matches the current MD5 of the local file. Remove stale chunks. If all chunks are stale, escalate to force_reindex.

---

## 12. Integration Points and API Contracts

### Continue to MCP Server

Protocol is MCP over stdio. Continue spawns the MCP server process using command and args from config.yaml. Continue injects the tool list into the LLM context. The LLM emits a tool call in JSON. Continue parses it, calls the local MCP server, receives the result, and injects it back into the conversation. Continue sends the currently open file path as active_file in the request body to /v1/chat/completions.

### MCP Server to Cloud VM

Endpoint: POST /index. Authentication via X-Api-Key header. Payload size limit: 2MB enforced by Nginx. Retry policy: 3 retries with exponential backoff starting at 500ms on 5xx errors. If all retries fail, the hash store is NOT updated. Timeout: 30 seconds per request.

### Continue to Cloud VM

Endpoint: POST /v1/chat/completions. Authentication via X-Api-Key header. Response format: text/event-stream with custom event types. All content events must use OpenAI delta format for backwards compatibility. Custom event types must not break Continue if it ignores unknown types.

### Cloud VM to vLLM

Endpoint: vLLM's OpenAI-compatible API at VLLM_BASE_URL/v1/chat/completions. Model identifier: qwen2.5-coder. Always stream=True. max_tokens: 4096 for generation, 400 for planning calls.

### Cloud VM to Qdrant

Client: qdrant-client async Python client. Connection via QDRANT_URL. Qdrant must be bound to 127.0.0.1 only, never exposed publicly. Access only from the FastAPI process on the same host.

---

## 13. Design Decisions and Rationale

### Decision 1: No .agent/indexing-tmp cache files

Tree-sitter parses a file in approximately 5ms. SQLite hash lookup costs approximately 1ms. Temporary files would introduce a three-way consistency problem (filesystem, tmp cache, Qdrant) instead of a two-way one. The cost-benefit analysis strongly favors simplicity.

### Decision 2: Force reindex on explicit file mention (Gate 1)

This is not about data freshness — it is about developer intent. When a developer types "explain UserService.go", they are looking at that file in their IDE and expect the agent to see what they see. A hash match from 2 hours ago does not represent the developer's current context. Gate 1 is an intent signal, not a freshness heuristic.

### Decision 3: Never cache read_file() results in Qdrant

File content changes with every save. A developer may have unsaved changes. Caching file content would guarantee staleness within minutes. The tool is explicitly designed as a volatile, single-use fetch.

### Decision 4: Index only project-local dependencies; discard stdlib and third-party

The LLM already has extensive knowledge of Spring, Django, Express, Gin, .NET BCL, and other common frameworks. Indexing them wastes token budget, degrades vector search precision, and adds indexing latency. The DepClassifier is the most critical correctness component in the system — its stdlib lists must be accurate and complete.

### Decision 5: Token budget with depth-based modes

The Qwen2.5-Coder-7B context window is 32K tokens. After reserving space for system prompt (~500), conversation history (~4000), output buffer (~4000), and headroom, approximately 23K tokens remain for RAG context. An 8K budget for dependency context is generous while still leaving room for multi-turn conversations. Deeper dependencies need less detail — the LLM needs to know they exist (names_only) but does not need their implementations.

### Decision 6: Language detection isolated to plugin layer

This allows adding a new language (e.g., Rust, Ruby) by writing exactly one new plugin file and adding one line to the registry. The RAG system, agent graph, Qdrant schema, and streaming layer require zero changes. This is the core extensibility guarantee of the design.

### Decision 7: Single Qdrant collection for all languages

Multiple collections would require per-collection management and coordination logic when a query spans languages (e.g., a Java service calling a Go gRPC service). A single collection with language filtering at query time is simpler, supports cross-language search, and reduces operational overhead.

### Decision 8: Planner node only for code_gen intent

Code_gen tasks are frequently multi-file and multi-step. Without a planning step, the LLM loses coherence midway through generation. Unit_test, explain, and search are inherently single-output tasks — planning overhead would add latency without benefit.

---

## 14. Error Handling and Edge Cases

### MCP Server Errors

**File not found:** read_file() and index_with_deps() must return a structured error dict {"error": "File not found: <path>"} rather than raising an exception.

**Parser failure:** If Tree-sitter fails to parse a file, the plugin must catch the exception and fall back to FallbackPlugin behavior for that file. Partially parsed chunks are never uploaded.

**Upload timeout:** If the HTTP POST to /index times out after 30 seconds, the Uploader returns a failure result. The hash store is not updated. The tool returns {"indexed": 0, "error": "Upload timeout"}.

### Agent Errors

**Qdrant unavailable:** Set rag_hit=False and proceed to generation without RAG context. Emit a thinking event noting "RAG unavailable, proceeding without context."

**RAG returns zero results and MCP indexing fails:** Generate without context. The LLM will produce a generic answer. This is preferable to a failure response.

**vLLM timeout or error:** Emit a tool_error event with the error message. Return HTTP 200 (the stream has already started) with the error embedded in the SSE stream. Never return HTTP 500 after streaming begins.

### Edge Cases

**File mention resolves to a file outside REPO_PATH:** Reject silently. Do not read or index files outside the configured repo root.

**Circular imports:** The BFS traversal uses a visited set keyed on absolute file path. Circular imports cause the second visit to be skipped, breaking the cycle.

**Very large files (>500KB):** Do not skip them when the user explicitly mentions the file. Instead, limit chunk extraction to the first 200 functions/classes and log a warning.

**Monorepo with multiple languages:** Fully supported. The plugin registry handles mixed-language directories. The lang filter in Qdrant search is set based on the active_file extension by default.

**User query in English vs Vietnamese:** The intent classifier patterns must include both languages as shown in Section 8. The LLM is multilingual — prompt assembly should match the user's message language.

**Concurrent requests from multiple developers:** Each request is stateless. Qdrant upserts use the deterministic chunk_id based on repo+path+symbol. Two developers indexing the same file will produce the same chunk IDs and the last write wins, which is acceptable.

---

## 15. Implementation Phases

### Phase 1 — MCP Server Foundation

Goal: A working MCP server that Continue can connect to and use for basic file operations. No indexing, no Qdrant yet.

Scope:
- Create the mcp_server/ package with pyproject.toml and dependency declarations.
- Implement the CodeChunk dataclass with all fields.
- Implement the HashStore class backed by SQLite at ~/.ai-agent/hash_store.db.
- Implement the FallbackPlugin using regex-based extraction (no Tree-sitter required in this phase).
- Implement the read_file() tool: read lines from a file, return structured response.
- Implement the search_symbol() tool: grep-based scan using the FallbackPlugin.
- Implement server.py with MCP tool declarations and routing.

Deliverables:
- MCP server process starts without errors.
- Continue config template showing how to add the server.
- read_file() returns correct content for any file in the repo.
- search_symbol() returns matching file paths and line numbers.

---

### Phase 2 — Language Plugins

Goal: Full AST-based parsing for all six target languages plus the fallback.

Scope:
- Install Tree-sitter and all six language grammar packages.
- Implement the LanguagePlugin abstract base class (finalize CodeChunk).
- Implement GoPlugin with full AST traversal for structs, interfaces, functions, methods; import resolution via go.mod.
- Implement JavaPlugin with AST traversal for classes, interfaces, methods, annotations; import resolution via package-to-path mapping.
- Implement PythonPlugin with AST traversal for classes, functions, async functions; relative and absolute import resolution.
- Implement TypeScriptPlugin with AST traversal for classes, functions, arrow functions, interfaces; resolution by ./ ../ prefix detection.
- Implement CSharpPlugin with AST traversal for classes, interfaces, methods, properties; namespace-to-path resolution.
- Implement HCLPlugin with AST traversal for all block types; config_block chunk_type; local module resolution.
- Implement FallbackPlugin with regex-based function/class detection and paragraph chunking.
- Implement the PluginRegistry with extension mapping and shebang detection.
- Implement the DepClassifier with all stdlib prefix lists.

Deliverables:
- Each plugin correctly extracts chunks from representative sample files.
- Dep resolution correctly classifies project vs stdlib vs third_party for each language.
- extract_skeleton() produces compact output fitting within 2000 tokens for a typical project.

---

### Phase 3 — Indexer and Skeleton Tools

Goal: The MCP server can index files and upload chunks to the server.

Scope:
- Implement the TokenBudget module that enforces depth-based mode selection and total token limits.
- Implement the Uploader class with async HTTP POST, retry logic, and error handling.
- Implement index_with_deps() tool with BFS dep traversal, hash checking, mode selection, and fail-safe hash store writes.
- Implement get_project_skeleton() tool with directory walking and package grouping.
- Connect all tools to server.py.

Deliverables:
- get_project_skeleton() returns a complete repo overview within 2000 tokens for a 50-file project.
- index_with_deps() correctly identifies changed files, skips unchanged ones, and uploads the right chunks with the right modes.
- Hash store is updated only after successful upload.
- Circular import cycles do not cause infinite loops.

---

### Phase 4 — Cloud VM RAG Infrastructure

Goal: The server can accept chunks, embed them, store them in Qdrant, and search them.

Scope:
- Set up Qdrant on the Cloud VM, bound to 127.0.0.1:6333.
- Implement the QdrantClient class with collection creation, upsert, hybrid search, delete, and RRF fusion.
- Implement the Embedder class producing dense (all-MiniLM-L6-v2) and sparse (BM25) vectors.
- Implement the /index endpoint with authentication, deletion, embedding, and upsert.
- Implement the FastAPI app skeleton with lifespan handler, CORS, and health check.

Deliverables:
- Qdrant collection created on startup if not present.
- Upload 50 test chunks and retrieve them via hybrid search with correct top-k results.
- Language filter correctly restricts results to the specified language.
- RRF fusion improves result quality over either dense or sparse alone.

---

### Phase 5 — LangGraph Agent

Goal: The agent correctly classifies intent, routes context, retrieves from RAG, and generates a response.

Scope:
- Implement AgentState TypedDict.
- Implement classify_intent node with all patterns for both Vietnamese and English.
- Implement route_context node with all 5 gates and file mention detection.
- Implement rag_search node with hash verification.
- Implement generate node with context assembly and vLLM streaming.
- Implement post_process node with intent-specific validation.
- Implement plan_steps node for code_gen intent.
- Wire the LangGraph graph with conditional edges.

Deliverables:
- Intent classification is correct for all test query variants.
- Gate 1 fires when a file name is mentioned; does not fire for generic queries.
- Gate 2 fires for freshness keywords; does not fire for neutral queries.
- RAG search returns relevant results for code-related queries.
- End-to-end query from a simulated Continue request returns a streamed response.

---

### Phase 6 — SSE Streaming Integration

Goal: Continue displays the three-layer streaming experience with no blank screens.

Scope:
- Implement all SSE event helpers in server/streaming/sse.py.
- Implement the /v1/chat/completions endpoint with the full streaming generator.
- Wire agent graph execution to emit thinking, tool, and content events at the correct points.
- Configure Nginx: proxy_buffering off, proxy_cache off, 300s read timeout, rate limiting.
- Implement the SSE heartbeat (": keep-alive" every 15 seconds).

Deliverables:
- First thinking event arrives in Continue within 50ms.
- Tool step cards appear and update in real time during indexing.
- Content tokens stream visibly as the LLM generates.
- Tool cards collapse when content begins.
- No blank screen exceeds 500ms under normal conditions.
- Stream closes cleanly with [DONE].

---

### Phase 7 — Hardening and Production Readiness

Goal: The system is reliable, observable, and ready for a small team.

Scope:
- Authentication middleware with constant-time API key comparison.
- Request-scoped logging with correlation ID in all log lines.
- Structured error responses — all errors return {"error": "...", "code": "..."} and never raw tracebacks.
- Graceful shutdown handling in FastAPI.
- Test the full stack with a mixed-language project (e.g., Java service + Terraform infrastructure).
- Load test with 5 concurrent developer sessions.
- Document the setup steps for a new developer machine.

Deliverables:
- All endpoints return appropriate HTTP status codes for all error conditions.
- Log output is structured JSON and includes correlation IDs.
- 5 concurrent requests do not cause data corruption or out-of-order SSE events.
- A new developer can set up and use the agent within 15 minutes following the documented steps.

---

## 16. Acceptance Criteria per Phase

### Phase 1
- Continue recognizes the MCP server without errors in its logs.
- read_file("src/Main.java", 1, 50) returns exactly lines 1-50 of the file.
- search_symbol("UserService") returns the correct file path and line number.
- MCP server restarts cleanly if the process is killed.

### Phase 2
- GoPlugin.extract_chunks on a file with 3 structs and 5 functions returns exactly 8 chunks.
- JavaPlugin.extract_chunks in signatures mode returns chunks with empty body fields.
- DepClassifier classifies "java.util.List" as stdlib, "org.springframework.stereotype.Service" as third_party, and "com.mycompany.order.OrderRepository" as project.
- HCLPlugin produces config_block chunks for resource and variable blocks.
- FallbackPlugin produces at least one chunk for any file regardless of extension.

### Phase 3
- get_project_skeleton() on a 50-file Java project returns output with fewer than 2000 tokens.
- index_with_deps("UserService.java") uploads chunks for the focal file (depth 0, full_body) and its direct project deps (depth 1, signatures).
- After indexing, running the same command with no file changes produces {"indexed": 0, "skipped": N}.
- Modifying a file and running again produces {"indexed": M, "skipped": N} where M covers only the changed file.

### Phase 4
- Uploading 100 chunks to /index completes in under 5 seconds.
- Searching for "create order" returns a chunk whose symbol_name contains "Order" or "Create" in the top-3 results.
- Language filter lang="go" returns no Java chunks from a mixed-language index.

### Phase 5
- "viết unit test cho UserService" classifies as unit_test.
- "phân tích cấu trúc project này" classifies as structural_analysis.
- "UserService.java" in query triggers Gate 1 (force_reindex=True).
- "vừa thêm PaymentService" triggers Gate 2 (force_reindex=True).
- "how does dependency injection work" triggers neither Gate 1 nor Gate 2.

### Phase 6
- First event arrives within 50ms measured from request receipt to first byte of response.
- Tool card appears within 100ms of an MCP tool being invoked.
- Content tokens begin streaming within 200ms of LLM generation starting.
- After streaming 500 tokens, the tool cards have collapsed to a summary line.
- Simulated 15-second delay between tokens does not cause connection drop.

### Phase 7
- Invalid API key returns HTTP 403 within 50ms.
- Qdrant connection failure falls back gracefully with a thinking event warning.
- 5 concurrent requests complete without errors or interleaved SSE events.
- A new developer follows setup docs and makes their first successful query within 15 minutes.

---

## 17. Configuration and Environment Variables

### MCP Server (set in Continue's mcpServers[].env)

| Variable | Required | Description |
|---|---|---|
| SERVER_URL | Yes | Base URL of the Cloud VM API, e.g., https://myvm.example.com |
| API_KEY | Yes | Shared secret for API authentication |
| REPO_PATH | Yes | Absolute path to the project root on the developer's machine |
| TOKEN_BUDGET | No | Total token budget for dep context. Default: 8000 |
| DEPTH_DEFAULT | No | Default BFS depth for index_with_deps. Default: 2 |

### Cloud VM Server (set as environment variables or in a .env file)

| Variable | Required | Description |
|---|---|---|
| API_KEY | Yes | Shared secret matching all client MCP servers |
| QDRANT_URL | Yes | Qdrant connection URL, e.g., http://127.0.0.1:6333 |
| VLLM_BASE_URL | Yes | vLLM OpenAI-compatible API base URL, e.g., http://127.0.0.1:8080/v1 |
| HOST | No | Bind address for FastAPI. Default: 0.0.0.0 |
| PORT | No | Port for FastAPI. Default: 8000 |
| LOG_LEVEL | No | Logging level. Default: INFO |

---

## 18. Dependency List

### MCP Server (client machine)

| Package | Minimum Version | Purpose |
|---|---|---|
| mcp | 1.0 | Model Context Protocol server framework |
| tree-sitter | 0.23 | Language-agnostic AST parsing engine |
| tree-sitter-java | latest | Java grammar |
| tree-sitter-go | latest | Go grammar |
| tree-sitter-python | latest | Python grammar |
| tree-sitter-c-sharp | latest | C# grammar |
| tree-sitter-javascript | latest | TypeScript/JavaScript grammar |
| tree-sitter-hcl | latest | Terraform HCL grammar |
| httpx | 0.27 | Async HTTP client for Uploader |

Python 3.11 or higher required.

### Cloud VM Server

| Package | Minimum Version | Purpose |
|---|---|---|
| fastapi | 0.115 | Async web framework |
| uvicorn[standard] | 0.32 | ASGI server |
| langgraph | 0.2 | Agent graph orchestration |
| langchain-core | 0.3 | Message types and base abstractions |
| qdrant-client[async] | 1.12 | Qdrant async Python client |
| sentence-transformers | 3.3 | all-MiniLM-L6-v2 dense embedding model |
| rank-bm25 | 0.2 | BM25 sparse vector computation |
| openai | 1.54 | OpenAI-compatible client for vLLM |

Python 3.11 or higher required.

### Infrastructure (Cloud VM)

| Component | Notes |
|---|---|
| Qdrant | Run via Docker. Bind to 127.0.0.1:6333. Persist data to a named volume. |
| vLLM | Serve Qwen2.5-Coder-7B-Instruct. OpenAI-compatible API. Internal only. |
| Nginx | TLS termination. proxy_buffering off for SSE endpoints. Rate limiting. 2MB request size limit. |

---

*This document fully reflects the design decisions made during the architecture session. Every section is implementation-ready. Phase sequence must be followed. Section 13 must be read before modifying any architectural decision.*