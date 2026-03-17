"""
Retrieve node — fetches RAG context + optional ContextBuilder enrichment.

Wraps:
  - rag/client.py  → RAGClient.search_by_class()
  - context/       → ContextBuilder.build_context() (optional)
"""

from __future__ import annotations

import re
from typing import Optional

import structlog

logger = structlog.get_logger()


async def retrieve_node(state: dict, *, rag_client, context_builder=None, cache_service=None) -> dict:
    """Fetch RAG context and optionally enrich with ContextBuilder.

    Args:
        state: UnitTestState dict.
        rag_client: RAGClient instance.
        context_builder: Optional ContextBuilder instance.
        cache_service: Optional CacheService for Redis-backed RAG cache.

    Returns:
        State updates: rag_chunks, context_result, class_name.
    """
    file_path = state.get("file_path", "")
    class_name = state.get("class_name") or _extract_class_name(file_path)
    collection_name = state.get("collection_name")

    if not class_name:
        logger.error("retrieve_node: no class name", file_path=file_path)
        return {
            "error": "Could not determine class name from file path",
            "rag_chunks": [],
            "context_result": {},
            "class_name": "",
        }

    logger.info("retrieve_node: starting", class_name=class_name, file_path=file_path)

    rag_chunks = []
    context_result = {}

    # Try ContextBuilder first (richer context pipeline)
    if context_builder:
        # Dynamic re-initialization if repo_path changed
        current_repo = state.get("repo_path")
        if current_repo and current_repo != context_builder.repo_path:
            logger.info(
                "retrieve_node: repo_path changed, re-initializing intelligence",
                old=context_builder.repo_path,
                new=current_repo,
            )
            context_builder.init_intelligence(current_repo)

        try:
            ctx_result = await context_builder.abuild_context(
                class_name=class_name,
                file_path=file_path,
                top_k=10,
                collection_name=collection_name,
            )
            rag_chunks = ctx_result.rag_chunks
            context_result = {
                "snippets": len(ctx_result.snippets),
                "intelligence_available": ctx_result.intelligence_available,
                "mock_types": ctx_result.mock_types,
                "elapsed_ms": round(ctx_result.elapsed_ms, 1),
            }
            logger.info(
                "retrieve_node: ContextBuilder OK",
                class_name=class_name,
                chunks=len(rag_chunks),
            )
        except Exception as e:
            logger.warning("retrieve_node: ContextBuilder failed, falling back to RAG", error=str(e))
            rag_chunks = await _get_rag_context_async(rag_client, class_name, file_path, collection_name, cache_service)
    else:
        # RAG-only path
        rag_chunks = await _get_rag_context_async(rag_client, class_name, file_path, collection_name, cache_service)

    # Serialize chunks for state (must be JSON-serializable)
    serialized_chunks = []
    for chunk in rag_chunks:
        if hasattr(chunk, "model_dump"):
            serialized_chunks.append(chunk.model_dump())
        elif hasattr(chunk, "__dict__"):
            serialized_chunks.append(chunk.__dict__)
        else:
            serialized_chunks.append(chunk)

    logger.info("retrieve_node: done", class_name=class_name, chunks=len(serialized_chunks))

    return {
        "rag_chunks": serialized_chunks,
        "context_result": context_result,
        "class_name": class_name,
    }


async def _get_rag_context_async(rag_client, class_name: str, file_path: str,
                           collection_name: Optional[str], cache_service=None) -> list:
    """Async variant of _get_rag_context."""
    # Check Redis cache first (assuming cache_service is sync for now)
    cache_key = f"{class_name}:{file_path}"
    if cache_service:
        try:
            cached = cache_service.get_rag_context(cache_key)
            if cached:
                logger.debug("retrieve_node: using cached RAG context", class_name=class_name)
                from rag.schema import CodeChunk
                return [CodeChunk(**c) for c in cached]
        except Exception:
            pass

    try:
        result = await rag_client.asearch_by_class(
            class_name=class_name,
            top_k=10,
            include_dependencies=True,
            collection_name=collection_name,
        )
        chunks = result.chunks
        
        # Cache in Redis
        if cache_service and chunks:
            try:
                serialized = [c.model_dump() if hasattr(c, "model_dump") else c.__dict__ for c in chunks]
                cache_service.cache_rag_context(cache_key, serialized, ttl=3600)
            except Exception:
                pass
        return chunks
    except Exception as e:
        logger.error("Async RAG search failed", class_name=class_name, error=str(e))
        return []


def _extract_class_name(file_path: str) -> Optional[str]:
    """Extract class name from Java file path."""
    if not file_path:
        return None
    # E.g., "src/main/java/com/example/UserService.java" → "UserService"
    name = file_path.replace("\\", "/").rsplit("/", 1)[-1]
    if name.endswith(".java"):
        return name[:-5]
    return name

