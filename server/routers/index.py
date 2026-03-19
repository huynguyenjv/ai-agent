"""
Index management endpoints.

Includes: /reindex, /v1/index-file, /index/stats,
          /index/lookup/{class_name}, /collections
"""

import os
from typing import Optional

import structlog
from fastapi import APIRouter, HTTPException, BackgroundTasks

from ..dependencies import (
    graph_orchestrator, index_builder,
    run_in_executor,
)
from ..schemas import (
    ReindexRequest, ReindexResponse,
    IndexFileRequest, IndexFileResponse,
)
from ..services.rag_resolver import _derive_collection_name

logger = structlog.get_logger()

router = APIRouter()


@router.post("/reindex", response_model=ReindexResponse)
async def reindex(request: ReindexRequest, background_tasks: BackgroundTasks):
    """Reindex a Java repository into a dedicated Qdrant collection.

    If ``collection`` is not provided, it is auto-derived from the repo
    folder name (e.g. ``vtrip.core.iam`` → ``vtrip_core_iam``).
    The mapping is stored in Redis so ``chat/completions`` can resolve it
    automatically from ``file_path``.
    """
    if not index_builder:
        raise HTTPException(status_code=503, detail="Service not initialized")

    # Validate path exists
    if not os.path.isdir(request.repo_path):
        raise HTTPException(status_code=400, detail=f"Directory not found: {request.repo_path}")

    # Derive collection name
    collection_name = request.collection or _derive_collection_name(request.repo_path)

    try:
        # Run indexing in thread pool to avoid blocking event loop
        points_indexed = await run_in_executor(
            index_builder.index_repository,
            request.repo_path,
            request.recreate,
            collection_name,
        )

        # Register collection → repo_path mapping in Redis
        from utils.cache_service import get_cache_service
        cache = get_cache_service()
        cache.register_collection(collection_name, request.repo_path, points_indexed)

        return ReindexResponse(
            success=True,
            message=f"Successfully indexed {points_indexed} code elements into collection '{collection_name}'",
            collection=collection_name,
            points_indexed=points_indexed,
        )

    except Exception as e:
        logger.error("Reindexing failed", error=str(e), collection=collection_name)
        return ReindexResponse(
            success=False,
            message="Reindexing failed",
            error=str(e),
        )


@router.post("/v1/index-file", response_model=IndexFileResponse)
async def index_file(request: IndexFileRequest):
    """Index a single Java file into Qdrant using Java-aware parsing.

    Continue.dev or any IDE client sends file content here.
    Server parses Java → creates embeddings → pushes to Qdrant.

    Unlike the built-in Continue indexing (generic chunking), this
    endpoint uses the full Java parser with:
    - Dependency tracking
    - Lombok/annotation awareness
    - Model/DTO detection
    - Smart summarization
    """
    if not index_builder:
        raise HTTPException(status_code=503, detail="Service not initialized")

    collection = request.collection or os.getenv("QDRANT_COLLECTION", "java_codebase")

    result = await run_in_executor(
        index_builder.index_single_file,
        request.content,
        request.file_path,
        collection,
    )

    return IndexFileResponse(
        success=result["success"],
        file_path=request.file_path,
        collection=collection,
        classes_indexed=result["classes_indexed"],
        points_created=result["points_created"],
        error=result.get("error"),
    )


@router.get("/index/stats")
async def get_index_stats(collection: Optional[str] = None):
    """Get statistics about the vector index.

    Query param ``collection`` lets you check a specific collection.
    If omitted, uses the default collection.
    """
    if not graph_orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")

    stats = await run_in_executor(graph_orchestrator.rag.get_stats, collection)
    return stats.model_dump()


@router.get("/collections")
async def list_collections():
    """List all indexed Qdrant collections with their metadata.

    Returns both Qdrant-side collections and the registry (repo_path mapping).
    """
    if not graph_orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")

    # Get collections from Qdrant
    qdrant_collections = await run_in_executor(graph_orchestrator.rag.list_collections)

    # Get registry from Redis/memory
    from utils.cache_service import get_cache_service
    cache = get_cache_service()
    registry = cache.get_collection_registry()

    # Merge info
    result = []
    for coll_name in qdrant_collections:
        entry = {"name": coll_name}
        if coll_name in registry:
            entry.update(registry[coll_name])
        result.append(entry)

    # Add registry-only entries (indexed but collection might have been deleted)
    for coll_name, info in registry.items():
        if coll_name not in qdrant_collections:
            entry = {"name": coll_name, "status": "registry_only (no Qdrant collection)"}
            entry.update(info)
            result.append(entry)

    return {"collections": result, "total": len(result)}


@router.get("/index/lookup/{class_name}")
async def lookup_class(class_name: str, collection: Optional[str] = None):
    """Diagnostic: check if a class is indexed in Qdrant and what info is stored.

    Usage: GET /index/lookup/UserProfile?collection=vtrip_core_iam
    Returns the full payload stored in Qdrant for that class, including
    fields, record_components, usage_hint, used_types, has_builder, etc.
    Useful for debugging why the LLM generates wrong construction patterns.
    """
    if not graph_orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")

    rag = graph_orchestrator.rag

    # Tier 1: metadata scroll (guaranteed find)
    chunk = await rag._ascroll_by_class_name(class_name, "class", collection)
    lookup_method = "scroll"

    # Tier 2: semantic search fallback
    if not chunk:
        result = await rag.asearch_by_class(class_name, 1, False, collection)
        chunk = result.chunks[0] if result.chunks else None
        lookup_method = "semantic"

    if not chunk:
        return {
            "found": False,
            "class_name": class_name,
            "message": f"'{class_name}' NOT found in Qdrant index. Re-index the codebase.",
        }

    return {
        "found": True,
        "lookup_method": lookup_method,
        "class_name": chunk.class_name,
        "fully_qualified_name": chunk.fully_qualified_name,
        "package": chunk.package,
        "element_type": chunk.element_type,
        "java_type": chunk.java_type,
        "is_record": chunk.java_type == "record",
        "has_builder": chunk.has_builder,
        "has_data": chunk.has_data,
        "has_value": chunk.has_value,
        "record_components": [
            {"type": rc.type, "name": rc.name} for rc in (chunk.record_components or [])
        ],
        "fields": [
            {"type": f.type, "name": f.name} for f in (chunk.fields or [])
        ],
        "used_types": chunk.used_types,
        "dependencies": chunk.dependencies,
        "annotations": chunk.annotations,
        "summary": chunk.summary,
    }
