"""
Collection/repository resolution and RAG enrichment helpers.

Extracted from api.py — used by chat and test_gen routers.
"""

import os
import re as _re
from typing import Optional

import structlog

from rag.schema import SearchQuery
from ..dependencies import _get_orchestrator, run_in_executor

logger = structlog.get_logger()


def _derive_collection_name(repo_path: str) -> str:
    """Derive a Qdrant-safe collection name from a repository path.

    Examples:
        C:\\Users\\...\\vtrip.core.iam      → vtrip_core_iam
        /home/ci/repos/my-awesome-service → my_awesome_service
    """
    folder = os.path.basename(os.path.normpath(repo_path))
    # Replace dots, hyphens, spaces with underscores; lowercase; strip non-alnum
    safe = _re.sub(r"[^a-zA-Z0-9_]", "_", folder).strip("_").lower()
    return safe or "java_codebase"


def _resolve_collection(
    explicit: Optional[str] = None,
    file_path: Optional[str] = None,
    workspace_path: Optional[str] = None,
) -> str:
    """Resolve which Qdrant collection to use (3-tier fallback).

    1. Explicit collection name from request (Continue extraBodyProperties / CI param)
    2. Auto-detect from file_path via Redis registry
    3. Default from env var
    """
    # Tier 1: explicit
    if explicit:
        return explicit

    # Tier 2: auto-detect from file_path (or workspace_path) via registry
    lookup_path = file_path or workspace_path
    if lookup_path:
        from utils.cache_service import get_cache_service
        cache = get_cache_service()
        matched = cache.resolve_collection_by_path(lookup_path)
        if matched:
            return matched

    # Tier 3: default
    return os.getenv("QDRANT_COLLECTION", "java_codebase")


def _resolve_repo_path(
    workspace_path: Optional[str] = None,
    collection: Optional[str] = None,
    file_path: Optional[str] = None,
) -> Optional[str]:
    """Resolve the repository root path (3-tier fallback).

    1. Explicit workspace_path from IDE
    2. Lookup registry by collection name
    3. Fallback to JAVA_REPO_PATH env var
    """
    # Tier 1: Explicit workspace path (IDE context)
    if workspace_path and os.path.isdir(workspace_path):
        return workspace_path

    from utils.cache_service import get_cache_service
    cache = get_cache_service()

    # Tier 2: Registry lookup
    reg_coll = collection
    if not reg_coll and file_path:
        reg_coll = cache.resolve_collection_by_path(file_path)
    
    if reg_coll:
        registry = cache.get_collection_registry()
        if reg_coll in registry:
            rp = registry[reg_coll].get("repo_path")
            if rp and os.path.isdir(rp):
                return rp

    # Tier 3: Default env var
    return os.getenv("JAVA_REPO_PATH")


async def _enrich_with_rag(
    user_content: str, file_path: str,
    collection_name: Optional[str] = None,
) -> str:
    """Add RAG context to the prompt for general chat."""
    active = _get_orchestrator()
    if not active:
        return user_content
        
    class_name = file_path.replace("\\", "/").split("/")[-1].replace(".java", "")
    try:
        search_result = await active.rag.asearch(
            SearchQuery(
                query=f"{class_name} service methods dependencies",
                top_k=5,
                score_threshold=0.4,
            ),
            collection_name=collection_name,
        )
        if search_result.chunks:
            rag_context = "\n\n## Codebase Context:\n"
            for chunk in search_result.chunks:
                rag_context += f"\n### {chunk.class_name} ({chunk.type}, {chunk.layer})\n"
                rag_context += f"```\n{chunk.summary[:500]}\n```\n"
            return f"{user_content}\n{rag_context}"
    except Exception as e:
        logger.warning("RAG search failed", error=str(e))
    return user_content
