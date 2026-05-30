"""Re-ranking layer for RAG results.

Uses cross-encoder models to re-rank retrieved documents by relevance.
Falls back to simple scoring if cross-encoder not available.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger("server.rag.reranker")

# Environment variable to enable/disable reranker
ENABLE_RERANKER = os.environ.get("ENABLE_RERANKER", "false").lower() in ("1", "true", "yes")
RERANKER_MODEL = os.environ.get("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# Lazy-loaded model
_reranker_model = None


def _get_reranker():
    """Lazy load the cross-encoder model."""
    global _reranker_model

    if _reranker_model is not None:
        return _reranker_model

    if not ENABLE_RERANKER:
        return None

    try:
        from sentence_transformers import CrossEncoder
        _reranker_model = CrossEncoder(RERANKER_MODEL)
        logger.info("Loaded reranker model: %s", RERANKER_MODEL)
        return _reranker_model
    except ImportError:
        logger.warning("sentence-transformers not installed, reranker disabled")
        return None
    except Exception as e:
        logger.error("Failed to load reranker model: %s", e)
        return None


def rerank(
    query: str,
    documents: list[dict],
    top_k: int = 5,
    score_field: str = "body",
) -> list[dict]:
    """Re-rank documents by relevance to query.

    Uses cross-encoder if available, otherwise falls back to
    keeping original order with score normalization.

    Args:
        query: Search query
        documents: List of document dicts
        top_k: Number of top results to return
        score_field: Field in document to use for scoring

    Returns:
        Re-ranked documents with updated scores
    """
    if not documents:
        return []

    if len(documents) <= 1:
        return documents[:top_k]

    reranker = _get_reranker()

    if reranker is not None:
        return _rerank_with_crossencoder(reranker, query, documents, top_k, score_field)
    else:
        return _fallback_rerank(query, documents, top_k, score_field)


def _rerank_with_crossencoder(
    model,
    query: str,
    documents: list[dict],
    top_k: int,
    score_field: str,
) -> list[dict]:
    """Re-rank using cross-encoder model."""
    # Prepare query-document pairs
    pairs = []
    for doc in documents:
        text = doc.get(score_field, doc.get("content", doc.get("body", "")))
        pairs.append((query, text))

    # Score all pairs
    try:
        scores = model.predict(pairs)
    except Exception as e:
        logger.error("Cross-encoder scoring failed: %s", e)
        return _fallback_rerank(query, documents, top_k, score_field)

    # Combine documents with scores
    scored_docs = list(zip(documents, scores))

    # Sort by score descending
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    # Return top_k with updated scores
    results = []
    for doc, score in scored_docs[:top_k]:
        result = {**doc}
        result["rerank_score"] = float(score)
        result["original_score"] = doc.get("score", 0)
        results.append(result)

    logger.info("Reranked %d documents, top score: %.3f",
                len(results), results[0]["rerank_score"] if results else 0)

    return results


def _fallback_rerank(
    query: str,
    documents: list[dict],
    top_k: int,
    score_field: str,
) -> list[dict]:
    """Fallback reranking using simple keyword matching."""
    query_terms = set(query.lower().split())

    scored_docs = []
    for doc in documents:
        text = doc.get(score_field, doc.get("content", doc.get("body", "")))
        text_terms = set(text.lower().split())

        # Simple overlap score
        overlap = len(query_terms & text_terms)
        original_score = doc.get("score", 0)

        # Combine overlap with original score
        combined_score = (overlap * 0.1) + (original_score * 0.9)

        scored_docs.append((doc, combined_score))

    # Sort by combined score
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    # Return top_k
    results = []
    for doc, score in scored_docs[:top_k]:
        result = {**doc}
        result["rerank_score"] = score
        results.append(result)

    return results


def batch_rerank(
    queries: list[str],
    documents_list: list[list[dict]],
    top_k: int = 5,
) -> list[list[dict]]:
    """Batch re-rank multiple query-document sets.

    Args:
        queries: List of queries
        documents_list: List of document lists (one per query)
        top_k: Top results per query

    Returns:
        List of re-ranked document lists
    """
    return [
        rerank(query, docs, top_k)
        for query, docs in zip(queries, documents_list)
    ]
