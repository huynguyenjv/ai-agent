"""Simple LRU cache for embeddings and LLM responses.

Provides in-memory caching with TTL and size limits.
For production, consider Redis-backed implementation.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("server.cache")


@dataclass
class CacheEntry:
    """Cache entry with TTL."""
    value: Any
    created_at: float
    ttl_seconds: float

    def is_expired(self) -> bool:
        return time.time() - self.created_at > self.ttl_seconds


class LRUCache:
    """Thread-safe LRU cache with TTL.

    Usage:
        cache = LRUCache(max_size=1000, default_ttl=3600)
        cache.set("key", value)
        result = cache.get("key")
    """

    def __init__(self, max_size: int = 1000, default_ttl: float = 3600):
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Any | None:
        """Get value from cache."""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._misses += 1
                return None

            if entry.is_expired():
                del self._cache[key]
                self._misses += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            return entry.value

    def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        """Set value in cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]

            # Evict oldest if at capacity
            while len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)

            self._cache[key] = CacheEntry(
                value=value,
                created_at=time.time(),
                ttl_seconds=ttl or self._default_ttl,
            )

    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> None:
        """Clear all entries."""
        with self._lock:
            self._cache.clear()

    def stats(self) -> dict:
        """Get cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / total if total > 0 else 0,
            }


# =============================================================================
# Embedding Cache
# =============================================================================

def _hash_text(text: str) -> str:
    """Create hash key for text content."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


class EmbeddingCache:
    """Cache for embeddings to avoid redundant API calls."""

    def __init__(self, max_size: int = 5000, ttl_hours: float = 24):
        self._cache = LRUCache(max_size=max_size, default_ttl=ttl_hours * 3600)

    def get(self, text: str) -> list[float] | None:
        """Get cached embedding for text."""
        key = _hash_text(text)
        return self._cache.get(key)

    def set(self, text: str, embedding: list[float]) -> None:
        """Cache embedding for text."""
        key = _hash_text(text)
        self._cache.set(key, embedding)

    def get_many(self, texts: list[str]) -> tuple[list[int], list[list[float]]]:
        """Get cached embeddings, return indices of cache hits and their embeddings."""
        hits_idx = []
        hits_embeddings = []
        for i, text in enumerate(texts):
            emb = self.get(text)
            if emb is not None:
                hits_idx.append(i)
                hits_embeddings.append(emb)
        return hits_idx, hits_embeddings

    def stats(self) -> dict:
        return self._cache.stats()


# =============================================================================
# Singleton
# =============================================================================

_embedding_cache: EmbeddingCache | None = None
_cache_lock = threading.Lock()


def get_embedding_cache() -> EmbeddingCache:
    """Get singleton EmbeddingCache instance."""
    global _embedding_cache

    if _embedding_cache is None:
        with _cache_lock:
            if _embedding_cache is None:
                _embedding_cache = EmbeddingCache()
                logger.info("EmbeddingCache initialized")

    return _embedding_cache


# =============================================================================
# LLM Response Cache (Phase 8.4)
# =============================================================================

class LLMResponseCache:
    """Cache LLM responses for repeated/similar queries.

    Only caches final responses (not tool-calling turns).
    Uses intent + recent messages as cache key.
    """

    def __init__(self, max_size: int = 500, ttl_hours: float = 1):
        self._cache = LRUCache(max_size=max_size, default_ttl=ttl_hours * 3600)

    def _make_key(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        intent: str,
    ) -> str:
        """Create cache key from request params."""
        import json

        # Only use last 3 messages for key
        recent = messages[-3:] if len(messages) > 3 else messages

        key_data = {
            "messages": [
                {"role": m.get("role", ""), "content": str(m.get("content", ""))[:500]}
                for m in recent
                if isinstance(m, dict)
            ],
            "tools": [t.get("function", {}).get("name", "") for t in (tools or [])],
            "intent": intent,
        }

        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()[:32]

    def get(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        intent: str,
    ) -> dict | None:
        """Get cached response if exists."""
        key = self._make_key(messages, tools, intent)
        result = self._cache.get(key)
        if result:
            logger.info("LLM cache hit for intent=%s", intent)
        return result

    def set(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        intent: str,
        response: dict,
    ) -> None:
        """Cache response."""
        key = self._make_key(messages, tools, intent)
        self._cache.set(key, response)
        logger.debug("LLM response cached for intent=%s", intent)

    def stats(self) -> dict:
        return self._cache.stats()


_llm_cache: LLMResponseCache | None = None


def get_llm_cache() -> LLMResponseCache:
    """Get singleton LLMResponseCache instance."""
    global _llm_cache

    if _llm_cache is None:
        with _cache_lock:
            if _llm_cache is None:
                _llm_cache = LLMResponseCache()
                logger.info("LLMResponseCache initialized")

    return _llm_cache


# =============================================================================
# RAG Result Cache (Phase 8.5)
# =============================================================================

class RAGResultCache:
    """Cache RAG search results.

    Short TTL (5 min) as index may change.
    """

    def __init__(self, max_size: int = 500, ttl_seconds: float = 300):
        self._cache = LRUCache(max_size=max_size, default_ttl=ttl_seconds)

    def _make_key(
        self,
        query_hash: str,
        lang_filter: str | None,
        top_k: int,
    ) -> str:
        """Create cache key for search."""
        return f"{query_hash}:{lang_filter or 'all'}:{top_k}"

    def get(
        self,
        query_hash: str,
        lang_filter: str | None = None,
        top_k: int = 8,
    ) -> list[dict] | None:
        """Get cached search results."""
        key = self._make_key(query_hash, lang_filter, top_k)
        result = self._cache.get(key)
        if result:
            logger.info("RAG cache hit")
        return result

    def set(
        self,
        query_hash: str,
        results: list[dict],
        lang_filter: str | None = None,
        top_k: int = 8,
    ) -> None:
        """Cache search results."""
        key = self._make_key(query_hash, lang_filter, top_k)
        self._cache.set(key, results)

    def invalidate(self) -> None:
        """Invalidate all cached results (e.g., after re-indexing)."""
        self._cache.clear()
        logger.info("RAG cache invalidated")

    def stats(self) -> dict:
        return self._cache.stats()


_rag_cache: RAGResultCache | None = None


def get_rag_cache() -> RAGResultCache:
    """Get singleton RAGResultCache instance."""
    global _rag_cache

    if _rag_cache is None:
        with _cache_lock:
            if _rag_cache is None:
                _rag_cache = RAGResultCache()
                logger.info("RAGResultCache initialized")

    return _rag_cache
