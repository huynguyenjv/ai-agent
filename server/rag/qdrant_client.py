"""QdrantClient wrapper — Section 9, Qdrant Collection Schema.

Single collection 'codebase' with dense (384-dim cosine) and sparse vectors.
Hybrid search with RRF fusion.
"""

from __future__ import annotations

import logging

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    SparseVector,
    VectorParams,
    SparseVectorParams,
    SparseIndexParams,
    Filter,
    FieldCondition,
    MatchValue,
    NamedVector,
    NamedSparseVector,
)

logger = logging.getLogger("server.qdrant")

COLLECTION_NAME = "codebase"
DENSE_DIM = 384


class QdrantService:
    """Manages Qdrant operations for the codebase collection."""

    def __init__(self, url: str = "http://127.0.0.1:6333") -> None:
        self._client = AsyncQdrantClient(url=url)
        self._collection = COLLECTION_NAME

    async def ensure_collection(self) -> None:
        """Create the codebase collection if it doesn't exist."""
        collections = await self._client.get_collections()
        names = [c.name for c in collections.collections]

        if COLLECTION_NAME not in names:
            await self._client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config={
                    "dense": VectorParams(
                        size=DENSE_DIM,
                        distance=Distance.COSINE,
                    ),
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(
                        index=SparseIndexParams(),
                    ),
                },
            )
            # Create payload indexes for filtering
            await self._client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="lang",
                field_schema="keyword",
            )
            await self._client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="chunk_type",
                field_schema="keyword",
            )
            await self._client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="file_path",
                field_schema="keyword",
            )
            logger.info("Created Qdrant collection: %s", COLLECTION_NAME)

    async def upsert_chunks(
        self,
        chunks: list[dict],
        dense_vectors: list[list[float]],
        sparse_vectors: list[dict[int, float]],
    ) -> int:
        """Upsert chunks with their vectors. Returns count of upserted points."""
        points = []
        for i, chunk in enumerate(chunks):
            # Build sparse vector
            sv = sparse_vectors[i]
            sparse_indices = list(sv.keys())
            sparse_values = list(sv.values())

            point = PointStruct(
                id=chunk["chunk_id"],
                vector={
                    "dense": dense_vectors[i],
                },
                payload={
                    "str_id": chunk["chunk_id"],
                    "symbol_name": chunk["symbol_name"],
                    "chunk_type": chunk["chunk_type"],
                    "file_path": chunk["file_path"],
                    "lang": chunk["lang"],
                    "start_line": chunk["start_line"],
                    "end_line": chunk["end_line"],
                    "file_hash": chunk["file_hash"],
                    "body": chunk["body"],
                    "deps": chunk.get("deps", []),
                    "embed_text": chunk["embed_text"],
                },
            )

            # Add sparse vector if non-empty
            if sparse_indices:
                point.vector["sparse"] = SparseVector(
                    indices=sparse_indices,
                    values=sparse_values,
                )

            points.append(point)

        if points:
            await self._client.upsert(
                collection_name=COLLECTION_NAME,
                points=points,
            )

        return len(points)

    async def delete_points(self, ids: list[str]) -> int:
        """Delete points by their IDs."""
        if not ids:
            return 0
        await self._client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=ids,
        )
        return len(ids)

    async def count_by_file(self, file_path: str) -> int:
        """Return number of stored chunks for a given file_path. 0 = miss."""
        try:
            result = await self._client.count(
                collection_name=self._collection,
                count_filter=Filter(
                    must=[FieldCondition(
                        key="file_path",
                        match=MatchValue(value=file_path),
                    )]
                ),
                exact=False,
            )
            return result.count
        except Exception:
            return 0

    async def hybrid_search(
        self,
        dense_vector: list[float],
        sparse_vector: dict[int, float],
        lang_filter: str | None = None,
        top_k: int = 8,
    ) -> list[dict]:
        """Section 9: Hybrid search with RRF fusion.

        1. Dense search: top 2*top_k
        2. Sparse search: top 2*top_k
        3. RRF fusion with k=60
        4. Return top top_k
        """
        query_filter = None
        if lang_filter:
            query_filter = Filter(
                must=[FieldCondition(key="lang", match=MatchValue(value=lang_filter))]
            )

        # Dense search via query_points
        dense_response = await self._client.query_points(
            collection_name=COLLECTION_NAME,
            query=dense_vector,
            using="dense",
            query_filter=query_filter,
            limit=2 * top_k,
            with_payload=True,
        )
        dense_results = dense_response.points

        # Sparse search
        sparse_indices = list(sparse_vector.keys())
        sparse_values = list(sparse_vector.values())
        sparse_results = []
        if sparse_indices:
            sparse_response = await self._client.query_points(
                collection_name=COLLECTION_NAME,
                query=SparseVector(
                    indices=sparse_indices,
                    values=sparse_values,
                ),
                using="sparse",
                query_filter=query_filter,
                limit=2 * top_k,
                with_payload=True,
            )
            sparse_results = sparse_response.points

        # RRF Fusion (k=60)
        rrf_k = 60
        scores: dict[str, float] = {}
        payloads: dict[str, dict] = {}

        for rank, hit in enumerate(dense_results):
            point_id = str(hit.id)
            scores[point_id] = scores.get(point_id, 0) + 1.0 / (rrf_k + rank + 1)
            payloads[point_id] = hit.payload

        for rank, hit in enumerate(sparse_results):
            point_id = str(hit.id)
            scores[point_id] = scores.get(point_id, 0) + 1.0 / (rrf_k + rank + 1)
            if point_id not in payloads:
                payloads[point_id] = hit.payload

        # Sort by combined RRF score
        sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)[:top_k]

        return [
            {**payloads[pid], "rrf_score": scores[pid]}
            for pid in sorted_ids
            if pid in payloads
        ]

    async def close(self) -> None:
        """Close the Qdrant client."""
        await self._client.close()
