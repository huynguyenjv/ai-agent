"""POST /index endpoint — Section 7, POST /index.

Accepts IndexRequest, embeds chunks, upserts to Qdrant.
Authentication via X-Api-Key header.
"""

from __future__ import annotations

import logging
import os

from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import BaseModel

from server.auth import verify_api_key

logger = logging.getLogger("server.index")

router = APIRouter()


class ChunkPayload(BaseModel):
    chunk_id: str
    chunk_type: str
    symbol_name: str
    embed_text: str
    body: str
    file_path: str
    lang: str
    start_line: int
    end_line: int
    deps: list[str] = []
    file_hash: str = ""


class IndexRequest(BaseModel):
    """Section 4.4 — IndexRequest (Client to Server)."""
    chunks: list[ChunkPayload]
    deleted_ids: list[str] = []


@router.post("/index")
async def index_chunks(
    request: IndexRequest,
    req: Request,
    x_api_key: str = Header(None),
    authorization: str = Header(None),
) -> dict:
    """Ingest chunks: delete stale, embed, upsert to Qdrant.

    Section 7: Processing order:
    1. Validate API key
    2. Delete deleted_ids from Qdrant
    3. Embed and upsert each chunk
    4. Return counts
    """
    verify_api_key(req, x_api_key, authorization)

    embedder = req.app.state.embedder
    if embedder is None:
        raise HTTPException(status_code=503, detail="Embedder not available")
    qdrant = req.app.state.qdrant

    # Step 2: Delete stale chunks
    deleted = 0
    if request.deleted_ids:
        deleted = await qdrant.delete_points(request.deleted_ids)

    # Step 3: Embed and upsert
    if not request.chunks:
        return {"indexed": 0, "deleted": deleted}

    chunk_dicts = [c.model_dump() for c in request.chunks]
    embed_texts = [c.embed_text for c in request.chunks]

    # Update IDF stats from indexed documents for better sparse search
    embedder.update_idf_stats(embed_texts)

    # Batch dense embedding
    dense_vectors = embedder.embed_dense_batch(embed_texts)

    # Sparse embedding per chunk
    sparse_vectors = [embedder.embed_sparse(text) for text in embed_texts]

    # Validate vector lengths match chunk count
    if len(dense_vectors) != len(chunk_dicts) or len(sparse_vectors) != len(chunk_dicts):
        raise HTTPException(
            status_code=500,
            detail=f"Vector count mismatch: {len(chunk_dicts)} chunks, "
                   f"{len(dense_vectors)} dense, {len(sparse_vectors)} sparse",
        )

    # Upsert — idempotent due to deterministic chunk_id
    indexed = await qdrant.upsert_chunks(chunk_dicts, dense_vectors, sparse_vectors)

    return {"indexed": indexed, "deleted": deleted}
