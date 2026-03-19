"""
Embeddings endpoint (OpenAI-compatible).

Includes: /v1/embeddings
"""

import os

import structlog
from fastapi import APIRouter, HTTPException

from ..dependencies import index_builder, run_in_executor
from ..schemas import (
    EmbeddingRequest, EmbeddingResponse,
    EmbeddingObject, EmbeddingUsage,
)

logger = structlog.get_logger()

router = APIRouter()


@router.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    """OpenAI-compatible embeddings endpoint.

    Uses the local ONNX embedding model (all-MiniLM-L6-v2, 384 dims).
    Compatible with Continue.dev ``embeddingsProvider`` config::

        provider: openai
        apiBase: http://localhost:8080/v1
        model: all-MiniLM-L6-v2-onnx
    """
    if not index_builder:
        raise HTTPException(status_code=503, detail="Service not initialized")

    texts = [request.input] if isinstance(request.input, str) else request.input

    embeddings_np = await run_in_executor(index_builder.embedder.encode, texts)

    # encode() returns 1D for single input, 2D for batch
    import numpy as np
    if embeddings_np.ndim == 1:
        embeddings_np = embeddings_np.reshape(1, -1)

    data = [
        EmbeddingObject(embedding=emb.tolist(), index=i)
        for i, emb in enumerate(embeddings_np)
    ]

    total_tokens = sum(len(t) for t in texts) // 4  # rough estimate

    return EmbeddingResponse(
        data=data,
        model=request.model,
        usage=EmbeddingUsage(prompt_tokens=total_tokens, total_tokens=total_tokens),
    )
