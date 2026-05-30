"""HyDE (Hypothetical Document Embedding) for improved retrieval.

Generates a hypothetical answer to the query, then uses that
hypothetical document's embedding for search instead of the query.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger("server.rag.hyde")

# Environment variable to enable HyDE
ENABLE_HYDE = os.environ.get("ENABLE_HYDE", "false").lower() in ("1", "true", "yes")

HYDE_PROMPT = """Given this code search query, write a hypothetical code snippet that would perfectly answer the query.
Write actual working code, not a description. Be specific and include realistic variable names and logic.

Query: {query}

Hypothetical code:
```
"""


async def generate_hypothetical_document(
    query: str,
    vllm_client,
    model: str,
    max_tokens: int = 300,
) -> str | None:
    """Generate a hypothetical document that answers the query.

    Args:
        query: User's search query
        vllm_client: vLLM client for generation
        model: Model name

    Returns:
        Hypothetical document text, or None if generation fails
    """
    if not ENABLE_HYDE:
        return None

    try:
        response = await vllm_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a code generator. Output only code, no explanations."
                },
                {
                    "role": "user",
                    "content": HYDE_PROMPT.format(query=query)
                },
            ],
            temperature=0.3,
            max_tokens=max_tokens,
        )

        content = response.choices[0].message.content or ""

        # Extract code from response
        if "```" in content:
            # Extract first code block
            start = content.find("```")
            end = content.find("```", start + 3)
            if end > start:
                code = content[start:end + 3]
                # Remove language identifier
                lines = code.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1] == "```":
                    lines = lines[:-1]
                content = "\n".join(lines)

        logger.info("HyDE: generated %d char hypothetical document", len(content))
        return content.strip()

    except Exception as e:
        logger.error("HyDE generation failed: %s", e)
        return None


async def hyde_search(
    query: str,
    vllm_client,
    model: str,
    embedder,
    qdrant,
    top_k: int = 5,
    combine_with_query: bool = True,
) -> list[dict]:
    """Search using HyDE (Hypothetical Document Embedding).

    Generates a hypothetical answer, embeds it, and searches with that
    embedding instead of (or combined with) the query embedding.

    Args:
        query: User's search query
        vllm_client: vLLM client
        model: Model name
        embedder: Embedding model
        qdrant: Qdrant service
        top_k: Number of results
        combine_with_query: Also search with original query and merge

    Returns:
        Search results
    """
    hypothetical = await generate_hypothetical_document(query, vllm_client, model)

    if not hypothetical:
        # Fall back to normal search
        logger.info("HyDE: falling back to normal search")
        return await qdrant.hybrid_search(query, top_k=top_k)

    # Embed hypothetical document
    try:
        hyde_embedding = await embedder.embed_async(hypothetical)
    except Exception as e:
        logger.error("HyDE embedding failed: %s", e)
        return await qdrant.hybrid_search(query, top_k=top_k)

    # Search with hypothetical embedding
    hyde_results = await qdrant.search_by_vector(
        dense_vector=hyde_embedding,
        top_k=top_k,
    )

    if not combine_with_query:
        return hyde_results

    # Also search with original query for ensemble
    query_results = await qdrant.hybrid_search(query, top_k=top_k)

    # Merge results using RRF
    merged = _reciprocal_rank_fusion([hyde_results, query_results], k=60)

    logger.info("HyDE: merged %d hyde + %d query results → %d final",
                len(hyde_results), len(query_results), len(merged))

    return merged[:top_k]


def _reciprocal_rank_fusion(
    result_lists: list[list[dict]],
    k: int = 60,
) -> list[dict]:
    """Merge multiple result lists using Reciprocal Rank Fusion.

    Args:
        result_lists: List of result lists to merge
        k: RRF parameter (default 60)

    Returns:
        Merged and re-ranked results
    """
    scores = {}
    docs_by_id = {}

    for results in result_lists:
        for rank, doc in enumerate(results, 1):
            doc_id = doc.get("id", doc.get("file_path", str(hash(str(doc)))))

            if doc_id not in scores:
                scores[doc_id] = 0
                docs_by_id[doc_id] = doc

            # RRF score
            scores[doc_id] += 1 / (k + rank)

    # Sort by RRF score
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    results = []
    for doc_id in sorted_ids:
        doc = docs_by_id[doc_id].copy()
        doc["rrf_score"] = scores[doc_id]
        results.append(doc)

    return results


def is_hyde_enabled() -> bool:
    """Check if HyDE is enabled."""
    return ENABLE_HYDE
