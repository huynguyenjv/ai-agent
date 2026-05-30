"""Query expansion for better RAG retrieval.

Generates query variants to improve recall by searching with
multiple phrasings of the same intent.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger("server.rag.query_expand")

ENABLE_QUERY_EXPANSION = os.environ.get("ENABLE_QUERY_EXPANSION", "false").lower() in ("1", "true", "yes")
MAX_VARIANTS = 3

EXPAND_PROMPT = """Generate {num_variants} alternative phrasings of this code search query.
Keep the same intent, vary the terminology and structure.
Include synonyms and different ways developers might describe the same thing.

Query: {query}

Output as JSON array only, no explanation:
["variant1", "variant2", "variant3"]
"""


async def expand_query(
    query: str,
    vllm_client=None,
    model: str = "",
    num_variants: int = MAX_VARIANTS,
) -> list[str]:
    """Generate query variants for better retrieval.

    Args:
        query: Original search query
        vllm_client: vLLM client for generation
        model: Model name
        num_variants: Number of variants to generate

    Returns:
        List of queries (original + variants)
    """
    if not ENABLE_QUERY_EXPANSION or vllm_client is None:
        return [query]

    try:
        response = await vllm_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You generate search query variants. Output only valid JSON array."
                },
                {
                    "role": "user",
                    "content": EXPAND_PROMPT.format(query=query, num_variants=num_variants)
                },
            ],
            temperature=0.7,
            max_tokens=200,
        )

        content = response.choices[0].message.content or ""
        variants = _parse_variants(content)

        if variants:
            logger.info("query_expand: generated %d variants for query", len(variants))
            return [query] + variants[:num_variants]

    except Exception as e:
        logger.error("query_expand: failed to generate variants: %s", e)

    return [query]


def _parse_variants(content: str) -> list[str]:
    """Parse LLM response to extract variants.

    Args:
        content: Raw LLM response

    Returns:
        List of variant strings
    """
    content = content.strip()

    # Try to extract JSON array
    if "[" in content:
        start = content.find("[")
        end = content.rfind("]") + 1
        if end > start:
            content = content[start:end]

    try:
        variants = json.loads(content)
        if isinstance(variants, list):
            return [v for v in variants if isinstance(v, str) and v.strip()]
    except json.JSONDecodeError:
        pass

    return []


async def search_with_expansion(
    query: str,
    search_fn,
    vllm_client=None,
    model: str = "",
    top_k: int = 5,
) -> list[dict]:
    """Search with query expansion and merge results.

    Args:
        query: Original query
        search_fn: Async search function
        vllm_client: vLLM client
        model: Model name
        top_k: Results per query

    Returns:
        Merged and deduplicated results
    """
    queries = await expand_query(query, vllm_client, model)

    if len(queries) == 1:
        # No expansion, just search
        return await search_fn(queries[0], top_k=top_k)

    # Search with all variants
    all_results = []
    for q in queries:
        results = await search_fn(q, top_k=top_k)
        all_results.append(results)

    # Merge with RRF
    merged = _merge_results(all_results)

    logger.info("query_expand: merged %d result sets → %d unique",
                len(all_results), len(merged))

    return merged[:top_k]


def _merge_results(result_lists: list[list[dict]], k: int = 60) -> list[dict]:
    """Merge multiple result lists using RRF.

    Args:
        result_lists: List of result lists
        k: RRF parameter

    Returns:
        Merged results
    """
    scores = {}
    docs_by_id = {}

    for results in result_lists:
        for rank, doc in enumerate(results, 1):
            doc_id = _get_doc_id(doc)

            if doc_id not in scores:
                scores[doc_id] = 0
                docs_by_id[doc_id] = doc

            scores[doc_id] += 1 / (k + rank)

    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    return [
        {**docs_by_id[doc_id], "expansion_score": scores[doc_id]}
        for doc_id in sorted_ids
    ]


def _get_doc_id(doc: dict) -> str:
    """Get unique ID for a document."""
    return doc.get("id", doc.get("file_path", "") + str(doc.get("start_line", "")))


def is_expansion_enabled() -> bool:
    """Check if query expansion is enabled."""
    return ENABLE_QUERY_EXPANSION
