"""Verify RAG sources and mitigate hallucination.

Checks that generated responses are grounded in retrieved sources
and adds citations to improve traceability.
"""

from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher
from typing import Any

logger = logging.getLogger("server.agent.verify_sources")

# Minimum similarity to consider grounded
GROUNDING_THRESHOLD = 0.6

# Code block pattern
CODE_BLOCK_PATTERN = re.compile(r"```[\w]*\n(.*?)```", re.DOTALL)


def extract_code_blocks(text: str) -> list[str]:
    """Extract code blocks from markdown text.

    Args:
        text: Markdown text with code blocks

    Returns:
        List of code block contents
    """
    matches = CODE_BLOCK_PATTERN.findall(text)
    # Filter out empty or very short blocks
    return [m.strip() for m in matches if len(m.strip()) > 20]


def compute_similarity(text1: str, text2: str) -> float:
    """Compute similarity between two text strings.

    Uses SequenceMatcher for efficient similarity computation.

    Args:
        text1: First text
        text2: Second text

    Returns:
        Similarity score between 0 and 1
    """
    if not text1 or not text2:
        return 0.0

    # Normalize whitespace
    text1 = " ".join(text1.split())
    text2 = " ".join(text2.split())

    return SequenceMatcher(None, text1, text2).ratio()


def find_best_match(
    code_block: str,
    rag_chunks: list[dict],
) -> dict | None:
    """Find the RAG chunk that best matches a code block.

    Args:
        code_block: Code block from response
        rag_chunks: Retrieved RAG chunks

    Returns:
        Best matching chunk with score, or None
    """
    if not rag_chunks:
        return None

    best_match = None
    best_score = 0.0

    for chunk in rag_chunks:
        chunk_body = chunk.get("body", chunk.get("content", ""))
        score = compute_similarity(code_block, chunk_body)

        if score > best_score:
            best_score = score
            best_match = {
                **chunk,
                "score": score,
            }

    return best_match if best_score > 0.3 else None


def verify_rag_sources(
    response: str,
    rag_chunks: list[dict],
    threshold: float = GROUNDING_THRESHOLD,
) -> dict[str, Any]:
    """Verify that response is grounded in retrieved sources.

    Args:
        response: Generated response text
        rag_chunks: RAG chunks used for context
        threshold: Minimum similarity to consider grounded

    Returns:
        Verification result dict
    """
    if not response:
        return {
            "total_blocks": 0,
            "grounded_blocks": 0,
            "grounding_rate": 1.0,
            "verifications": [],
            "potentially_hallucinated": [],
        }

    code_blocks = extract_code_blocks(response)

    if not code_blocks:
        # No code blocks to verify
        return {
            "total_blocks": 0,
            "grounded_blocks": 0,
            "grounding_rate": 1.0,
            "verifications": [],
            "potentially_hallucinated": [],
        }

    verifications = []
    for block in code_blocks:
        best_match = find_best_match(block, rag_chunks)

        verification = {
            "block_preview": block[:100] + "..." if len(block) > 100 else block,
            "source": best_match.get("file_path") if best_match else None,
            "similarity": best_match.get("score", 0) if best_match else 0,
            "grounded": (best_match.get("score", 0) >= threshold) if best_match else False,
        }
        verifications.append(verification)

    grounded_count = sum(1 for v in verifications if v["grounded"])
    potentially_hallucinated = [v for v in verifications if not v["grounded"]]

    grounding_rate = grounded_count / len(verifications) if verifications else 1.0

    logger.info("verify_sources: %d/%d blocks grounded (%.1f%%)",
                grounded_count, len(verifications), grounding_rate * 100)

    return {
        "total_blocks": len(code_blocks),
        "grounded_blocks": grounded_count,
        "grounding_rate": grounding_rate,
        "verifications": verifications,
        "potentially_hallucinated": potentially_hallucinated,
    }


def add_citations(
    response: str,
    rag_chunks: list[dict],
    max_citations: int = 5,
) -> str:
    """Add source citations to response.

    Args:
        response: Generated response
        rag_chunks: RAG chunks used
        max_citations: Maximum citations to add

    Returns:
        Response with citations appended
    """
    if not rag_chunks or not response:
        return response

    # Find which chunks were actually used
    code_blocks = extract_code_blocks(response)
    used_chunks = []

    for block in code_blocks:
        match = find_best_match(block, rag_chunks)
        if match and match.get("score", 0) >= 0.4:
            # Avoid duplicates
            file_path = match.get("file_path")
            if not any(c.get("file_path") == file_path for c in used_chunks):
                used_chunks.append(match)

    if not used_chunks:
        return response

    # Limit citations
    used_chunks = used_chunks[:max_citations]

    # Build citations section
    citations = "\n\n---\n**Sources:**\n"
    for i, chunk in enumerate(used_chunks, 1):
        file_path = chunk.get("file_path", "unknown")
        start_line = chunk.get("start_line", "?")
        end_line = chunk.get("end_line", "?")
        citations += f"[{i}] `{file_path}` (lines {start_line}-{end_line})\n"

    return response + citations


def check_hallucination_risk(
    response: str,
    rag_chunks: list[dict],
) -> dict[str, Any]:
    """Quick check for hallucination risk without full verification.

    Args:
        response: Generated response
        rag_chunks: RAG chunks

    Returns:
        Risk assessment dict
    """
    if not rag_chunks:
        return {
            "risk_level": "high",
            "reason": "No RAG context provided",
            "recommendation": "Response may be based on model knowledge only",
        }

    code_blocks = extract_code_blocks(response)

    if not code_blocks:
        return {
            "risk_level": "low",
            "reason": "No code blocks in response",
            "recommendation": None,
        }

    # Quick check - just verify first block
    first_match = find_best_match(code_blocks[0], rag_chunks)

    if not first_match or first_match.get("score", 0) < 0.5:
        return {
            "risk_level": "medium",
            "reason": "First code block has low similarity to sources",
            "recommendation": "Consider verifying response accuracy",
        }

    return {
        "risk_level": "low",
        "reason": "Response appears grounded in sources",
        "recommendation": None,
    }
