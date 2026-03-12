"""
Check strategy node — decides single_pass vs two_phase.

Wraps the complexity calculation from two_phase_strategy.py.
"""

from __future__ import annotations

import structlog

logger = structlog.get_logger()


def check_strategy_node(state: dict) -> dict:
    """Decide single_pass vs two_phase based on complexity and flags.

    Logic mirrors AgentOrchestrator._should_use_two_phase() +
    ComplexityCalculator from two_phase_strategy.py.

    Returns:
        State updates: strategy, complexity_score.
    """
    # Explicit overrides
    if state.get("force_single_pass"):
        logger.debug("check_strategy: force_single_pass=True")
        return {"strategy": "single_pass", "complexity_score": 0}

    if state.get("force_two_phase"):
        logger.debug("check_strategy: force_two_phase=True")
        return {"strategy": "two_phase", "complexity_score": 999}

    # Incremental mode always uses single-pass
    if state.get("existing_test_code"):
        logger.debug("check_strategy: incremental mode → single_pass")
        return {"strategy": "single_pass", "complexity_score": 0}

    # Calculate complexity from RAG chunks
    rag_chunks = state.get("rag_chunks", [])
    complexity = _calculate_complexity(rag_chunks, state.get("class_name", ""))

    threshold = state.get("complexity_threshold", 10)

    strategy = "two_phase" if complexity >= threshold else "single_pass"

    logger.info(
        "check_strategy: decided",
        strategy=strategy,
        complexity=complexity,
        threshold=threshold,
        class_name=state.get("class_name"),
    )

    return {"strategy": strategy, "complexity_score": complexity}


def _calculate_complexity(rag_chunks: list[dict], class_name: str) -> int:
    """Calculate complexity score from RAG metadata.

    Formula: (dependencies × 2) + (used_types × 3) + method_count

    Mirrors ComplexityCalculator from two_phase_strategy.py.
    """
    # Find the main class chunk
    main_chunk = None
    for chunk in rag_chunks:
        if isinstance(chunk, dict):
            if chunk.get("class_name") == class_name:
                main_chunk = chunk
                break
        elif hasattr(chunk, "class_name") and chunk.class_name == class_name:
            main_chunk = chunk
            break

    if not main_chunk:
        return 0

    # Extract counts
    if isinstance(main_chunk, dict):
        deps = len(main_chunk.get("dependencies", []))
        types = len(main_chunk.get("used_types", []))
        methods = len(main_chunk.get("methods", []))
    else:
        deps = len(getattr(main_chunk, "dependencies", []) or [])
        types = len(getattr(main_chunk, "used_types", []) or [])
        methods = len(getattr(main_chunk, "methods", []) or [])

    return (deps * 2) + (types * 3) + methods
