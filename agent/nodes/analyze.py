"""
Analyze node — Phase 1 of Two-Phase Strategy.

Wraps TwoPhaseStrategy.analyze() + search_missing_context().
Only runs when strategy == "two_phase".
"""

from __future__ import annotations

import structlog

logger = structlog.get_logger()


def analyze_node(state: dict, *, two_phase_strategy, domain_registry=None) -> dict:
    """Run Phase 1 analysis (two-phase only).

    Calls:
      - TwoPhaseStrategy.analyze() — LLM-based service analysis
      - TwoPhaseStrategy.search_missing_context() — enrich context
      - DomainTypeRegistry context building

    Args:
        state: UnitTestState dict.
        two_phase_strategy: TwoPhaseStrategy instance.
        domain_registry: Optional DomainTypeRegistry instance.

    Returns:
        State updates: analysis_result, registry_context, rag_chunks (enriched).
    """
    class_name = state.get("class_name", "")
    source_code = state.get("source_code", "")
    file_path = state.get("file_path", "")
    collection_name = state.get("collection_name")

    if not source_code:
        logger.warning("analyze_node: no source code available, skipping analysis")
        return {"analysis_result": {}, "registry_context": ""}

    logger.info("analyze_node: starting Phase 1 analysis", class_name=class_name)

    analysis_result = {}
    registry_context = ""

    try:
        # Phase 1: Analyze the service
        result = two_phase_strategy.analyze(
            source_code=source_code,
            class_name=class_name,
            file_path=file_path,
        )

        if result:
            analysis_result = result.to_dict() if hasattr(result, "to_dict") else {}

            # Search for missing context (domain types not in RAG)
            try:
                enriched_chunks = two_phase_strategy.search_missing_context(
                    analysis_result=result,
                    rag_chunks=state.get("rag_chunks", []),
                    source_code=source_code,
                    collection_name=collection_name,
                )
                if enriched_chunks:
                    # Merge with existing chunks
                    existing_rag = list(state.get("rag_chunks", []))
                    existing_rag.extend(enriched_chunks)
                    logger.info(
                        "analyze_node: enriched context",
                        new_chunks=len(enriched_chunks),
                        total=len(existing_rag),
                    )
            except Exception as e:
                logger.warning("analyze_node: context enrichment failed", error=str(e))

            logger.info(
                "analyze_node: analysis complete",
                class_name=class_name,
                complexity=result.complexity_score if hasattr(result, "complexity_score") else 0,
                methods=len(result.methods) if hasattr(result, "methods") else 0,
            )

    except Exception as e:
        logger.error("analyze_node: Phase 1 analysis failed", error=str(e))
        return {"analysis_result": {}, "registry_context": "", "error": str(e)}

    # Build registry context
    if domain_registry:
        try:
            domain_registry.build_from_collection(collection_name=collection_name)
            # Get construction patterns for types used by this class
            if hasattr(analysis_result, "all_domain_types"):
                registry_context = domain_registry.build_context_for_types(
                    analysis_result.all_domain_types
                )
        except Exception as e:
            logger.warning("analyze_node: registry context failed", error=str(e))

    return {
        "analysis_result": analysis_result,
        "registry_context": registry_context,
    }
