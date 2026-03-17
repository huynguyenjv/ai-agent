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
        # Extract available types from pre-fetched RAG chunks
        # Reconstruct CodeChunk objects for attribute access
        rag_chunks = _deserialize_chunks(state.get("rag_chunks", []))
        available_types = [chunk.class_name for chunk in rag_chunks] if rag_chunks else []
        
        # Phase 1: Analyze the service
        result = two_phase_strategy.analyze(
            source_code=source_code,
            class_name=class_name,
            file_path=file_path,
            available_types=available_types,
        )

        if result:
            analysis_result = result.to_dict() if hasattr(result, "to_dict") else {}

            # Search for missing context (domain types not in RAG)
            try:
                enriched_chunks = two_phase_strategy.search_missing_context(
                    analysis_result=result,
                    rag_chunks=rag_chunks,
                    source_code=source_code,
                    collection_name=collection_name,
                )
                if enriched_chunks:
                    # Merge with existing chunks and serialize
                    # existing_rag is a list of dicts from state
                    current_serialized = list(state.get("rag_chunks", []))
                    
                    # Serialize new chunks
                    new_serialized = []
                    for c in enriched_chunks:
                        if hasattr(c, "model_dump"):
                            new_serialized.append(c.model_dump())
                        else:
                            new_serialized.append(c.__dict__)
                            
                    current_serialized.extend(new_serialized)
                    
                    # Update state with enriched list
                    enriched_rag_to_save = current_serialized
                    
                    logger.info(
                        "analyze_node: enriched context",
                        new_chunks=len(enriched_chunks),
                        total=len(enriched_rag_to_save),
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
            # analysis_result is a dict here
            all_types = analysis_result.get("all_domain_types", [])
            if all_types:
                registry_context = domain_registry.build_context_for_types(all_types)
        except Exception as e:
            logger.warning("analyze_node: registry context failed", error=str(e))

    updates = {
        "analysis_result": analysis_result,
        "registry_context": registry_context,
    }
    if 'enriched_rag_to_save' in locals():
        updates["rag_chunks"] = enriched_rag_to_save
        
    return updates


def _deserialize_chunks(raw_chunks: list) -> list:
    """Reconstruct CodeChunk objects from serialized dicts."""
    if not raw_chunks:
        return []
    if hasattr(raw_chunks[0], "class_name"):
        return raw_chunks
    try:
        from rag.schema import CodeChunk
        return [CodeChunk(**c) for c in raw_chunks]
    except Exception:
        return raw_chunks
