"""
Build prompt node — constructs system + user prompts for LLM.

Wraps:
  - agent/prompt.py  → PromptBuilder
  - Handles single-pass, two-phase, incremental, and repair prompt types
"""

from __future__ import annotations

import structlog

logger = structlog.get_logger()


def build_prompt_node(state: dict, *, prompt_builder, two_phase_strategy=None) -> dict:
    """Build LLM prompts based on strategy and mode.

    Routes to the correct prompt builder method:
      - single_pass → PromptBuilder.build_test_generation_prompt()
      - two_phase   → TwoPhaseStrategy.build_generation_prompt()
      - incremental → PromptBuilder.build_incremental_update_prompt()

    Args:
        state: UnitTestState dict.
        prompt_builder: PromptBuilder instance.
        two_phase_strategy: Optional TwoPhaseStrategy instance.

    Returns:
        State updates: system_prompt, user_prompt.
    """
    strategy = state.get("strategy", "single_pass")
    class_name = state.get("class_name", "")
    file_path = state.get("file_path", "")

    # Reconstruct RAG chunks from serialized state
    # P2: rag_chunks are now raw CodeChunk objects from retrieve_node
    rag_chunks = _ensure_chunk_objects(state.get("rag_chunks", []))

    system_prompt = prompt_builder.build_system_prompt()

    # ── Incremental mode ──
    if state.get("existing_test_code"):
        user_prompt = prompt_builder.build_incremental_update_prompt(
            class_name=class_name,
            file_path=file_path,
            rag_chunks=rag_chunks,
            existing_test_code=state["existing_test_code"],
            changed_methods=state.get("changed_methods", []),
            task_description=state.get("task_description"),
        )
        logger.info("build_prompt_node: incremental mode", class_name=class_name)

    # ── Two-phase mode ──
    elif strategy == "two_phase" and two_phase_strategy:
        try:
            from agent.analysis_prompt import AnalysisResult
            
            analysis_data = state.get("analysis_result")
            if isinstance(analysis_data, dict):
                analysis_obj = AnalysisResult.from_dict(analysis_data)
            else:
                analysis_obj = analysis_data
                
            system_prompt, user_prompt = two_phase_strategy.build_generation_prompt(
                analysis_result=analysis_obj,
                class_name=class_name,
                file_path=file_path,
                rag_chunks=rag_chunks,
                source_code=state.get("source_code", ""),
                collection_name=state.get("collection_name"),
            )
            logger.info("build_prompt_node: two_phase mode", class_name=class_name)
        except Exception as e:
            logger.warning("build_prompt_node: two-phase prompt failed, falling back", error=str(e))
            user_prompt = _build_single_pass_prompt(
                prompt_builder, state, class_name, file_path, rag_chunks,
            )

    # ── Single-pass mode (default) ──
    else:
        user_prompt = _build_single_pass_prompt(
            prompt_builder, state, class_name, file_path, rag_chunks,
        )

    return {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
    }


def _build_single_pass_prompt(prompt_builder, state, class_name, file_path, rag_chunks) -> str:
    """Build single-pass test generation prompt.

    Mirrors _step_build_prompt logic from orchestrator.py.
    """
    task_desc = state.get("task_description") or ""
    source_code = state.get("source_code")

    # If source_code is provided but not already in task_description, wrap it
    if source_code and "```" not in task_desc:
        task_desc = (
            f"{task_desc}\n\n"
            f"```{file_path}\n"
            f"{source_code}\n"
            f"```"
        )

    # P6: Wrap user input in delimiters to prevent prompt injection
    sanitized_task = f"<user_task>\n{task_desc}\n</user_task>" if task_desc else ""

    return prompt_builder.build_test_generation_prompt(
        class_name=class_name,
        file_path=file_path,
        rag_chunks=rag_chunks,
        task_description=sanitized_task,
    )


def _ensure_chunk_objects(chunks: list) -> list:
    """Ensure chunks are CodeChunk objects (P2: usually a no-op now)."""
    if not chunks:
        return []
    if hasattr(chunks[0], "class_name"):
        return chunks
    # Fallback for legacy/external callers passing dicts
    try:
        from rag.schema import CodeChunk
        return [CodeChunk(**c) for c in chunks]
    except Exception:
        return chunks
