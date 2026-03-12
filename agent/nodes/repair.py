"""
Repair node — 3-level escalating repair with FailureMemory.

Wraps:
  - agent/repair.py → RepairStrategySelector, FailureMemory, RepairReasoningEngine
  - agent/prompt.py → PromptBuilder.build_refinement_prompt()
  - vllm/client.py  → VLLMClient.generate()

Levels:
  1. Targeted → specific fix instructions from validation issues
  2. Reasoning → LLM-based root cause analysis
  3. Regenerate → full regeneration with enriched context
"""

from __future__ import annotations

import structlog

logger = structlog.get_logger()


def repair_node(
    state: dict,
    *,
    repair_selector,
    vllm_client,
    prompt_builder,
    validation_pipeline,
) -> dict:
    """Build repair prompt, call LLM, re-validate.

    Mirrors _step_repair_code + streaming repair loop from orchestrator.py.

    This node:
      1. Determines escalation level (targeted/reasoning/regenerate)
      2. Optionally runs RepairReasoningEngine (Level 2)
      3. Builds repair plan with FailureMemory
      4. Constructs repair prompt
      5. Calls LLM for repair
      6. Extracts code

    The graph loop (repair → validate) handles re-validation.

    Args:
        state: UnitTestState dict.
        repair_selector: RepairStrategySelector instance.
        vllm_client: VLLMClient instance.
        prompt_builder: PromptBuilder instance.
        validation_pipeline: ValidationPipeline instance (for FailureMemory compat).

    Returns:
        State updates: test_code, llm_output, repair_level, failure_memory, user_prompt.
    """
    from agent.repair import FailureMemory, RepairReasoningEngine, RepairLevel

    test_code = state.get("test_code", "")
    issues = state.get("validation_issues", [])
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)
    rag_chunks = state.get("rag_chunks", [])

    # Restore or create FailureMemory
    failure_memory_data = state.get("failure_memory", [])
    memory = FailureMemory()
    if failure_memory_data:
        memory.restore_from_records(failure_memory_data)

    attempt = retry_count  # retry_count already incremented by call_llm

    logger.info(
        "repair_node: starting",
        attempt=attempt,
        max_retries=max_retries,
        issues_count=len(issues),
    )

    # Determine escalation level
    level = repair_selector.determine_level(
        attempt_number=attempt,
        max_attempts=max_retries,
        memory=memory,
    )

    # Level 2+: Run reasoning engine for root-cause analysis
    reasoning = None
    if level == RepairLevel.REASONING:
        try:
            reasoning_engine = RepairReasoningEngine()
            sys_p, usr_p = reasoning_engine.build_reasoning_prompt(
                test_code=test_code,
                validation_issues=issues,
                memory=memory,
            )
            reasoning_resp = vllm_client.generate(
                system_prompt=sys_p,
                user_prompt=usr_p,
                temperature=0.1,
                max_tokens=1500,
            )
            if reasoning_resp.success:
                reasoning = reasoning_engine.parse_reasoning(reasoning_resp.content)
                logger.info("repair_node: reasoning analysis complete")
        except Exception as e:
            logger.warning("repair_node: reasoning engine failed", error=str(e))

    # Build repair plan
    # We need the actual ValidationResult, but we only have the summary.
    # Re-validate to get the structured result for repair planning.
    validation_result = None
    try:
        chunk_objects = _deserialize_chunks(rag_chunks)
        validation_result = validation_pipeline.validate(test_code, rag_chunks=chunk_objects)
    except Exception as e:
        logger.warning("repair_node: re-validation for repair plan failed", error=str(e))

    if validation_result:
        repair_plan = repair_selector.build_repair_plan(
            validation_result=validation_result,
            attempt_number=attempt,
            max_attempts=max_retries,
            memory=memory,
            reasoning=reasoning,
        )
        repair_section = repair_plan.get_repair_prompt_section()
    else:
        repair_section = ""

    # Build repair prompt
    feedback_parts = []
    if repair_section:
        feedback_parts.append(repair_section)
    feedback_parts.append(
        f"Repair attempt {attempt}/{max_retries} (Level: {level.value}). "
        f"Fix these issues:\n" + "\n".join(f"- {msg}" for msg in issues)
    )

    system_prompt = prompt_builder.build_system_prompt()
    user_prompt = prompt_builder.build_refinement_prompt(
        original_code=test_code,
        feedback="\n\n".join(feedback_parts),
        validation_issues=issues,
        rag_chunks=_deserialize_chunks(rag_chunks),
    )

    # Call LLM for repair
    try:
        response = vllm_client.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
    except Exception as e:
        logger.error("repair_node: LLM repair call failed", error=str(e))
        return {"error": f"Repair LLM call failed: {e}"}

    if not response.success:
        return {"error": f"Repair LLM failed: {response.error}"}

    # Extract code
    repaired_code = _extract_code(response.content)

    # Record attempt in failure memory
    issues_after = []  # Will be populated after re-validation in validate node
    try:
        record = memory.record_attempt(
            attempt_number=attempt,
            strategy=level.value,
            instructions_used=[i.description for i in repair_plan.instructions] if repair_plan else [],
            issues_before=issues,
            issues_after=issues_after,
            reasoning=reasoning.get("overall_strategy") if reasoning else None,
        )
    except Exception:
        pass

    # Serialize failure memory for state
    serialized_memory = memory.get_records() if hasattr(memory, "get_records") else []

    tokens_used = state.get("tokens_used", 0) + (response.tokens_used or 0)

    logger.info(
        "repair_node: done",
        level=level.value,
        code_len=len(repaired_code),
        tokens=response.tokens_used,
    )

    return {
        "test_code": repaired_code,
        "llm_output": response.content,
        "repair_level": level.value,
        "failure_memory": serialized_memory,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "tokens_used": tokens_used,
    }


def _extract_code(response: str) -> str:
    """Extract Java code from LLM response."""
    import re
    if not response:
        return ""
    java_match = re.search(r'```java\s*\n(.*?)```', response, re.DOTALL)
    if java_match:
        return java_match.group(1).strip()
    generic_match = re.search(r'```\s*\n(.*?)```', response, re.DOTALL)
    if generic_match:
        return generic_match.group(1).strip()
    return response.strip()


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
