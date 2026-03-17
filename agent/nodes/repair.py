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
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from ..repair import FailureMemory, RepairReasoningEngine, RepairLevel, RepairAttemptRecord
from ..validation_schema import ValidationResult, ValidationIssue, IssueSeverity, IssueCategory

logger = structlog.get_logger()


async def repair_node(
    state: dict,
    *,
    repair_selector,
    vllm_client,
    prompt_builder,
    validation_pipeline,
) -> dict:
    """Build repair prompt, call LLM, re-validate."""
    
    test_code = state.get("test_code", "")
    issues = state.get("validation_issues", [])
    raw_issues_data = state.get("validation_issues_raw", [])
    execution_output = state.get("execution_output", "")
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)
    rag_chunks = state.get("rag_chunks", [])
    
    attempt = retry_count + 1
    
    logger.info("repair_node: starting", attempt=attempt, issues_count=len(issues))

    # 1. Reconstruct ValidationResult for the selector
    reconstructed_issues = []
    for issue_dict in raw_issues_data:
        try:
            # Handle string vs enum conversion if needed, but ValidationIssue constructor usually handles strings
            reconstructed_issues.append(ValidationIssue(**issue_dict))
        except Exception:
            logger.warning("repair_node: failed to reconstruct issue", issue=issue_dict)
    
    validation_result = ValidationResult(
        issues=reconstructed_issues,
        test_count=state.get("validation_result", {}).get("test_count", 0)
    )

    # 2. Reconstruct FailureMemory from state
    memory = FailureMemory()
    serialized_memory = state.get("failure_memory", [])
    for rec_dict in serialized_memory:
        try:
            # Convert dict back to RepairAttemptRecord
            record = RepairAttemptRecord(**rec_dict)
            memory._attempts.append(record)
        except Exception:
            pass

    # 3. Determine repair level and perform reasoning if Level 2
    level = repair_selector.determine_level(attempt, max_retries, memory)
    
    reasoning = None
    if level == RepairLevel.REASONING:
        logger.info("repair_node: level 2 - generating reasoning")
        sys_p, usr_p = RepairReasoningEngine.build_reasoning_prompt(test_code, issues, memory)
        resp = await vllm_client.agenerate(system_prompt=sys_p, user_prompt=usr_p)
        if resp.success:
            reasoning = RepairReasoningEngine.parse_reasoning(resp.content)

    # 4. Build repair plan
    repair_plan = repair_selector.build_repair_plan(
        validation_result=validation_result,
        attempt_number=attempt,
        max_attempts=max_retries,
        memory=memory,
        reasoning=reasoning
    )
    
    repair_section = repair_plan.get_repair_prompt_section()
    
    feedback_parts = []
    if execution_output:
        feedback_parts.append(f"### Real-world Execution Logs (Maven):\n{execution_output}")
    
    if repair_section:
        feedback_parts.append(repair_section)
    
    feedback_parts.append(
        f"Repair attempt {attempt}/{max_retries} (Level: {level.value}). "
        f"Fix these issues:\n" + "\n".join(f"- {msg}" for msg in issues)
    )

    # 5. Build prompts and call LLM
    system_prompt = prompt_builder.build_system_prompt()
    user_prompt = prompt_builder.build_refinement_prompt(
        original_code=test_code,
        feedback="\n\n".join(feedback_parts),
        validation_issues=issues,
        rag_chunks=_deserialize_chunks(rag_chunks),
    )

    try:
        response = await vllm_client.agenerate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
    except Exception as e:
        logger.error("repair_node: LLM repair call failed", error=str(e))
        return {"error": f"Repair LLM call failed: {e}", "retry_count": attempt}

    if not response.success:
        return {"error": f"Repair LLM failed: {response.error}", "retry_count": attempt}

    # 6. Extract code and prepare update
    repaired_code = _extract_code(response.content)

    # Note: Recording the attempt will happen after re-validation in the NEXT turn or inside validate_node.
    # But we can pre-record the 'before' state here to be updated later, or just wait for the next repair call.
    # In the current graph flow, repair is followed by validate. 
    # FailureMemory record_attempt needs issues_after, which we don't have yet.
    # So we'll update memory in the NEXT repair_node call based on what happened in between.
    # Actually, let's record the partial attempt now.
    
    record = memory.record_attempt(
        attempt_number=attempt,
        strategy=level.value,
        instructions_used=[i.description for i in repair_plan.instructions] if repair_plan else [],
        issues_before=issues,
        issues_after=[], # To be filled later
        reasoning=reasoning.get("overall_strategy") if reasoning else None,
    )

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
        "failure_memory": [asdict(r) for r in memory.attempts],
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "tokens_used": tokens_used,
        "retry_count": attempt,
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
