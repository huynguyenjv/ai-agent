"""
Save result node — persists the final test code and updates session memory.

Wraps: agent/memory.py → SessionMemory + MemoryManager
"""

from __future__ import annotations

import json

import structlog

logger = structlog.get_logger()


def save_result_node(state: dict, *, memory_manager=None) -> dict:
    """Persist final result and update session memory.

    Args:
        state: UnitTestState dict.
        memory_manager: Optional MemoryManager instance.

    Returns:
        State updates: final_test_code, subgraph_result.
    """
    test_code = state.get("test_code", "")
    class_name = state.get("class_name", "")
    session_id = state.get("session_id")
    validation_passed = state.get("validation_passed", False)
    validation_issues = state.get("validation_issues", [])

    logger.info(
        "save_result_node: persisting",
        class_name=class_name,
        code_len=len(test_code),
        validation_passed=validation_passed,
    )

    # Update session memory if available
    if memory_manager and session_id:
        try:
            session = memory_manager.get_or_create_session(session_id)
            session.add_assistant_message(
                test_code,
                metadata={"validation_passed": validation_passed},
            )
            session.record_generated_test(
                class_name=class_name,
                test_code=test_code,
                success=validation_passed,
            )
            logger.debug("save_result_node: session memory updated", session_id=session_id)
        except Exception as e:
            logger.warning("save_result_node: session update failed", error=str(e))

    # Build the final result dict
    result = {
        "success": True,
        "test_code": test_code,
        "class_name": class_name,
        "file_path": state.get("file_path", ""),
        "validation_passed": validation_passed,
        "validation_issues": validation_issues,
        "tokens_used": state.get("tokens_used", 0),
        "rag_chunks_used": len(state.get("rag_chunks", [])),
        "strategy_used": state.get("strategy", "single_pass"),
        "complexity_score": state.get("complexity_score", 0),
        "repair_attempts": max(0, state.get("retry_count", 1) - 1),
        "analysis_result": state.get("analysis_result"),
    }

    # Add error info if present
    error = state.get("error")
    if error:
        result["success"] = False
        result["error"] = error

    return {
        "final_test_code": test_code,
        "subgraph_result": json.dumps(result),
    }
