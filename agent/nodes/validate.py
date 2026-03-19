"""
Validate node — runs the full 7-pass ValidationPipeline.

Wraps: agent/validation.py → ValidationPipeline.validate()
"""

from __future__ import annotations

import structlog

logger = structlog.get_logger()


from dataclasses import asdict
from ..validation_schema import ValidationResult
from typing import Any

def validate_node(state: dict, *, validation_pipeline) -> dict:
    """Run 7-pass validation on generated test code.

    Passes:
      1. Structural (empty, class declaration, braces)
      2. Forbidden patterns (@SpringBootTest, etc.)
      3. Required annotations (@Test, @Mock, @InjectMocks)
      4. AAA pattern adherence
      5. Quality metrics (test count, assertions)
      6. Anti-patterns
      7. RAG-aware construction pattern cross-check

    Args:
        state: UnitTestState dict.
        validation_pipeline: ValidationPipeline instance.

    Returns:
        State updates: validation_passed, validation_issues, validation_result, validation_issues_raw.
    """
    test_code = state.get("test_code", "")
    rag_chunks = state.get("rag_chunks", [])

    if not test_code:
        return {
            "validation_passed": False,
            "validation_issues": ["Empty test code"],
            "validation_result": {"passed": False, "errors": 1},
            "validation_issues_raw": [],
        }

    # P2: rag_chunks are now raw CodeChunk objects from retrieve_node
    chunk_objects = _ensure_chunk_objects(rag_chunks)

    logger.info("validate_node: running 7-pass validation", code_len=len(test_code))

    result = validation_pipeline.validate(test_code, rag_chunks=chunk_objects)

    validation_result = result.get_summary() if hasattr(result, "get_summary") else {}

    logger.info(
        "validate_node: done",
        passed=result.passed,
        errors=len(result.errors),
        warnings=len(result.warnings),
        test_count=result.test_count,
    )

    # If static validation passes, request Maven execution (Option C)
    tool_request = None
    if result.passed:
        tool_request = {
            "name": "run_test",
            "parameters": {
                "test_class": state.get("class_name", "") + "Test" if not state.get("class_name", "").endswith("Test") else state.get("class_name")
            }
        }

    # Serialize issues for state
    raw_issues = [asdict(i) for i in result.issues]

    return {
        "validation_passed": result.passed,
        "validation_issues": result.all_messages,
        "validation_result": validation_result,
        "validation_issues_raw": raw_issues,
        "tool_request": tool_request,
    }


def _ensure_chunk_objects(chunks: list) -> list:
    """Ensure chunks are CodeChunk objects (P2: usually a no-op now)."""
    if not chunks:
        return []
    # Already objects — fast path
    if hasattr(chunks[0], "class_name"):
        return chunks
    # Fallback: deserialize dicts (legacy or external callers)
    try:
        from rag.schema import CodeChunk
        return [CodeChunk(**c) for c in chunks]
    except Exception:
        return chunks
