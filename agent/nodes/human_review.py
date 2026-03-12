"""
Human review node — interrupt point for human approval.

Uses LangGraph's interrupt() to pause execution and wait for user input.
Auto-approves when require_human_review is False (default).
"""

from __future__ import annotations

import structlog

logger = structlog.get_logger()


def human_review_node(state: dict) -> dict:
    """Interrupt for human approval (or auto-approve).

    Behavior:
      - require_human_review=False (default) → auto-approve, no interrupt
      - require_human_review=True → call interrupt(), pause graph

    When the graph is resumed after interrupt, the caller provides
    human_approved (bool) and human_feedback (str) via update_state().

    Returns:
        State updates: human_approved, human_feedback.
    """
    require_review = state.get("require_human_review", False)

    if not require_review:
        # Auto-approve — no interrupt
        logger.debug("human_review_node: auto-approve (no review requested)")
        return {
            "human_approved": True,
            "human_feedback": "",
        }

    # ── Interrupt for human review ──
    from langgraph.types import interrupt

    test_code = state.get("test_code", "")
    validation_passed = state.get("validation_passed", False)
    validation_issues = state.get("validation_issues", [])
    class_name = state.get("class_name", "")

    logger.info(
        "human_review_node: interrupting for review",
        class_name=class_name,
        validation_passed=validation_passed,
    )

    # Interrupt — graph execution pauses here.
    # The caller resumes by providing a review decision.
    review_data = interrupt({
        "type": "human_review",
        "class_name": class_name,
        "test_code": test_code,
        "validation_passed": validation_passed,
        "validation_issues": validation_issues,
        "message": (
            f"Review generated test for {class_name}.\n"
            f"Validation: {'PASSED' if validation_passed else 'FAILED'}\n"
            f"Issues: {len(validation_issues)}"
        ),
    })

    # After resume, review_data contains the user's decision
    approved = review_data.get("approved", True) if isinstance(review_data, dict) else True
    feedback = review_data.get("feedback", "") if isinstance(review_data, dict) else ""

    logger.info(
        "human_review_node: review received",
        approved=approved,
        has_feedback=bool(feedback),
    )

    return {
        "human_approved": approved,
        "human_feedback": feedback,
    }
