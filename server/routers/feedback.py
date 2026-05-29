"""POST /v1/feedback — User feedback collection.

Allows users to rate responses and provide feedback for improvement.
"""

from __future__ import annotations

import logging
import os
import json
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Request
from pydantic import BaseModel

from server.metrics.prometheus import record_feedback

logger = logging.getLogger("server.feedback")

router = APIRouter()

FEEDBACK_DIR = Path(os.environ.get("FEEDBACK_DIR", "data/feedback"))


class FeedbackRequest(BaseModel):
    """User feedback on a response."""
    request_id: str
    rating: int  # 1-5 stars or -1/0/+1 for thumbs
    feedback_type: str = "rating"  # rating, thumbs, comment
    comment: str | None = None
    conversation_id: str | None = None
    intent: str | None = None
    model: str | None = None


class FeedbackResponse(BaseModel):
    success: bool
    message: str


@router.post("/v1/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest, req: Request):
    """Submit user feedback on a response."""
    try:
        # Ensure feedback directory exists
        FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)

        # Create feedback record
        feedback = {
            "timestamp": datetime.now().isoformat(),
            "request_id": request.request_id,
            "rating": request.rating,
            "feedback_type": request.feedback_type,
            "comment": request.comment,
            "conversation_id": request.conversation_id,
            "intent": request.intent,
            "model": request.model,
            "client_ip": req.client.host if req.client else None,
        }

        # Append to daily feedback file
        date_str = datetime.now().strftime("%Y-%m-%d")
        feedback_file = FEEDBACK_DIR / f"feedback-{date_str}.jsonl"

        with open(feedback_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(feedback) + "\n")

        logger.info(
            "Feedback received: request_id=%s rating=%d type=%s",
            request.request_id, request.rating, request.feedback_type
        )

        # Update aggregate stats
        _update_stats(request.rating, request.feedback_type, request.intent)

        # Record Prometheus metrics
        record_feedback(request.feedback_type, request.rating)

        return FeedbackResponse(success=True, message="Thank you for your feedback!")

    except Exception as e:
        logger.error("Failed to save feedback: %s", e)
        return FeedbackResponse(success=False, message=str(e))


@router.get("/v1/feedback/stats")
async def get_feedback_stats():
    """Get aggregated feedback statistics."""
    stats_file = FEEDBACK_DIR / "stats.json"

    if not stats_file.exists():
        return {
            "total_feedback": 0,
            "average_rating": 0,
            "by_intent": {},
            "by_type": {},
        }

    with open(stats_file, "r", encoding="utf-8") as f:
        return json.load(f)


def _update_stats(rating: int, feedback_type: str, intent: str | None):
    """Update aggregate statistics."""
    stats_file = FEEDBACK_DIR / "stats.json"

    if stats_file.exists():
        with open(stats_file, "r", encoding="utf-8") as f:
            stats = json.load(f)
    else:
        stats = {
            "total_feedback": 0,
            "total_rating": 0,
            "average_rating": 0,
            "by_intent": {},
            "by_type": {},
        }

    # Update totals
    stats["total_feedback"] += 1
    if feedback_type == "rating" and 1 <= rating <= 5:
        stats["total_rating"] += rating
        stats["average_rating"] = stats["total_rating"] / stats["total_feedback"]
    elif feedback_type == "thumbs":
        # Convert thumbs to approximate rating for average
        thumb_rating = 5 if rating > 0 else (1 if rating < 0 else 3)
        stats["total_rating"] += thumb_rating
        stats["average_rating"] = stats["total_rating"] / stats["total_feedback"]

    # Update by type
    stats["by_type"][feedback_type] = stats["by_type"].get(feedback_type, 0) + 1

    # Update by intent
    if intent:
        if intent not in stats["by_intent"]:
            stats["by_intent"][intent] = {"count": 0, "total_rating": 0}
        stats["by_intent"][intent]["count"] += 1
        if feedback_type in ("rating", "thumbs"):
            eff_rating = rating if feedback_type == "rating" else (5 if rating > 0 else 1)
            stats["by_intent"][intent]["total_rating"] += eff_rating

    # Save stats
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
