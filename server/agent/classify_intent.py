"""Node: classify_intent — Keyword scoring approach.

Score-based intent classification with Vietnamese and English support.
Each keyword has a weight. Intent with highest score wins.
"""

from __future__ import annotations

import logging
import unicodedata

from langchain_core.messages import ToolMessage

from server.agent.state import AgentState

logger = logging.getLogger("server.classify_intent")


# Intent keywords with weights (higher = stronger signal)
# Format: {intent: [(keyword, weight), ...]}
INTENT_KEYWORDS: dict[str, list[tuple[str, int]]] = {
    "explain": [
        # Vietnamese
        ("giải thích", 10),
        ("giải-thích", 10),
        ("giaithich", 10),
        ("cấu trúc", 5),
        ("hoạt động", 5),
        ("làm gì", 5),
        ("như thế nào", 5),
        ("là gì", 5),
        ("đọc", 3),
        # English
        ("explain", 10),
        ("how does", 8),
        ("what does", 8),
        ("what is", 5),
        ("how it works", 8),
        ("understand", 5),
    ],
    "code_review": [
        # Vietnamese
        ("review code", 10),
        ("code review", 10),
        ("review lại", 8),
        ("review giúp", 8),
        ("đánh giá code", 8),
        ("kiểm tra code", 6),
        # English
        ("review", 5),  # lower weight - needs context
        ("audit", 8),
        ("merge request", 8),
        ("pull request", 8),
        ("pr review", 10),
        ("mr review", 10),
    ],
    "unit_test": [
        # Vietnamese
        ("viết test", 10),
        ("tạo test", 10),
        ("sinh test", 8),
        ("unit test", 10),
        # English
        ("write test", 10),
        ("generate test", 10),
        ("create test", 10),
        ("unit test", 10),
        ("test case", 8),
    ],
    "structural_analysis": [
        # Vietnamese
        ("phân tích cấu trúc", 10),
        ("kiến trúc", 8),
        ("tổng quan", 8),
        ("toàn bộ project", 8),
        ("cấu trúc project", 8),
        # English
        ("architecture", 8),
        ("project structure", 10),
        ("analyze structure", 10),
        ("overview", 6),
        ("dependency", 5),
    ],
    "search": [
        # Vietnamese
        ("tìm hàm", 10),
        ("tìm class", 10),
        ("tìm function", 10),
        ("tìm method", 10),
        ("ở đâu", 8),
        ("nằm ở", 6),
        # English
        ("find function", 10),
        ("find class", 10),
        ("find method", 10),
        ("where is", 8),
        ("search for", 8),
        ("locate", 6),
    ],
    "refine": [
        # Vietnamese
        ("sửa lại", 8),
        ("cải thiện", 8),
        ("tối ưu", 8),
        ("fix lỗi", 10),
        ("fix bug", 10),
        # English
        ("refactor", 10),
        ("improve", 6),
        ("optimize", 8),
        ("fix bug", 10),
        ("fix this", 6),
        ("clean up", 6),
    ],
}

# Negative keywords: reduce score for certain intents
NEGATIVE_KEYWORDS: dict[str, list[tuple[str, int]]] = {
    "code_review": [
        ("giải thích", -8),  # "giải thích" should not be code_review
        ("explain", -8),
        ("cấu trúc", -5),
    ],
}


def _normalize_text(text: str) -> str:
    """Normalize text for keyword matching."""
    # Unicode normalize
    text = unicodedata.normalize("NFC", text)
    # Lowercase
    text = text.lower()
    # Remove extra whitespace
    text = " ".join(text.split())
    return text


def _calculate_intent_scores(text: str) -> dict[str, int]:
    """Calculate scores for each intent based on keyword matches."""
    scores: dict[str, int] = {intent: 0 for intent in INTENT_KEYWORDS}

    text_lower = text.lower()

    # Positive keywords
    for intent, keywords in INTENT_KEYWORDS.items():
        for keyword, weight in keywords:
            if keyword.lower() in text_lower:
                scores[intent] += weight
                logger.debug("  +%d for '%s' -> %s", weight, keyword, intent)

    # Negative keywords
    for intent, keywords in NEGATIVE_KEYWORDS.items():
        for keyword, weight in keywords:
            if keyword.lower() in text_lower:
                scores[intent] += weight  # weight is negative
                logger.debug("  %d for '%s' -> %s", weight, keyword, intent)

    return scores


def classify_intent(state: AgentState) -> dict:
    """Classify the user's intent using keyword scoring.

    MUST check is_tool_result_turn first — before any scoring.
    Returns intent with highest score. Default: code_gen if all scores are 0.
    """
    messages = state.get("messages", [])
    if not messages:
        return {"intent": "code_gen", "is_tool_result_turn": False}

    # FIRST CHECK: detect Turn 2 (tool result messages present)
    has_tool_result = any(
        isinstance(m, ToolMessage) or
        (hasattr(m, "type") and m.type == "tool") or
        (isinstance(m, dict) and m.get("role") == "tool")
        for m in messages
    )
    if has_tool_result:
        # Preserve intent from prior state; do not re-classify
        return {
            "intent": state.get("intent", "code_gen"),
            "is_tool_result_turn": True,
        }

    # Extract text from last message
    last_msg = messages[-1]
    if hasattr(last_msg, "content"):
        text = last_msg.content
    elif isinstance(last_msg, dict):
        text = last_msg.get("content", "")
    else:
        text = str(last_msg)

    # Handle list content (multimodal)
    if isinstance(text, list):
        text = " ".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in text
        )

    # Normalize and take first 500 chars
    text = _normalize_text(text or "")
    text_for_classification = text[:500]

    logger.info("classify_intent: input='%s'", text_for_classification[:120].replace('\n', ' '))

    # Calculate scores
    scores = _calculate_intent_scores(text_for_classification)

    # Find best intent
    best_intent = max(scores, key=lambda k: scores[k])
    best_score = scores[best_intent]

    # Log all scores for debugging
    scores_str = ", ".join(f"{k}={v}" for k, v in sorted(scores.items(), key=lambda x: -x[1]) if v != 0)
    logger.info("classify_intent: scores=[%s]", scores_str or "all zero")

    # Threshold: need at least score > 0 to not default to code_gen
    if best_score <= 0:
        logger.info("classify_intent: no strong signal, defaulting to code_gen")
        return {"intent": "code_gen", "is_tool_result_turn": False}

    logger.info("classify_intent: selected '%s' (score=%d)", best_intent, best_score)
    return {"intent": best_intent, "is_tool_result_turn": False}
