"""Node: classify_intent — Section 8.

Priority-ordered keyword and regex patterns.
Supports both Vietnamese and English queries.
"""

from __future__ import annotations

import re

from langchain_core.messages import ToolMessage

from server.agent.state import AgentState

# Section 8: Intent priority order (must be preserved)
# Pre-compiled for performance — patterns are checked on every request.
INTENT_PATTERNS: list[tuple[str, list[re.Pattern]]] = [
    ("unit_test", [
        re.compile(r"viết\s*test"), re.compile(r"write\s*test"), re.compile(r"unit\s*test"),
        re.compile(r"generate\s*test"), re.compile(r"tạo\s*test"),
    ]),
    ("structural_analysis", [
        re.compile(r"phân\s*tích\s*cấu\s*trúc"), re.compile(r"analyze\s*architecture"),
        re.compile(r"circular\s*dep"), re.compile(r"overview"), re.compile(r"project\s*structure"),
        re.compile(r"toàn\s*bộ\s*project"),
    ]),
    ("explain", [
        re.compile(r"giải\s*thích"), re.compile(r"explain"), re.compile(r"how\s*does"),
        re.compile(r"hoạt\s*động\s*như\s*thế\s*nào"), re.compile(r"làm\s*gì"), re.compile(r"what\s*does"),
    ]),
    ("search", [
        re.compile(r"tìm"), re.compile(r"find"), re.compile(r"search"),
        re.compile(r"where\s*is"), re.compile(r"ở\s*đâu"),
    ]),
    ("refine", [
        re.compile(r"sửa\s*lại"), re.compile(r"refactor"), re.compile(r"improve"),
        re.compile(r"fix"), re.compile(r"optimize"), re.compile(r"cải\s*thiện"),
    ]),
]


def classify_intent(state: AgentState) -> dict:
    """Classify the user's intent from the last message.

    MUST check is_tool_result_turn first — before any pattern matching.
    Default: code_gen if none of the patterns match.
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

    last_msg = messages[-1]
    if hasattr(last_msg, "content"):
        text = last_msg.content
    elif isinstance(last_msg, dict):
        text = last_msg.get("content", "")
    else:
        text = str(last_msg)

    text_lower = text.lower()

    for intent, patterns in INTENT_PATTERNS:
        for pattern in patterns:
            if pattern.search(text_lower):
                return {"intent": intent, "is_tool_result_turn": False}

    return {"intent": "code_gen", "is_tool_result_turn": False}
