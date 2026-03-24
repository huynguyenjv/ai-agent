"""Node: route_context — Section 8 + Section 11 (5-Gate Decision Flow).

Implements the 5-Gate context decision flow.
Gates are evaluated in strict order. First gate that fires determines strategy.
"""

from __future__ import annotations

import re

from server.agent.state import AgentState

# Gate 1: File mention patterns
_FILE_PATTERNS = [
    # Direct filename: XxxService.java, handler.go, main.tf, etc.
    re.compile(r"\b(\w+\.(?:java|go|py|ts|tsx|js|jsx|cs|tf|hcl))\b", re.IGNORECASE),
    # @mention syntax: @UserService.java
    re.compile(r"@(\w+\.(?:java|go|py|ts|tsx|js|jsx|cs|tf|hcl))\b", re.IGNORECASE),
    # Class name patterns: UserService, OrderController
    re.compile(r"\b([A-Z][a-zA-Z]+(?:Service|Controller|Repository|Handler|Manager|Factory|Provider|Adapter|Config|Entity|Model|Dto))\b"),
]

# Gate 1: Deictic references
_DEICTIC_PATTERNS = re.compile(
    r"\b(?:file\s*này|class\s*này|this\s*file|this\s*class|đây|nó|it|here)\b",
    re.IGNORECASE,
)

# Gate 2: Freshness/temporal keywords
_FRESHNESS_PATTERNS = re.compile(
    r"(?:vừa\s*thêm|vừa\s*sửa|vừa\s*commit|just\s*added|just\s*changed|"
    r"hiện\s*tại\s*đang\s*có\s*bug|lỗi\s*đang\s*xảy\s*ra|recent\s*change)",
    re.IGNORECASE,
)

# Gate 3: Volatile data type keywords
_VOLATILE_PATTERNS = re.compile(
    r"\b(?:git\s*diff|runtime\s*log|live\s*metric|error\s*stack\s*trace|"
    r"running\s*process)\b",
    re.IGNORECASE,
)


def route_context(state: AgentState) -> dict:
    """Implement the 5-Gate decision flow (Section 11).

    Returns updates to mentioned_files, force_reindex, freshness_signal.
    """
    messages = state.get("messages", [])
    active_file = state.get("active_file")

    if not messages:
        return {
            "mentioned_files": [],
            "force_reindex": False,
            "freshness_signal": False,
        }

    last_msg = messages[-1]
    if hasattr(last_msg, "content"):
        text = last_msg.content
    elif isinstance(last_msg, dict):
        text = last_msg.get("content", "")
    else:
        text = str(last_msg)

    # --- Gate 1: Explicit File Mention ---
    mentioned_files: list[str] = []

    for pattern in _FILE_PATTERNS:
        for match in pattern.finditer(text):
            mentioned_files.append(match.group(1))

    # Deictic reference to active file
    if active_file and _DEICTIC_PATTERNS.search(text):
        if active_file not in mentioned_files:
            mentioned_files.append(active_file)

    if mentioned_files:
        return {
            "mentioned_files": mentioned_files,
            "force_reindex": True,
            "freshness_signal": False,
        }

    # --- Gate 2: Freshness Force Signal ---
    if _FRESHNESS_PATTERNS.search(text):
        return {
            "mentioned_files": [],
            "force_reindex": True,
            "freshness_signal": True,
        }

    # --- Gate 3: Volatile Data Type (Section 11) ---
    if _VOLATILE_PATTERNS.search(text):
        return {
            "mentioned_files": [],
            "force_reindex": False,
            "freshness_signal": False,
            "volatile_rejected": True,
        }

    # --- Gate 4 & 5: RAG lookup (handled in rag_search node) ---
    return {
        "mentioned_files": [],
        "force_reindex": False,
        "freshness_signal": False,
    }
