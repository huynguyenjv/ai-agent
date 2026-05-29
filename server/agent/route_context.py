"""Node: route_context — Section 8 + Section 11 (5-Gate Decision Flow).

Implements the 5-Gate context decision flow.
Gates are evaluated in strict order. First gate that fires determines strategy.
"""

from __future__ import annotations

import re
import threading

from server.agent.rules_loader import get_rules_loader
from server.agent.state import AgentState

# GitLab MR URL: https://gitlab.xxx/group/project/-/merge_requests/123
_GITLAB_MR_URL = re.compile(
    r"https?://([\w.-]+)/([\w./-]+?)/-/merge_requests/(\d+)",
    re.IGNORECASE,
)

# Gate 1: file mention regexes are compiled lazily from rules.yaml so they
# stay in sync with classify_intent's keyword list. Cache invalidates when
# the underlying rules dict identity changes (RulesLoader replaces it on reload).
_FILE_PATTERN_CACHE: dict = {"rules_id": None, "patterns": []}
_FILE_PATTERN_LOCK = threading.Lock()


def _get_file_patterns() -> list[re.Pattern]:
    loader = get_rules_loader()
    rules = loader.get_rules()
    rules_id = id(rules)

    with _FILE_PATTERN_LOCK:
        if _FILE_PATTERN_CACHE["rules_id"] == rules_id and _FILE_PATTERN_CACHE["patterns"]:
            return _FILE_PATTERN_CACHE["patterns"]

        extensions = loader.get_file_extensions() or [".java", ".py", ".go", ".ts"]
        suffixes = loader.get_file_suffixes() or ["Service", "Controller", "Repository"]

        ext_alt = "|".join(re.escape(ext.lstrip(".")) for ext in extensions)
        suffix_alt = "|".join(re.escape(s) for s in suffixes)

        patterns = [
            re.compile(rf"\b(\w+\.(?:{ext_alt}))\b", re.IGNORECASE),
            re.compile(rf"@(\w+\.(?:{ext_alt}))\b", re.IGNORECASE),
            re.compile(rf"\b([A-Z][a-zA-Z0-9]+(?:{suffix_alt}))\b"),
        ]

        _FILE_PATTERN_CACHE["rules_id"] = rules_id
        _FILE_PATTERN_CACHE["patterns"] = patterns
        return patterns

# Gate 1: Deictic references
_DEICTIC_PATTERNS = re.compile(
    r"\b(?:file\s*này|class\s*này|this\s*file|this\s*class|đây|nó|it|here)\b",
    re.IGNORECASE,
)

# Gate 2: Freshness/temporal keywords - loaded from rules.yaml
_FRESHNESS_CACHE: dict = {"rules_id": None, "pattern": None}
_FRESHNESS_LOCK = threading.Lock()


def _get_freshness_pattern() -> re.Pattern:
    """Get freshness pattern compiled from rules.yaml keywords."""
    loader = get_rules_loader()
    rules = loader.get_rules()
    rules_id = id(rules)

    with _FRESHNESS_LOCK:
        if _FRESHNESS_CACHE["rules_id"] == rules_id and _FRESHNESS_CACHE["pattern"]:
            return _FRESHNESS_CACHE["pattern"]

        keywords = loader.get_freshness_keywords()
        if not keywords:
            keywords = ["recently", "just", "latest", "current"]

        escaped = [re.escape(kw).replace(r"\ ", r"\s*") for kw in keywords]
        pattern = re.compile(rf"(?:{'|'.join(escaped)})", re.IGNORECASE)

        _FRESHNESS_CACHE["rules_id"] = rules_id
        _FRESHNESS_CACHE["pattern"] = pattern
        return pattern

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

    # --- Code review: detect PR URL or file mode ---
    if state.get("intent") == "code_review":
        mr_match = _GITLAB_MR_URL.search(text)
        if mr_match:
            host, repo, pr_id = mr_match.group(1), mr_match.group(2), int(mr_match.group(3))
            pr_ctx = dict(state.get("pr_context") or {})
            pr_ctx.update({"provider": "gitlab", "host": host, "repo": repo, "pr_id": pr_id})
            return {
                "mentioned_files": [],
                "force_reindex": False,
                "freshness_signal": False,
                "review_mode": "pr",
                "pr_context": pr_ctx,
            }
        # No PR URL → file mode (Continue sent full file in code fence)
        return {
            "mentioned_files": [],
            "force_reindex": False,
            "freshness_signal": False,
            "review_mode": state.get("review_mode") or "file",
        }

    # --- Gate 1: Explicit File Mention ---
    mentioned_files: list[str] = []

    for pattern in _get_file_patterns():
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
    if _get_freshness_pattern().search(text):
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
