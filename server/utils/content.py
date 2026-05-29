"""Content normalization utilities."""

from __future__ import annotations


def normalize_content(content) -> str:
    """Normalize content from various formats to plain string.

    Handles:
    - None → ""
    - str → str
    - list[{"type": "text", "text": "..."}] → joined text (multimodal format)
    - list[str] → joined strings
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def estimate_tokens(text: str, divisor: int = 4) -> int:
    """Estimate token count from text.

    Args:
        text: Input text
        divisor: Chars per token (default 4 for code/English)

    Returns:
        Estimated token count
    """
    if not text:
        return 0
    return len(text) // divisor
