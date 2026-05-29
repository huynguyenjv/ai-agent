"""Prompt injection defense and input sanitization.

Security module to protect against:
- Prompt injection attacks
- Jailbreak attempts
- Malicious input patterns
"""

from __future__ import annotations

import logging
import re
from typing import NamedTuple

logger = logging.getLogger("server.utils.sanitize")

# Maximum user input length (characters)
MAX_USER_INPUT_LENGTH = 32000

# System prompt markers that could be used for injection
SYSTEM_MARKERS = [
    "<|system|>", "<|assistant|>", "<|user|>", "<|end|>",
    "<<SYS>>", "<</SYS>>",
    "[INST]", "[/INST]",
    "<s>", "</s>",
    "### System:", "### Assistant:", "### Human:",
    "SYSTEM:", "ASSISTANT:", "USER:",
]

# Jailbreak detection patterns
JAILBREAK_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"ignore\s+(all\s+)?prior\s+instructions",
    r"disregard\s+(all\s+)?previous",
    r"forget\s+(all\s+)?previous",
    r"pretend\s+(you\s+)?(are|to\s+be)",
    r"act\s+as\s+if",
    r"you\s+are\s+now\s+",
    r"new\s+persona",
    r"bypass\s+(safety|security|filter)",
    r"disable\s+(safety|security|filter)",
    r"jailbreak",
    r"dan\s+mode",
    r"developer\s+mode\s+enabled",
    r"do\s+anything\s+now",
]

# Compiled patterns for performance
_jailbreak_re = [re.compile(p, re.IGNORECASE) for p in JAILBREAK_PATTERNS]


class SanitizeResult(NamedTuple):
    """Result of sanitization."""
    text: str
    was_truncated: bool
    markers_removed: int
    jailbreak_detected: bool


def sanitize_user_input(text: str) -> SanitizeResult:
    """Sanitize user input to prevent prompt injection.

    Args:
        text: Raw user input

    Returns:
        SanitizeResult with cleaned text and metadata
    """
    if not text:
        return SanitizeResult("", False, 0, False)

    original_len = len(text)
    markers_removed = 0

    # Remove system prompt markers
    for marker in SYSTEM_MARKERS:
        count = text.count(marker)
        if count > 0:
            markers_removed += count
            text = text.replace(marker, "")

    # Escape code block markers that could inject system context
    text = text.replace("```system", "```text")
    text = text.replace("```assistant", "```text")

    # Detect jailbreak attempts
    jailbreak = detect_jailbreak(text)
    if jailbreak:
        logger.warning("Jailbreak pattern detected in user input")

    # Truncate if too long
    was_truncated = len(text) > MAX_USER_INPUT_LENGTH
    if was_truncated:
        text = text[:MAX_USER_INPUT_LENGTH]
        logger.info("User input truncated from %d to %d chars", original_len, MAX_USER_INPUT_LENGTH)

    if markers_removed > 0:
        logger.warning("Removed %d prompt injection markers from user input", markers_removed)

    return SanitizeResult(
        text=text,
        was_truncated=was_truncated,
        markers_removed=markers_removed,
        jailbreak_detected=jailbreak,
    )


def detect_jailbreak(text: str) -> bool:
    """Detect common jailbreak patterns in text.

    Args:
        text: Text to analyze

    Returns:
        True if jailbreak pattern detected
    """
    if not text:
        return False

    for pattern in _jailbreak_re:
        if pattern.search(text):
            return True

    return False


def sanitize_tool_output(output: str, max_length: int = 50000) -> str:
    """Sanitize tool output before including in context.

    Prevents tool output from containing injection attempts.

    Args:
        output: Raw tool output
        max_length: Maximum allowed length

    Returns:
        Sanitized output
    """
    if not output:
        return ""

    # Remove any system markers that might be in output
    for marker in SYSTEM_MARKERS:
        output = output.replace(marker, "")

    # Truncate if needed
    if len(output) > max_length:
        output = output[:max_length] + f"\n[truncated, {len(output)} bytes total]"

    return output


def escape_for_prompt(text: str) -> str:
    """Escape text for safe inclusion in a prompt.

    Use this when including user content in prompt templates.

    Args:
        text: Text to escape

    Returns:
        Escaped text safe for prompt inclusion
    """
    if not text:
        return ""

    # Wrap in clear delimiters
    return f"<user_content>\n{text}\n</user_content>"
