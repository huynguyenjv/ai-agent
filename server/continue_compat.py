"""Continue IDE compatibility layer.

Continue sends standard OpenAI chat completion requests. It does NOT send
custom fields like active_file or repo_path in the request body. Instead,
it injects context as code blocks in the message content:

    Generate tests for OrderService

    ```src/main/java/com/example/OrderService.java
    public class OrderService { }
    ```

This module extracts structured information from Continue's message format.
"""

from __future__ import annotations

import re

# Pattern: ```path/to/file.ext or ```language path/to/file.ext
_CODE_FENCE_FILE = re.compile(
    r"```(?:[a-zA-Z]*\s+)?(\S+\.(?:java|go|py|ts|tsx|js|jsx|cs|tf|hcl|kt|rs|rb|php|swift|c|cpp|h))\b",
    re.IGNORECASE,
)

# Pattern for @file mentions in Continue
_AT_MENTION = re.compile(
    r"@(\S+\.(?:java|go|py|ts|tsx|js|jsx|cs|tf|hcl))\b",
    re.IGNORECASE,
)


def extract_active_file(messages: list, explicit_active_file: str | None = None) -> str | None:
    """Extract the active file from Continue's message format.

    Priority:
    1. Explicit active_file field (if Continue ever sends it)
    2. First code fence with a file path in the last user message
    3. First @mention in the last user message
    4. None
    """
    if explicit_active_file:
        return explicit_active_file

    if not messages:
        return None

    last_msg = messages[-1]
    if hasattr(last_msg, "content"):
        text = last_msg.content
    elif isinstance(last_msg, dict):
        text = last_msg.get("content", "")
    else:
        text = str(last_msg)

    # Check code fences first (most reliable — Continue puts the file path here)
    match = _CODE_FENCE_FILE.search(text)
    if match:
        return match.group(1)

    # Check @mentions
    match = _AT_MENTION.search(text)
    if match:
        return match.group(1)

    return None


def extract_code_blocks(text: str) -> list[dict]:
    """Extract code blocks from Continue's message, including file paths.

    Returns list of {"file_path": str|None, "lang": str, "content": str}.
    """
    blocks = []
    # Match ```lang path\n...content...\n``` or ```path\n...content...\n```
    pattern = re.compile(
        r"```(\w*)\s*(\S+\.(?:java|go|py|ts|tsx|js|jsx|cs|tf|hcl|kt|rs|rb))?\s*\n(.*?)```",
        re.DOTALL | re.IGNORECASE,
    )
    for m in pattern.finditer(text):
        lang = m.group(1) or ""
        file_path = m.group(2)
        content = m.group(3).strip()
        blocks.append({"file_path": file_path, "lang": lang, "content": content})

    return blocks
