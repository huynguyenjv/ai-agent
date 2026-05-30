"""Conversation summarization for context window management.

Summarizes long conversations to preserve important context
while fitting within token limits.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("server.agent.summarize")

# Threshold for triggering summarization
MESSAGE_THRESHOLD = 10
TOKEN_THRESHOLD = 6000

SUMMARY_PROMPT = """Summarize this conversation concisely, preserving:
1. User's main goal and requirements
2. Key decisions and agreements made
3. Important files/code mentioned
4. Current state of the task
5. Any errors or issues encountered

Conversation:
{conversation}

Provide a concise summary (max {max_tokens} tokens):"""


async def summarize_conversation(
    messages: list,
    vllm_client,
    model: str,
    max_tokens: int = 500,
) -> str:
    """Summarize long conversation to fit context window.

    Args:
        messages: Conversation messages
        vllm_client: vLLM client
        model: Model name
        max_tokens: Max tokens for summary

    Returns:
        Summary string
    """
    if not messages:
        return ""

    # Format conversation for summarization
    conversation_text = _format_messages_for_summary(messages)

    try:
        response = await vllm_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a conversation summarizer. Be concise and preserve key technical details."
                },
                {
                    "role": "user",
                    "content": SUMMARY_PROMPT.format(
                        conversation=conversation_text,
                        max_tokens=max_tokens,
                    )
                },
            ],
            temperature=0.3,
            max_tokens=max_tokens,
        )

        summary = response.choices[0].message.content or ""
        logger.info("summarize: generated %d char summary from %d messages",
                    len(summary), len(messages))
        return summary.strip()

    except Exception as e:
        logger.error("summarize: failed to generate summary: %s", e)
        # Fallback: return truncated recent messages
        return _fallback_summary(messages)


def _format_messages_for_summary(messages: list) -> str:
    """Format messages for summarization prompt.

    Args:
        messages: List of messages

    Returns:
        Formatted conversation string
    """
    parts = []

    for msg in messages:
        role = _get_role(msg)
        content = _get_content(msg)

        if not content:
            continue

        # Truncate very long messages
        if len(content) > 1000:
            content = content[:1000] + "..."

        parts.append(f"{role.upper()}: {content}")

    return "\n\n".join(parts)


def _get_role(msg) -> str:
    """Extract role from message."""
    if hasattr(msg, "type"):
        return msg.type
    elif isinstance(msg, dict):
        return msg.get("role", "unknown")
    return "unknown"


def _get_content(msg) -> str:
    """Extract content from message."""
    if hasattr(msg, "content"):
        content = msg.content
    elif isinstance(msg, dict):
        content = msg.get("content", "")
    else:
        return ""

    if isinstance(content, list):
        # Multimodal content
        return " ".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content) if content else ""


def _fallback_summary(messages: list) -> str:
    """Generate simple fallback summary without LLM.

    Args:
        messages: Conversation messages

    Returns:
        Simple summary
    """
    # Extract first user message as goal
    first_user = ""
    for msg in messages:
        if _get_role(msg) in ("user", "human"):
            first_user = _get_content(msg)[:200]
            break

    # Count message types
    user_count = sum(1 for m in messages if _get_role(m) in ("user", "human"))
    assistant_count = sum(1 for m in messages if _get_role(m) in ("assistant", "ai"))

    return f"Conversation with {user_count} user messages and {assistant_count} assistant responses. Initial request: {first_user}"


def should_summarize(messages: list, token_estimate: int = 0) -> bool:
    """Check if conversation should be summarized.

    Args:
        messages: Conversation messages
        token_estimate: Estimated token count

    Returns:
        True if summarization recommended
    """
    if len(messages) > MESSAGE_THRESHOLD:
        return True

    if token_estimate > TOKEN_THRESHOLD:
        return True

    return False


def truncate_with_summary(
    messages: list,
    summary: str,
    keep_recent: int = 4,
) -> list:
    """Replace old messages with summary, keeping recent ones.

    Args:
        messages: Original messages
        summary: Conversation summary
        keep_recent: Number of recent messages to keep

    Returns:
        New message list with summary
    """
    if len(messages) <= keep_recent:
        return messages

    # Create summary message
    summary_msg = {
        "role": "system",
        "content": f"[Previous conversation summary: {summary}]"
    }

    # Keep recent messages
    recent = messages[-keep_recent:]

    return [summary_msg] + recent
