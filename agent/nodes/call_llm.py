"""
Call LLM node — sends prompts to vLLM and extracts Java code.

Wraps: vllm/client.py → VLLMClient.generate()
"""

from __future__ import annotations

import re

import structlog

logger = structlog.get_logger()


def call_llm_node(state: dict, *, vllm_client) -> dict:
    """Call vLLM with system + user prompts, extract Java code.

    Args:
        state: UnitTestState dict.
        vllm_client: VLLMClient instance.

    Returns:
        State updates: llm_output, test_code, tokens_used, retry_count.
    """
    system_prompt = state.get("system_prompt", "")
    user_prompt = state.get("user_prompt", "")
    retry_count = state.get("retry_count", 0)

    logger.info(
        "call_llm_node: generating",
        retry_count=retry_count,
        system_len=len(system_prompt),
        user_len=len(user_prompt),
    )

    try:
        response = vllm_client.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
    except Exception as e:
        logger.error("call_llm_node: vLLM call failed", error=str(e))
        return {
            "error": f"LLM generation failed: {e}",
            "retry_count": retry_count + 1,
        }

    if not response.success:
        return {
            "error": f"LLM generation failed: {response.error}",
            "retry_count": retry_count + 1,
        }

    llm_output = response.content
    test_code = _extract_code(llm_output)
    tokens_used = state.get("tokens_used", 0) + (response.tokens_used or 0)

    logger.info(
        "call_llm_node: done",
        tokens=response.tokens_used,
        code_len=len(test_code),
    )

    return {
        "llm_output": llm_output,
        "test_code": test_code,
        "tokens_used": tokens_used,
        "retry_count": retry_count + 1,
    }


def _extract_code(response: str) -> str:
    """Extract Java code from LLM response.

    Handles:
      1. ```java ... ``` fenced blocks
      2. ``` ... ``` generic fenced blocks
      3. Raw code (if no fences found)

    Mirrors AgentOrchestrator._extract_code().
    """
    if not response:
        return ""

    # Try ```java blocks first
    java_match = re.search(r'```java\s*\n(.*?)```', response, re.DOTALL)
    if java_match:
        return java_match.group(1).strip()

    # Try generic code blocks
    generic_match = re.search(r'```\s*\n(.*?)```', response, re.DOTALL)
    if generic_match:
        return generic_match.group(1).strip()

    # Fallback: return raw response (might be just code)
    return response.strip()
