"""Node: classify_intent — LLM-based classifier with hot-reload rules.

Primary path: LLM reads rules from rules.yaml and classifies intent.
Fallback path: Python keyword matching if LLM fails or confidence < threshold.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.messages import ToolMessage

from server.agent.rules_loader import RulesLoader, get_rules_loader
from server.agent.state import AgentState
from server.utils.sanitize import sanitize_user_input

logger = logging.getLogger("server.classify_intent")


# =============================================================================
# Python Fallback Classifier
# =============================================================================

def _extract_file_target(text: str, rules_loader: RulesLoader, active_file: str | None = None) -> str | None:
    """Extract file target from text.

    Detects:
    - Filenames with known extensions (.java, .py, etc.)
    - Class names ending with known suffixes (Service, Controller, etc.)
    - @mention syntax (@ClassName)
    - Deictic references (file này, this file) -> uses active_file
    """
    text_lower = text.lower()

    # 1. Check for @mentions: @UserService, @OrderController.java
    at_mention = re.search(r"@(\w+(?:\.\w+)?)", text)
    if at_mention:
        return at_mention.group(1)

    # 2. Check for filenames with extensions
    extensions = rules_loader.get_file_extensions()
    ext_pattern = r"\b(\w+(?:" + "|".join(re.escape(ext) for ext in extensions) + r"))\b"
    ext_match = re.search(ext_pattern, text, re.IGNORECASE)
    if ext_match:
        return ext_match.group(1)

    # 3. Check for class names ending with known suffixes
    suffixes = rules_loader.get_file_suffixes()
    for suffix in suffixes:
        # Match PascalCase names ending with suffix
        pattern = rf"\b([A-Z][a-zA-Z0-9]*{re.escape(suffix)})\b"
        match = re.search(pattern, text)
        if match:
            return match.group(1)

    # 4. Check for deictic references (only if active_file is present)
    if active_file:
        deictic_patterns = [
            r"\bfile\s*này\b", r"\bthis\s*file\b", r"\bclass\s*này\b",
            r"\bthis\s*class\b", r"\bnó\b", r"\bit\b", r"\bhere\b", r"\bđây\b"
        ]
        for pattern in deictic_patterns:
            if re.search(pattern, text_lower):
                return active_file

    return None


def _detect_freshness_signal(text: str, rules_loader: RulesLoader) -> bool:
    """Check if text contains freshness keywords."""
    text_lower = text.lower()
    for keyword in rules_loader.get_freshness_keywords():
        if keyword.lower() in text_lower:
            return True
    return False


def _python_classify(
    text: str,
    rules_loader: RulesLoader,
    active_file: str | None = None,
) -> dict[str, Any]:
    """Python fallback classifier using keyword matching.

    This is synchronous and must never fail - it's the safety net.
    """
    text_lower = text.lower()
    intents = rules_loader.get_intents()  # Already sorted by priority

    # Find matching intent
    matched_intent = "code_gen"
    matched_keyword = ""
    confidence = 0.4

    for intent in intents:
        name = intent.get("name", "")
        keywords_vi = intent.get("keywords_vi", [])
        keywords_en = intent.get("keywords_en", [])

        for keyword in keywords_vi + keywords_en:
            if keyword.lower() in text_lower:
                matched_intent = name
                matched_keyword = keyword
                confidence = 0.6
                break

        if matched_keyword:
            break

    # Detect file target and freshness
    file_target = _extract_file_target(text, rules_loader, active_file)
    freshness_signal = _detect_freshness_signal(text, rules_loader)

    reasoning = f"Python fallback: matched keyword '{matched_keyword}' for intent '{matched_intent}'" if matched_keyword else "Python fallback: no keyword match, using default code_gen"

    return {
        "intent": matched_intent,
        "sub_intent": "",
        "file_target": file_target,
        "freshness_signal": freshness_signal,
        "is_tool_result_turn": False,
        "confidence": confidence,
        "reasoning": reasoning,
    }


# =============================================================================
# LLM Classifier
# =============================================================================

def _build_llm_prompt(text: str, rules_loader: RulesLoader) -> str:
    """Build classification prompt dynamically from rules."""
    intents = rules_loader.get_intents()
    file_suffixes = rules_loader.get_file_suffixes()
    freshness_keywords = rules_loader.get_freshness_keywords()

    # Build intents section with clearer formatting
    intents_section = []
    for intent in intents:
        keywords_vi = intent.get("keywords_vi", [])[:5]  # Limit for prompt size
        keywords_en = intent.get("keywords_en", [])[:5]
        intents_section.append(
            f"  - **{intent['name']}** (P{intent.get('priority', 99)}): {intent.get('description', '')}\n"
            f"    VI: {', '.join(keywords_vi)}\n"
            f"    EN: {', '.join(keywords_en)}"
        )

    prompt = f"""You are a senior intent classifier for an AI coding assistant.
Your job is to accurately route developer queries to the correct handler.

## AVAILABLE INTENTS (ordered by priority - evaluate from top to bottom):

{chr(10).join(intents_section)}

## CLASSIFICATION RULES:

1. **Match by semantics, not just keywords** - understand what the user wants to achieve
2. **Priority matters** - if multiple intents could match, pick the higher priority one
3. **Be confident when clear** - if the intent is obvious, confidence should be 0.85-0.95
4. **Be cautious when ambiguous** - if unclear, confidence should be 0.5-0.7
5. **Default to code_gen** - when nothing matches, use code_gen with low confidence

## FILE DETECTION:

Extract file_target if the query mentions:
- Filename with extension: OrderService.java, main.py, config.ts
- Class name with suffix: {', '.join(file_suffixes[:10])}
- @mention syntax: @UserController

## FRESHNESS DETECTION:

Set freshness_signal=true if query contains temporal keywords like:
{', '.join(freshness_keywords[:8])}

## EXAMPLES:

Query: "viết unit test cho UserService"
→ {{"intent": "unit_test", "file_target": "UserService", "confidence": 0.92}}

Query: "tại sao OrderController bị lỗi null pointer"
→ {{"intent": "debug", "file_target": "OrderController", "confidence": 0.88}}

Query: "explain how the payment flow works"
→ {{"intent": "explain", "file_target": null, "confidence": 0.85}}

Query: "refactor this code to be cleaner"
→ {{"intent": "refine", "file_target": null, "confidence": 0.80}}

## OUTPUT:

Return ONLY valid JSON (no markdown, no explanation):
{{"intent": "<intent_name>", "sub_intent": "", "file_target": "<filename or null>", "freshness_signal": <true|false>, "confidence": <0.0-1.0>, "reasoning": "<brief explanation>"}}

## QUERY TO CLASSIFY:

{text}"""

    return prompt


async def _llm_classify(
    text: str,
    rules_loader: RulesLoader,
    vllm_client,
    model: str,
) -> dict[str, Any]:
    """LLM-based classifier. Raises on failure so caller can trigger fallback."""
    prompt = _build_llm_prompt(text, rules_loader)

    response = await vllm_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.0,
    )

    content = response.choices[0].message.content.strip()

    # Try to extract JSON from response (might have markdown wrapping)
    # Use non-greedy match to get first JSON object only
    json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", content)
    if not json_match:
        raise ValueError(f"No JSON found in LLM response: {content[:100]}")

    result = json.loads(json_match.group())

    # Validate required fields
    required_fields = ["intent", "confidence"]
    for field in required_fields:
        if field not in result:
            raise ValueError(f"LLM output missing required field: {field}")

    # Validate intent is known
    valid_intents = rules_loader.get_intent_names()
    if result["intent"] not in valid_intents:
        raise ValueError(f"Unknown intent from LLM: {result['intent']}")

    # Validate confidence is float 0-1
    confidence = float(result["confidence"])
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"Invalid confidence value: {confidence}")

    # Ensure all fields present with defaults
    return {
        "intent": result["intent"],
        "sub_intent": result.get("sub_intent", ""),
        "file_target": result.get("file_target"),
        "freshness_signal": bool(result.get("freshness_signal", False)),
        "is_tool_result_turn": False,
        "confidence": confidence,
        "reasoning": result.get("reasoning", "LLM classification"),
    }


# =============================================================================
# Main Node Function
# =============================================================================

async def classify_intent(
    state: AgentState,
    vllm_client=None,
    model: str = "",
) -> dict[str, Any]:
    """Classify user intent using LLM with Python fallback.

    This is a LangGraph node. Returns dict to merge into AgentState.

    Args:
        state: Current agent state
        vllm_client: Optional vLLM client for LLM classification
        model: Model name for vLLM

    Returns:
        Dict with intent, file_target, freshness_signal, etc.
    """
    messages = state.get("messages", [])

    # Step 1: Check for Turn 2 (tool result messages present)
    has_tool_result = any(
        isinstance(m, ToolMessage) or
        (hasattr(m, "type") and m.type == "tool") or
        (isinstance(m, dict) and m.get("role") == "tool")
        for m in messages
    )

    if has_tool_result:
        logger.info("classify_intent: Turn 2 detected, preserving prior intent")
        return {
            "intent": state.get("intent", "code_gen"),
            "sub_intent": state.get("sub_intent", ""),
            "file_target": state.get("file_target"),
            "freshness_signal": state.get("freshness_signal", False),
            "is_tool_result_turn": True,
            "confidence": 1.0,
            "reasoning": "Turn 2: tool result messages detected, preserving prior intent",
        }

    # Step 2: Extract last user message text
    text = ""
    for msg in reversed(messages):
        msg_role = None
        msg_content = None

        if hasattr(msg, "type"):
            msg_role = msg.type
            msg_content = getattr(msg, "content", "")
        elif isinstance(msg, dict):
            msg_role = msg.get("role", "")
            msg_content = msg.get("content", "")

        if msg_role in ("human", "user"):
            if isinstance(msg_content, list):
                # Handle multimodal content
                text = " ".join(
                    item.get("text", "") if isinstance(item, dict) else str(item)
                    for item in msg_content
                )
            else:
                text = msg_content or ""
            break

    if not text:
        logger.warning("classify_intent: no user message found, defaulting to code_gen")
        return {
            "intent": "code_gen",
            "sub_intent": "",
            "file_target": None,
            "freshness_signal": False,
            "is_tool_result_turn": False,
            "confidence": 0.3,
            "reasoning": "No user message found, using default",
        }

    # Step 2.5: Sanitize user input (prompt injection defense)
    sanitize_result = sanitize_user_input(text)
    text = sanitize_result.text
    if sanitize_result.jailbreak_detected:
        logger.warning("classify_intent: jailbreak pattern detected, proceeding with caution")

    # Step 3: Load rules
    rules_loader = get_rules_loader()
    active_file = state.get("active_file")

    # Log input for debugging
    logger.info("classify_intent: input='%s'", text[:120].replace('\n', ' '))

    # Step 4: Try LLM classification
    if vllm_client is not None and model:
        try:
            result = await _llm_classify(text, rules_loader, vllm_client, model)

            if result["confidence"] >= rules_loader.get_confidence_threshold():
                logger.info(
                    "classify_intent: LLM result intent=%s confidence=%.2f",
                    result["intent"], result["confidence"]
                )
                return result

            logger.info(
                "classify_intent: LLM confidence %.2f < threshold %.2f, using fallback",
                result["confidence"], rules_loader.get_confidence_threshold()
            )
            # Fall through to Python fallback

        except Exception as e:
            logger.warning("classify_intent: LLM failed (%s), using Python fallback", e)
            # Fall through to Python fallback

    # Step 5: Python fallback
    result = _python_classify(text, rules_loader, active_file)
    logger.info(
        "classify_intent: Python fallback intent=%s confidence=%.2f",
        result["intent"], result["confidence"]
    )
    return result
