"""Node: planner — decompose complex tasks into steps.

Analyzes user request and breaks it down into atomic, verifiable steps
with dependency tracking.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from server.agent.state import AgentState

logger = logging.getLogger("server.agent.planner")

PLANNER_PROMPT = """You are a task planner for a coding assistant.

Given the user request, analyze if it's a simple or complex task and break it down if needed.

A task is COMPLEX if it requires:
- Multiple file reads/modifications
- Multi-step reasoning (read → analyze → generate → verify)
- Coordination between different operations

A task is SIMPLE if it can be done in 1-2 tool calls.

For COMPLEX tasks, decompose into atomic steps. Each step should be:
- Self-contained with clear inputs/outputs
- Verifiable (can check if done correctly)
- Properly ordered with dependencies

Output JSON format:
{
    "complexity": "simple" | "complex",
    "reasoning": "why this classification",
    "steps": [
        {
            "id": "step_1",
            "action": "read_file" | "search_symbol" | "generate_code" | "edit_file" | "run_command" | "git_operation",
            "description": "what this step does",
            "target": "file path or symbol name if applicable",
            "depends_on": ["step_ids this depends on"],
            "tools": ["vtrip_read_file", "vtrip_search_symbol", ...],
            "can_parallel": true | false
        }
    ],
    "estimated_tool_calls": 3
}

For SIMPLE tasks, return minimal plan:
{
    "complexity": "simple",
    "reasoning": "single operation task",
    "steps": [],
    "estimated_tool_calls": 1
}

Current request context:
- Intent: {intent}
- File target: {file_target}
- User message: {user_message}

Respond with JSON only, no markdown."""

# Complexity threshold - tasks with estimated tools > this are complex
COMPLEXITY_THRESHOLD = 3


async def plan_task(
    state: AgentState,
    vllm_client=None,
    model: str = "",
) -> dict[str, Any]:
    """Decompose complex task into steps.

    This is a LangGraph node that analyzes task complexity and
    creates an execution plan if needed.

    Args:
        state: Current agent state
        vllm_client: vLLM client for LLM calls
        model: Model name

    Returns:
        Dict with task_plan, plan_steps, complexity, etc.
    """
    intent = state.get("intent", "code_gen")
    file_target = state.get("file_target")

    # Extract user message
    messages = state.get("messages", [])
    user_message = ""
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type in ("human", "user"):
            user_message = getattr(msg, "content", "")
            break
        elif isinstance(msg, dict) and msg.get("role") in ("human", "user"):
            user_message = msg.get("content", "")
            break

    if not user_message:
        logger.info("planner: no user message, skipping planning")
        return {
            "complexity": "simple",
            "task_plan": None,
            "plan_steps": [],
            "current_step": 0,
        }

    # Quick heuristics for simple tasks
    if _is_obviously_simple(user_message, intent):
        logger.info("planner: task classified as simple via heuristics")
        return {
            "complexity": "simple",
            "task_plan": None,
            "plan_steps": [],
            "current_step": 0,
        }

    # Use LLM to analyze and plan
    if vllm_client is None:
        logger.warning("planner: no vllm_client, defaulting to simple")
        return {
            "complexity": "simple",
            "task_plan": None,
            "plan_steps": [],
            "current_step": 0,
        }

    prompt = PLANNER_PROMPT.format(
        intent=intent,
        file_target=file_target or "none",
        user_message=user_message[:2000],  # Limit length
    )

    try:
        response = await vllm_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a task planning assistant. Output JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=1000,
        )

        content = response.choices[0].message.content or ""
        plan = _parse_plan_response(content)

        if plan["complexity"] == "complex":
            logger.info("planner: complex task with %d steps", len(plan.get("steps", [])))
        else:
            logger.info("planner: simple task (estimated %d tool calls)",
                        plan.get("estimated_tool_calls", 1))

        return {
            "complexity": plan["complexity"],
            "task_plan": plan,
            "plan_steps": plan.get("steps", []),
            "current_step": 0,
            "planner_reasoning": plan.get("reasoning", ""),
        }

    except Exception as e:
        logger.exception("planner: LLM call failed, defaulting to simple")
        return {
            "complexity": "simple",
            "task_plan": None,
            "plan_steps": [],
            "current_step": 0,
            "planner_error": str(e),
        }


def _is_obviously_simple(text: str, intent: str) -> bool:
    """Quick heuristics to identify obviously simple tasks.

    Args:
        text: User message
        intent: Classified intent

    Returns:
        True if task is obviously simple
    """
    text_lower = text.lower()

    # Single-file operations
    simple_patterns = [
        r"^explain\s+",
        r"^what\s+(is|does|are)",
        r"^read\s+",
        r"^show\s+(me\s+)?",
        r"^summarize\s+",
        r"^list\s+",
    ]

    for pattern in simple_patterns:
        if re.match(pattern, text_lower):
            return True

    # Very short requests are usually simple
    if len(text.split()) < 10:
        return True

    # Question intents are simple
    if intent in ("explain", "question"):
        return True

    return False


def _parse_plan_response(content: str) -> dict[str, Any]:
    """Parse LLM plan response.

    Args:
        content: Raw LLM response

    Returns:
        Parsed plan dict
    """
    # Try to extract JSON from response
    content = content.strip()

    # Remove markdown code blocks if present
    if content.startswith("```"):
        lines = content.split("\n")
        json_lines = []
        in_block = False
        for line in lines:
            if line.startswith("```"):
                in_block = not in_block
                continue
            if in_block or not line.startswith("```"):
                json_lines.append(line)
        content = "\n".join(json_lines)

    try:
        plan = json.loads(content)
        # Validate required fields
        if "complexity" not in plan:
            plan["complexity"] = "simple"
        if "steps" not in plan:
            plan["steps"] = []
        return plan
    except json.JSONDecodeError:
        logger.warning("planner: failed to parse JSON response")
        return {
            "complexity": "simple",
            "reasoning": "Failed to parse plan response",
            "steps": [],
            "estimated_tool_calls": 1,
        }


def validate_plan(plan: dict[str, Any]) -> tuple[bool, str]:
    """Validate plan for issues like cycles or missing dependencies.

    Args:
        plan: Plan dict to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    steps = plan.get("steps", [])
    if not steps:
        return True, ""

    step_ids = {s["id"] for s in steps}

    # Check all dependencies exist
    for step in steps:
        for dep in step.get("depends_on", []):
            if dep not in step_ids:
                return False, f"Step {step['id']} depends on unknown step {dep}"

    # Check for cycles using DFS
    visited = set()
    rec_stack = set()

    def has_cycle(step_id: str) -> bool:
        visited.add(step_id)
        rec_stack.add(step_id)

        step = next((s for s in steps if s["id"] == step_id), None)
        if step:
            for dep in step.get("depends_on", []):
                if dep not in visited:
                    if has_cycle(dep):
                        return True
                elif dep in rec_stack:
                    return True

        rec_stack.remove(step_id)
        return False

    for step in steps:
        if step["id"] not in visited:
            if has_cycle(step["id"]):
                return False, "Circular dependency detected"

    return True, ""
