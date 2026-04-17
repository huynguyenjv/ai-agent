"""Node: plan_steps — Section 8.

Only invoked for code_gen intent.
Calls the LLM with a planning-specific prompt.
Max 800ms, ~400 tokens.
"""

from __future__ import annotations

import asyncio
import logging

from openai import AsyncOpenAI

from server.agent.state import AgentState

logger = logging.getLogger("server.agent.plan_steps")

PLANNING_SYSTEM_PROMPT = """You are a code planning assistant. Given the user's request and the retrieved code context, output a structured JSON plan describing:
1. Which files need to be read or modified
2. What each modification step is
3. What the expected output should look like

Output ONLY valid JSON with this structure:
{
  "steps": [
    {"action": "read|modify|create", "file": "path", "description": "what to do"}
  ],
  "expected_output": "brief description of final result"
}"""


async def plan_steps(state: AgentState, vllm_client: AsyncOpenAI, model: str) -> dict:
    """Generate a structured plan for code_gen tasks.

    Section 8: Max 800ms, ~400 max_tokens.
    """
    messages = state.get("messages", [])
    rag_chunks = state.get("rag_chunks", [])

    # Build context from RAG chunks
    context_parts = []
    for chunk in rag_chunks[:5]:
        symbol = chunk.get("symbol_name", "unknown")
        body = chunk.get("body", "")
        if body:
            context_parts.append(f"// {symbol}\n{body}")

    context = "\n\n".join(context_parts)

    # Build planning messages
    plan_messages = [
        {"role": "system", "content": PLANNING_SYSTEM_PROMPT},
    ]

    if context:
        plan_messages.append({
            "role": "system",
            "content": f"Code context:\n{context}",
        })

    # Add the user's last message
    if messages:
        last = messages[-1]
        content = last.content if hasattr(last, "content") else last.get("content", "")
        plan_messages.append({"role": "user", "content": content})

    try:
        response = await asyncio.wait_for(
            vllm_client.chat.completions.create(
                model=model,
                messages=plan_messages,
                max_tokens=400,
                temperature=0.1,
            ),
            timeout=0.8,  # 800ms max per Section 8
        )
        plan_text = response.choices[0].message.content
        return {"tool_results": state.get("tool_results", []) + [{"plan": plan_text}]}
    except Exception as e:
        logger.warning("Planning step failed: %s", e)
        return {"tool_results": state.get("tool_results", [])}
