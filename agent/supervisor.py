import json
import re
from typing import Any, Dict, List, Optional

import structlog
from agent.tools import ToolManager, list_files, grep_search, read_file
from agent.prompt import TemplateEngine

logger = structlog.get_logger()

# ── Tool Setup ──────────────────────────────────────────────────────────

def create_discovery_tools() -> ToolManager:
    """Create and register discovery tools."""
    manager = ToolManager()
    manager.register_tool("list_files", list_files)
    manager.register_tool("grep_search", grep_search)
    manager.register_tool("read_file", read_file)
    return manager

# ── Agentic Supervisor ──────────────────────────────────────────────────

async def supervisor_node(
    state: dict,
    *,
    vllm_client,
    template_engine: Optional[TemplateEngine] = None,
) -> dict:
    """Agentic Supervisor — uses LLM + Tools to understand intent and codebase.

    Args:
        state: AgentState dict.
        vllm_client: VLLMClient for LLM calls.
        template_engine: TemplateEngine for prompts.

    Returns:
        State updates: intent, discovery_context, and any parameters.
    """
    user_input = state.get("user_input", "")
    if not user_input:
        return {"intent": "unit_test"}

    if template_engine is None:
        template_engine = TemplateEngine()

    tools = create_discovery_tools()
    
    logger.info("supervisor_node: starting agentic discovery", input_preview=user_input[:50])

    # Initial prompt rendering
    prompt = template_engine.render(
        "supervisor_prompt.jinja2",
        user_input=user_input,
        current_context=state.get("discovery_context", {})
    )

    # Tool calling loop (max 3 iterations for discovery)
    discovery_context = {}
    intent = "unit_test"
    params = {}

    for i in range(3):
        logger.debug("supervisor_node: llm call", iteration=i)
        
        # Note: We need a way to pass tools to agenerate.
        # For now, we'll use a simplified loop where the LLM can ask for a tool.
        # In a real system, we'd use the VLLM tool-calling API.
        
        response = await vllm_client.agenerate(
            system_prompt="You are a helpful assistant.",
            user_prompt=prompt
        )

        try:
            # Try to parse JSON from the response
            # Note: This is a simplification. Real tool-calling would use dedicated API.
            clean_response = _extract_json(response)
            if not clean_response:
                logger.warning("supervisor_node: failed to extract JSON", response=response)
                break
            
            data = json.loads(clean_response)
            
            # Check if it wants to call a tool (manual simulation for now)
            if "tool_call" in data:
                tool_name = data["tool_call"]["name"]
                tool_args = data["tool_call"]["parameters"]
                
                logger.info("supervisor_node: executing discovery tool", tool=tool_name)
                result = await tools.acall_tool(tool_name, **tool_args)
                
                # Feed back to loop
                prompt += f"\n\nTool Result ({tool_name}):\n{result}"
                continue
            
            # Final result
            discovery_context = data.get("discovery_context", {})
            intent = data.get("intent", "unit_test")
            params = data.get("parameters", {})
            break

        except Exception as e:
            logger.error("supervisor_node: loop error", error=str(e))
            break

    logger.info("supervisor_node: finished", intent=intent, discovered_files=len(discovery_context.get("relevant_files", [])))

    # Merge params into state updates
    updates = {
        "intent": intent,
        "discovery_context": discovery_context,
    }
    updates.update(params)
    return updates

def _extract_json(text: str) -> Optional[str]:
    """Extract JSON block from LLM text."""
    match = re.search(r"({.*})", text, re.DOTALL)
    return match.group(1) if match else None
