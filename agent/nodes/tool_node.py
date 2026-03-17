"""
tool_node — deterministic executor for "Action Tools" in LangGraph.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
import structlog
from agent.tools import ToolManager, run_compile, run_test

logger = structlog.get_logger()

def create_action_tools() -> ToolManager:
    """Create and register action tools."""
    manager = ToolManager()
    manager.register_tool("run_compile", run_compile)
    manager.register_tool("run_test", run_test)
    return manager

async def tool_node(state: dict, *, action_tools: Optional[ToolManager] = None) -> dict:
    """Deterministic node to execute tools requested in the state.
    
    Expected state fields:
      - tool_request: {"name": "run_test", "parameters": {"test_class": "..."}}
    """
    if action_tools is None:
        action_tools = create_action_tools()
        
    request = state.get("tool_request")
    if not request:
        logger.warning("tool_node: no tool_request in state")
        return {}
        
    tool_name = request.get("name")
    tool_params = request.get("parameters", {})
    
    logger.info("tool_node: executing action tool", tool=tool_name)
    
    try:
        result = await action_tools.acall_tool(tool_name, **tool_params)
        
        # Save results to state
        return {
            "tool_result": result,
            "execution_passed": result.get("passed", False),
            "execution_output": result.get("stdout", "") + result.get("stderr", ""),
            "tool_request": None # Clear request after execution
        }
    except Exception as e:
        logger.error("tool_node: execution failed", error=str(e))
        return {
            "tool_result": {"passed": False, "error": str(e)},
            "execution_passed": False,
            "tool_request": None
        }
