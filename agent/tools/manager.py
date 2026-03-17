"""
ToolManager — registry and invocation hub for agent tools.
"""
from __future__ import annotations
import inspect
from typing import Any, Callable, Dict, List, Optional
import structlog

logger = structlog.get_logger()

class Tool:
    """Wrapper for a tool function with metadata."""
    def __init__(self, name: str, func: Callable, description: str):
        self.name = name
        self.func = func
        self.description = description
        self.signature = inspect.signature(func)

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

class ToolManager:
    """Manages registration and execution of agent tools."""
    
    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register_tool(self, name: str, func: Callable, description: Optional[str] = None):
        """Register a new tool."""
        desc = description or func.__doc__ or "No description provided."
        self._tools[name] = Tool(name, func, desc)
        logger.debug("Tool registered", name=name)

    def get_tool(self, name: str) -> Optional[Tool]:
        """Retrieve a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> List[Dict[str, str]]:
        """List all registered tools with descriptions."""
        return [
            {"name": t.name, "description": t.description}
            for t in self._tools.values()
        ]

    def get_tool_definitions_for_llm(self) -> List[Dict[str, Any]]:
        """Return tool definitions in a format suitable for LLM function calling (OpenAI-like)."""
        definitions = []
        for tool in self._tools.values():
            params: Dict[str, Any] = {
                "type": "object",
                "properties": {},
                "required": []
            }
            props = params["properties"]
            required = params["required"]
            
            for name, param in tool.signature.parameters.items():
                if name == "self": continue
                
                param_type = "string" # Default
                if param.annotation == int: param_type = "integer"
                elif param.annotation == bool: param_type = "boolean"
                
                props[name] = {
                    "type": param_type,
                    "description": f"Parameter {name}"
                }
                if param.default == inspect.Parameter.empty:
                    required.append(name)

            definitions.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": params
                }
            })
        return definitions

    async def acall_tool(self, name: str, **kwargs) -> Any:
        """Asynchronously call a tool."""
        tool = self.get_tool(name)
        if not tool:
            raise ValueError(f"Tool '{name}' not found.")
        
        logger.info("Calling tool", name=name, args=kwargs)
        try:
            if inspect.iscoroutinefunction(tool.func):
                return await tool.func(**kwargs)
            else:
                return tool.func(**kwargs)
        except Exception as e:
            logger.error("Tool execution failed", name=name, error=str(e))
            raise
