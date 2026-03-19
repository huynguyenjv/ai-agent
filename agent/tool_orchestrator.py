import json
import re
import structlog
from typing import List, Optional, Dict, Any
from pydantic import BaseModel

logger = structlog.get_logger()

# Re-use models from api.py if possible, but for standalone logic we define them or import them.
# To avoid circular imports, we'll use simple dicts or define minimal structures here.

class ToolOrchestrator:
    """Orchestrates tool-calling for models that don't support native tool-calling.
    
    Uses a custom XML-based protocol:
    <tool_call>
    {"name": "tool_name", "arguments": {"arg1": "val1"}}
    </tool_call>
    """

    def __init__(self):
        self.tool_call_pattern = re.compile(r'<tool_call>(.*?)</tool_call>', re.DOTALL)

    def build_tool_system_prompt(self, tools: List[Any]) -> str:
        """Construct a system prompt segment describing available tools."""
        if not tools:
            return ""

        tools_desc = []
        for tool in tools:
            # tool is expected to be a ToolDefinition (see api.py)
            func = tool.function
            desc = f"- **{func.name}**: {func.description or 'No description provided'}\n"
            if func.parameters:
                desc += f"  Parameters: {json.dumps(func.parameters, indent=2)}\n"
            tools_desc.append(desc)

        tools_block = "\n".join(tools_desc)

        return f"""
# Tool Use Instructions
You have access to the following tools. If the user's request requires using a tool, you MUST respond with a SPECIAL XML TAG containing a JSON object for the tool call.

## Available Tools:
{tools_block}

## How to Call a Tool:
To call a tool, use the following format:
<tool_call>
{{"name": "tool_name", "arguments": {{"param1": "value1"}}}}
</tool_call>

## Rules:
1. ONLY use the tools listed above.
2. Provide valid JSON inside the `<tool_call>` tag.
3. You can call multiple tools by repeating the tag if necessary.
4. After you receive a tool result (as a 'tool' role message), summarize it for the user.
5. Do NOT try to execute tools yourself; only output the XML tags.
"""

    def parse_tool_calls(self, text: str) -> List[Dict[str, Any]]:
        """Extract tool calls from model output text.
        
        P3: Uses multi-strategy JSON extraction with schema validation
        for robustness against malformed model output.
        """
        calls = []
        matches = self.tool_call_pattern.finditer(text)
        for match in matches:
            content = match.group(1).strip()
            try:
                call_data = self._extract_json(content)
                if call_data and self._validate_tool_schema(call_data):
                    calls.append(call_data)
                else:
                    logger.warning("Tool call failed schema validation",
                                   raw=content[:200])
            except Exception as exc:
                logger.warning("Malformed tool call content",
                               error=str(exc), raw=content[:200])
        return calls

    def _extract_json(self, content: str) -> Optional[Dict[str, Any]]:
        """Multi-strategy JSON extraction from tool call content.
        
        Strategies (in order):
        1. Direct JSON parse
        2. Find first '{' to last '}' and parse
        3. Strip markdown code fence then parse
        """
        content = content.strip()

        # Strategy 1: Direct parse
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Strategy 2: Extract outermost JSON object
        first_brace = content.find("{")
        last_brace = content.rfind("}")
        if first_brace != -1 and last_brace > first_brace:
            try:
                return json.loads(content[first_brace:last_brace + 1])
            except json.JSONDecodeError:
                pass

        # Strategy 3: Strip markdown code fence (```json ... ```)
        fence_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', content, re.DOTALL)
        if fence_match:
            try:
                return json.loads(fence_match.group(1).strip())
            except json.JSONDecodeError:
                pass

        return None

    @staticmethod
    def _validate_tool_schema(data: Any) -> bool:
        """Validate that parsed data has the required tool call structure."""
        if not isinstance(data, dict):
            return False
        if "name" not in data or not isinstance(data["name"], str):
            return False
        if "arguments" in data and not isinstance(data["arguments"], dict):
            return False
        return True

    def extract_partial_tool_call(self, text: str) -> Optional[str]:
        """Detect if the text contains an opening <tool_call> tag but no closing tag yet.
        
        Returns the content after <tool_call> if a partial call is detected.
        """
        start_tag = "<tool_call>"
        end_tag = "</tool_call>"
        
        start_idx = text.rfind(start_tag)
        if start_idx == -1:
            return None
            
        end_idx = text.find(end_tag, start_idx)
        if end_idx != -1:
            # Full tag found, already handled by parse_tool_calls or caller
            return None
            
        return text[start_idx + len(start_tag):]
