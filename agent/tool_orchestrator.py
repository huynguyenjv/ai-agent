import json
import re
from typing import List, Optional, Dict, Any
from pydantic import BaseModel

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
        """Extract tool calls from model output text."""
        calls = []
        matches = self.tool_call_pattern.finditer(text)
        for match in matches:
            content = match.group(1).strip()
            try:
                call_data = json.loads(content)
                if "name" in call_data:
                    calls.append(call_data)
            except json.JSONDecodeError:
                # Log or handle malformed JSON
                pass
        return calls

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
