from __future__ import annotations

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read contents of a file by absolute path. Optionally limit to a line range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute file path"},
                    "start_line": {"type": "integer", "description": "Optional 1-indexed start line"},
                    "end_line": {"type": "integer", "description": "Optional 1-indexed end line"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep_code",
            "description": "Search for a regex pattern across the codebase using ripgrep-compatible syntax.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern"},
                    "path": {"type": "string", "description": "Optional path to restrict search"},
                    "glob": {"type": "string", "description": "Optional glob filter, e.g. *.java"},
                },
                "required": ["pattern"],
            },
        },
    },
]


def merge_tools(client_tools: list[dict] | None) -> list[dict]:
    """Merge server registry with client-provided tools.

    Dedupe by function.name. Server schemas take priority (overwrite client).
    """
    out: dict[str, dict] = {}
    for t in client_tools or []:
        name = t.get("function", {}).get("name")
        if name:
            out[name] = t
    for t in TOOL_SCHEMAS:
        out[t["function"]["name"]] = t
    return list(out.values())
