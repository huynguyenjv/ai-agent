"""MCP Server — Section 5.

stdio transport MCP server spawned by Continue IDE.
Exposes tools to the LLM via Model Context Protocol.

Tools: read_file, search_symbol, get_project_skeleton, index_with_deps.
"""

from __future__ import annotations

import json
import os
import sys
import logging
from pathlib import Path


from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from mcp_server.plugins.registry import PluginRegistry
from mcp_server.plugins.fallback import FallbackPlugin
from mcp_server.plugins.java_plugin import JavaPlugin
from mcp_server.plugins.go_plugin import GoPlugin
from mcp_server.plugins.python_plugin import PythonPlugin
from mcp_server.plugins.typescript_plugin import TypeScriptPlugin
from mcp_server.plugins.csharp_plugin import CSharpPlugin
from mcp_server.plugins.hcl_plugin import HCLPlugin
from mcp_server.hash_store import HashStore
from mcp_server.dep_classifier import DepClassifier
from mcp_server.uploader import Uploader
from mcp_server.tools import read_file, search_symbol
from mcp_server.tools_indexer import get_project_skeleton, index_with_deps
from mcp_server.tools_review import (
    get_pr_diff as review_get_pr_diff,
    get_mr_note as review_get_mr_note,
    upsert_mr_comment as review_upsert_mr_comment,
)

logger = logging.getLogger("mcp_server")

# Environment variables injected by Continue (Section 5)
REPO_PATH = os.environ.get("REPO_PATH", ".")
SERVER_URL = os.environ.get("SERVER_URL", "http://localhost:8000")
API_KEY = os.environ.get("API_KEY", "")
TOKEN_BUDGET = int(os.environ.get("TOKEN_BUDGET", "8000"))
DEPTH_DEFAULT = int(os.environ.get("DEPTH_DEFAULT", "2"))


def create_server() -> Server:
    """Create and configure the MCP server with all tools and plugins."""
    server = Server("ai-coding-agent")

    # Build plugin registry — Section 6
    registry = PluginRegistry()
    registry.register(JavaPlugin())
    registry.register(GoPlugin())
    registry.register(PythonPlugin())
    registry.register(TypeScriptPlugin())
    registry.register(CSharpPlugin())
    registry.register(HCLPlugin())
    registry.register(FallbackPlugin())

    hash_store = HashStore()
    dep_classifier = DepClassifier()
    uploader = Uploader(SERVER_URL, API_KEY)

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """Declare available tools per MCP protocol."""
        return [
            Tool(
                name="read_file",
                description=(
                    "Read a contiguous range of lines from a file. "
                    "Used when the LLM needs exact, guaranteed-fresh content."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path relative to repo root",
                        },
                        "start_line": {
                            "type": "integer",
                            "description": "Start line (1-based), default 1",
                            "default": 1,
                        },
                        "end_line": {
                            "type": "integer",
                            "description": "End line (1-based), default 150",
                            "default": 150,
                        },
                    },
                    "required": ["file_path"],
                },
            ),
            Tool(
                name="search_symbol",
                description=(
                    "Locate a class, function, or method by name anywhere "
                    "in the repository, returning file path and line number."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Symbol name to search for",
                        },
                        "type_filter": {
                            "type": "string",
                            "enum": ["class", "function", "method", "any"],
                            "description": "Filter by symbol type, default 'any'",
                            "default": "any",
                        },
                    },
                    "required": ["name"],
                },
            ),
            Tool(
                name="get_project_skeleton",
                description=(
                    "Return a compact structural overview of the entire repository. "
                    "Used for wide structural queries like 'analyze the architecture'."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "include_methods": {
                            "type": "boolean",
                            "description": "Include public method names (not bodies), default true",
                            "default": True,
                        },
                    },
                },
            ),
            Tool(
                name="index_with_deps",
                description=(
                    "Parse a specific file and its project-local dependencies "
                    "up to a given depth, then upload all changed chunks to the "
                    "server for embedding and storage."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path relative to repo root",
                        },
                        "depth": {
                            "type": "integer",
                            "description": "BFS depth (default 2, max 3)",
                            "default": 2,
                        },
                    },
                    "required": ["file_path"],
                },
            ),
            Tool(
                name="get_pr_diff",
                description="Fetch GitLab MR unified diff and metadata by project path and MR IID.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "provider": {"type": "string", "enum": ["gitlab"], "default": "gitlab"},
                        "repo": {"type": "string", "description": "Project path, e.g. group/project"},
                        "pr_id": {"type": "integer", "description": "MR IID"},
                    },
                    "required": ["repo", "pr_id"],
                },
            ),
            Tool(
                name="get_mr_note",
                description="Find existing AI review note on an MR by marker substring.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "provider": {"type": "string", "enum": ["gitlab"], "default": "gitlab"},
                        "repo": {"type": "string"},
                        "pr_id": {"type": "integer"},
                        "marker": {"type": "string"},
                    },
                    "required": ["repo", "pr_id", "marker"],
                },
            ),
            Tool(
                name="upsert_mr_comment",
                description="Create or update a comment on an MR. Pass note_id to update, omit to create.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "provider": {"type": "string", "enum": ["gitlab"], "default": "gitlab"},
                        "repo": {"type": "string"},
                        "pr_id": {"type": "integer"},
                        "body": {"type": "string"},
                        "note_id": {"type": "integer"},
                    },
                    "required": ["repo", "pr_id", "body"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        """Route tool calls to their implementations."""
        if name == "read_file":
            result = read_file(
                repo_path=REPO_PATH,
                file_path=arguments["file_path"],
                start_line=arguments.get("start_line", 1),
                end_line=arguments.get("end_line", 150),
            )
        elif name == "search_symbol":
            result = search_symbol(
                repo_path=REPO_PATH,
                registry=registry,
                name=arguments["name"],
                type_filter=arguments.get("type_filter", "any"),
            )
        elif name == "get_project_skeleton":
            result = get_project_skeleton(
                repo_path=REPO_PATH,
                registry=registry,
                include_methods=arguments.get("include_methods", True),
            )
        elif name == "index_with_deps":
            result = await index_with_deps(
                repo_path=REPO_PATH,
                registry=registry,
                hash_store=hash_store,
                uploader=uploader,
                dep_classifier=dep_classifier,
                file_path=arguments["file_path"],
                depth=arguments.get("depth", DEPTH_DEFAULT),
                token_budget=TOKEN_BUDGET,
            )
        elif name == "get_pr_diff":
            result = await review_get_pr_diff(
                provider=arguments.get("provider", "gitlab"),
                repo=arguments["repo"],
                pr_id=arguments["pr_id"],
            )
        elif name == "get_mr_note":
            result = await review_get_mr_note(
                provider=arguments.get("provider", "gitlab"),
                repo=arguments["repo"],
                pr_id=arguments["pr_id"],
                marker=arguments["marker"],
            )
        elif name == "upsert_mr_comment":
            result = await review_upsert_mr_comment(
                provider=arguments.get("provider", "gitlab"),
                repo=arguments["repo"],
                pr_id=arguments["pr_id"],
                body=arguments["body"],
                note_id=arguments.get("note_id"),
            )
        else:
            result = {"error": f"Unknown tool: {name}"}

        return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]

    return server


async def run_server() -> None:
    """Run the MCP server over stdio transport."""
    server = create_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main() -> None:
    """Entry point for the MCP server."""
    import asyncio

    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    logger.info("Starting MCP server for repo: %s", REPO_PATH)
    asyncio.run(run_server())


if __name__ == "__main__":
    main()
