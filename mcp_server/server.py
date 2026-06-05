"""MCP Server — Section 5.

stdio transport MCP server spawned by Continue IDE.
Exposes tools to the LLM via Model Context Protocol.

Tools: read_file, search_symbol, grep, get_project_skeleton, index_with_deps,
run_command, diff_preview, apply_edits, git_* (status/diff/log/commit/branch),
run_tests, lint_code, rename_symbol, extract_function, inline_variable,
and GitLab review tools (get_pr_diff/get_mr_note/upsert_mr_comment).
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
from mcp_server.tools import (
    read_file,
    search_symbol,
    grep_content,
    run_command,
    diff_preview,
    apply_edits,
    git_status,
    git_diff,
    git_log,
    git_commit,
    git_branch,
    run_tests,
    lint_code,
)
from mcp_server.tools_indexer import get_project_skeleton, index_with_deps
from mcp_server.tools_multifile import apply_multi_file_edits
from mcp_server.tools_refactor import (
    rename_symbol,
    extract_function,
    inline_variable,
)
from mcp_server.tools_review import (
    get_pr_diff as review_get_pr_diff,
    get_mr_note as review_get_mr_note,
    upsert_mr_comment as review_upsert_mr_comment,
)

logger = logging.getLogger("mcp_server")

# Environment variables injected by Continue (Section 5)
REPO_PATH = os.environ.get("REPO_PATH", ".")
SERVER_URL = os.environ.get("SERVER_URL", "https://research-rd.internal.prd.vtrip.cloudhms.io")
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
                name="vtrip_read_file",
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
                name="vtrip_search_symbol",
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
                name="vtrip_get_project_skeleton",
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
                name="vtrip_index_with_deps",
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
                name="vtrip_grep",
                description=(
                    "Full-text/regex content search across the repo (ripgrep-style). "
                    "Reads files fresh from disk; use to find where text/patterns appear "
                    "when you don't know the exact symbol name."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "description": "Regular expression to search for"},
                        "path_glob": {"type": "string", "description": "Optional glob filter, e.g. '**/*.py'"},
                        "ignore_case": {"type": "boolean", "default": False},
                        "max_results": {"type": "integer", "default": 100},
                    },
                    "required": ["pattern"],
                },
            ),
            Tool(
                name="vtrip_run_command",
                description="Execute a whitelisted shell command (tests, lint, build) in the repo.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Command to execute"},
                        "working_dir": {"type": "string", "description": "Subdirectory relative to repo root"},
                    },
                    "required": ["command"],
                },
            ),
            Tool(
                name="vtrip_diff_preview",
                description="Preview a unified diff of proposed changes to a file before applying.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "Path relative to repo root"},
                        "new_content": {"type": "string", "description": "Proposed new file content"},
                    },
                    "required": ["file_path", "new_content"],
                },
            ),
            Tool(
                name="vtrip_apply_edits",
                description="Apply edits to multiple files atomically (full content or search/replace).",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "edits": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "file_path": {"type": "string"},
                                    "new_content": {"type": "string"},
                                    "search": {"type": "string"},
                                    "replace": {"type": "string"},
                                },
                                "required": ["file_path"],
                            },
                        },
                        "dry_run": {"type": "boolean", "default": False},
                    },
                    "required": ["edits"],
                },
            ),
            Tool(
                name="vtrip_apply_edits_atomic",
                description=(
                    "Apply edits to multiple files as ONE transaction: backup, conflict "
                    "detection, apply all, roll back every file if any fails. dry_run previews diffs."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "edits": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "file_path": {"type": "string"},
                                    "new_content": {"type": "string"},
                                    "search": {"type": "string"},
                                    "replace": {"type": "string"},
                                },
                                "required": ["file_path"],
                            },
                        },
                        "dry_run": {"type": "boolean", "default": False},
                    },
                    "required": ["edits"],
                },
            ),
            Tool(
                name="vtrip_git_status",
                description="Get git status: branch, staged, modified, untracked files.",
                inputSchema={"type": "object", "properties": {}},
            ),
            Tool(
                name="vtrip_git_diff",
                description="Get git diff for a file or the entire repo.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                        "staged": {"type": "boolean", "default": False},
                    },
                },
            ),
            Tool(
                name="vtrip_git_log",
                description="Get recent git commits.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "count": {"type": "integer", "default": 10},
                        "file_path": {"type": "string"},
                    },
                },
            ),
            Tool(
                name="vtrip_git_commit",
                description="Stage files and create a git commit.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "message": {"type": "string"},
                        "files": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["message"],
                },
            ),
            Tool(
                name="vtrip_git_branch",
                description="List branches or create/checkout a branch.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "checkout": {"type": "boolean", "default": False},
                    },
                },
            ),
            Tool(
                name="vtrip_run_tests",
                description="Run tests with auto framework detection and parse pass/fail results.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "test_file": {"type": "string"},
                        "test_name": {"type": "string"},
                        "framework": {"type": "string", "default": "auto"},
                    },
                },
            ),
            Tool(
                name="vtrip_lint_code",
                description="Run a linter (auto-detected) and return issues; optionally auto-fix.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                        "fix": {"type": "boolean", "default": False},
                        "linter": {"type": "string", "default": "auto"},
                    },
                },
            ),
            Tool(
                name="vtrip_rename_symbol",
                description="Rename a symbol across the codebase (AST-aware). dry_run previews edits.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "old_name": {"type": "string"},
                        "new_name": {"type": "string"},
                        "scope": {"type": "string", "enum": ["project", "file"], "default": "project"},
                        "dry_run": {"type": "boolean", "default": True},
                    },
                    "required": ["old_name", "new_name"],
                },
            ),
            Tool(
                name="vtrip_extract_function",
                description="Extract a line range into a new function. dry_run previews edits.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                        "start_line": {"type": "integer"},
                        "end_line": {"type": "integer"},
                        "new_function_name": {"type": "string"},
                        "dry_run": {"type": "boolean", "default": True},
                    },
                    "required": ["file_path", "start_line", "end_line", "new_function_name"],
                },
            ),
            Tool(
                name="vtrip_inline_variable",
                description="Inline a variable by replacing its uses with its value. dry_run previews edits.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                        "variable_name": {"type": "string"},
                        "dry_run": {"type": "boolean", "default": True},
                    },
                    "required": ["file_path", "variable_name"],
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
        if name == "vtrip_read_file":
            result = read_file(
                repo_path=REPO_PATH,
                file_path=arguments["file_path"],
                start_line=arguments.get("start_line", 1),
                end_line=arguments.get("end_line", 150),
            )
        elif name == "vtrip_search_symbol":
            result = search_symbol(
                repo_path=REPO_PATH,
                registry=registry,
                name=arguments["name"],
                type_filter=arguments.get("type_filter", "any"),
            )
        elif name == "vtrip_get_project_skeleton":
            result = get_project_skeleton(
                repo_path=REPO_PATH,
                registry=registry,
                include_methods=arguments.get("include_methods", True),
            )
        elif name == "vtrip_index_with_deps":
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
        elif name == "vtrip_grep":
            result = grep_content(
                repo_path=REPO_PATH,
                pattern=arguments["pattern"],
                path_glob=arguments.get("path_glob"),
                ignore_case=arguments.get("ignore_case", False),
                max_results=arguments.get("max_results", 100),
            )
        elif name == "vtrip_run_command":
            result = run_command(
                repo_path=REPO_PATH,
                command=arguments["command"],
                working_dir=arguments.get("working_dir"),
            )
        elif name == "vtrip_diff_preview":
            result = diff_preview(
                repo_path=REPO_PATH,
                file_path=arguments["file_path"],
                new_content=arguments["new_content"],
            )
        elif name == "vtrip_apply_edits":
            result = apply_edits(
                repo_path=REPO_PATH,
                edits=arguments["edits"],
                dry_run=arguments.get("dry_run", False),
            )
        elif name == "vtrip_apply_edits_atomic":
            result = apply_multi_file_edits(
                repo_path=REPO_PATH,
                edits=arguments["edits"],
                dry_run=arguments.get("dry_run", False),
            )
        elif name == "vtrip_git_status":
            result = git_status(repo_path=REPO_PATH)
        elif name == "vtrip_git_diff":
            result = git_diff(
                repo_path=REPO_PATH,
                file_path=arguments.get("file_path"),
                staged=arguments.get("staged", False),
            )
        elif name == "vtrip_git_log":
            result = git_log(
                repo_path=REPO_PATH,
                count=arguments.get("count", 10),
                file_path=arguments.get("file_path"),
            )
        elif name == "vtrip_git_commit":
            result = git_commit(
                repo_path=REPO_PATH,
                message=arguments["message"],
                files=arguments.get("files"),
            )
        elif name == "vtrip_git_branch":
            result = git_branch(
                repo_path=REPO_PATH,
                name=arguments.get("name"),
                checkout=arguments.get("checkout", False),
            )
        elif name == "vtrip_run_tests":
            result = run_tests(
                repo_path=REPO_PATH,
                test_file=arguments.get("test_file"),
                test_name=arguments.get("test_name"),
                framework=arguments.get("framework", "auto"),
            )
        elif name == "vtrip_lint_code":
            result = lint_code(
                repo_path=REPO_PATH,
                file_path=arguments.get("file_path"),
                fix=arguments.get("fix", False),
                linter=arguments.get("linter", "auto"),
            )
        elif name == "vtrip_rename_symbol":
            result = rename_symbol(
                repo_path=REPO_PATH,
                registry=registry,
                old_name=arguments["old_name"],
                new_name=arguments["new_name"],
                scope=arguments.get("scope", "project"),
                dry_run=arguments.get("dry_run", True),
            )
        elif name == "vtrip_extract_function":
            result = extract_function(
                repo_path=REPO_PATH,
                file_path=arguments["file_path"],
                start_line=arguments["start_line"],
                end_line=arguments["end_line"],
                new_function_name=arguments["new_function_name"],
                dry_run=arguments.get("dry_run", True),
            )
        elif name == "vtrip_inline_variable":
            result = inline_variable(
                repo_path=REPO_PATH,
                file_path=arguments["file_path"],
                variable_name=arguments["variable_name"],
                dry_run=arguments.get("dry_run", True),
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

    log_file = os.path.join(os.path.expanduser("~"), "mcp-server-debug.log")
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stderr),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )
    logger.info("Starting MCP server for repo: %s (log: %s)", REPO_PATH, log_file)
    asyncio.run(run_server())


if __name__ == "__main__":
    main()
