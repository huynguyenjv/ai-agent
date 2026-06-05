"""Node: generate — native tool-call streaming.

Forwards merged tools (server registry + client) to vLLM, streams content
tokens, and captures tool_call deltas to accumulate a final pending_tool_calls
list for Turn 2.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any

from openai import AsyncOpenAI

from server.agent.context_builder import build_optimal_context
from server.agent.fallback import llm_unavailable_draft
from server.agent.state import AgentState
from server.cache import get_llm_cache
from server.circuit_breaker import get_circuit_breaker
from server.utils.sanitize import sanitize_tool_output

logger = logging.getLogger("server.agent.generate")

# =============================================================================
# MCP Tools Schema
# =============================================================================

MCP_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "vtrip_read_file",
            "description": "Read a contiguous range of lines from a file. Used when you need exact, guaranteed-fresh content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path relative to repo root"},
                    "start_line": {"type": "integer", "description": "Start line (1-based), default 1", "default": 1},
                    "end_line": {"type": "integer", "description": "End line (1-based), default 150", "default": 150},
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "vtrip_search_symbol",
            "description": "Locate a class, function, or method by name anywhere in the repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Symbol name to search for"},
                    "type_filter": {"type": "string", "enum": ["class", "function", "method", "any"], "default": "any"},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "vtrip_grep",
            "description": "Full-text/regex content search across the repo (ripgrep-style). Reads files fresh from disk. Use to find where text/patterns appear when you don't know the exact symbol name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regular expression to search for"},
                    "path_glob": {"type": "string", "description": "Optional glob filter, e.g. '**/*.py'"},
                    "ignore_case": {"type": "boolean", "default": False},
                    "max_results": {"type": "integer", "default": 100},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "vtrip_get_project_skeleton",
            "description": "Return a compact structural overview of the entire repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "include_methods": {"type": "boolean", "default": True},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "vtrip_index_with_deps",
            "description": "Parse a specific file and its dependencies, upload chunks for embedding.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path relative to repo root"},
                    "depth": {"type": "integer", "default": 2},
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "vtrip_run_command",
            "description": "Execute a shell command to run tests, lint, or build. Use to verify code changes work correctly.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Command to execute (e.g., 'npm test', 'pytest tests/', 'mvn test')",
                    },
                    "working_dir": {
                        "type": "string",
                        "description": "Subdirectory to run in (relative to repo root)",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "vtrip_diff_preview",
            "description": "Preview changes before applying. Shows unified diff of proposed edits to a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path relative to repo root",
                    },
                    "new_content": {
                        "type": "string",
                        "description": "Proposed new content for the file",
                    },
                },
                "required": ["file_path", "new_content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "vtrip_apply_edits",
            "description": "Apply edits to multiple files atomically. Supports full content replacement or search/replace.",
            "parameters": {
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
                        "description": "List of edits. Each needs file_path + (new_content OR search+replace)",
                    },
                    "dry_run": {
                        "type": "boolean",
                        "default": False,
                        "description": "Preview only, don't apply changes",
                    },
                },
                "required": ["edits"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "vtrip_apply_edits_atomic",
            "description": "Apply edits to multiple files atomically with backup + rollback on failure + conflict detection. Prefer this over vtrip_apply_edits for multi-file changes. dry_run previews diffs.",
            "parameters": {
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
        },
    },
    # Git Integration Tools
    {
        "type": "function",
        "function": {
            "name": "vtrip_git_status",
            "description": "Get git status: branch, staged files, modified files, untracked files.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "vtrip_git_diff",
            "description": "Get git diff for a file or entire repo.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "File to diff (optional, defaults to all)"},
                    "staged": {"type": "boolean", "default": False, "description": "Show staged changes only"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "vtrip_git_log",
            "description": "Get recent git commits.",
            "parameters": {
                "type": "object",
                "properties": {
                    "count": {"type": "integer", "default": 10, "description": "Number of commits"},
                    "file_path": {"type": "string", "description": "Filter by file (optional)"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "vtrip_git_commit",
            "description": "Stage files and create a git commit.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "Commit message"},
                    "files": {"type": "array", "items": {"type": "string"}, "description": "Files to stage (optional, defaults to all)"},
                },
                "required": ["message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "vtrip_git_branch",
            "description": "List branches or create/checkout a branch.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Branch name (optional, omit to list)"},
                    "checkout": {"type": "boolean", "default": False, "description": "Checkout after creating"},
                },
            },
        },
    },
    # Code Intelligence Tools (Phase 7)
    {
        "type": "function",
        "function": {
            "name": "vtrip_run_tests",
            "description": "Run tests with automatic framework detection (pytest/jest/junit/go/cargo) and parse pass/fail results. Use to verify generated code behaves correctly.",
            "parameters": {
                "type": "object",
                "properties": {
                    "test_file": {"type": "string", "description": "Specific test file (optional)"},
                    "test_name": {"type": "string", "description": "Specific test name/pattern (optional)"},
                    "framework": {"type": "string", "default": "auto", "description": "Framework or 'auto'"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "vtrip_lint_code",
            "description": "Run a linter (auto-detected: ruff/eslint/gofmt/...) and return issues; set fix=true to auto-fix where supported.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Specific file to lint (optional)"},
                    "fix": {"type": "boolean", "default": False, "description": "Auto-fix issues if supported"},
                    "linter": {"type": "string", "default": "auto", "description": "Linter or 'auto'"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "vtrip_rename_symbol",
            "description": "Rename a symbol (class/function/method) across the codebase using AST-aware search. Use dry_run to preview edits first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "old_name": {"type": "string", "description": "Current symbol name"},
                    "new_name": {"type": "string", "description": "New symbol name"},
                    "scope": {"type": "string", "enum": ["project", "file"], "default": "project"},
                    "dry_run": {"type": "boolean", "default": True, "description": "Preview only"},
                },
                "required": ["old_name", "new_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "vtrip_extract_function",
            "description": "Extract a contiguous line range into a new function. Use dry_run to preview edits first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path relative to repo root"},
                    "start_line": {"type": "integer", "description": "Start line (1-based)"},
                    "end_line": {"type": "integer", "description": "End line (1-based)"},
                    "new_function_name": {"type": "string", "description": "Name for the extracted function"},
                    "dry_run": {"type": "boolean", "default": True, "description": "Preview only"},
                },
                "required": ["file_path", "start_line", "end_line", "new_function_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "vtrip_inline_variable",
            "description": "Inline a variable by replacing all its uses with its value. Use dry_run to preview edits first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "File containing the variable"},
                    "variable_name": {"type": "string", "description": "Variable to inline"},
                    "dry_run": {"type": "boolean", "default": True, "description": "Preview only"},
                },
                "required": ["file_path", "variable_name"],
            },
        },
    },
]

MCP_TOOL_NAMES = {t["function"]["name"] for t in MCP_TOOLS}

# =============================================================================
# Intent-based System Prompts - Senior Level
# =============================================================================

INTENT_PROMPTS = {
    "unit_test": """You are a Senior QA Engineer with 10+ years of experience in test-driven development.

EXPERTISE:
- Write comprehensive unit tests with high coverage
- Mock external dependencies properly (databases, APIs, file systems)
- Test edge cases, boundary conditions, and error scenarios
- Follow testing best practices: AAA pattern (Arrange-Act-Assert), single responsibility per test
- Use appropriate testing frameworks for the language (JUnit/Mockito for Java, pytest for Python, Jest for JS/TS)

APPROACH:
1. Analyze the code structure and identify testable units
2. Identify dependencies that need mocking
3. Write tests for happy path first
4. Add edge cases and error scenarios
5. Ensure tests are independent and repeatable

OUTPUT FORMAT:
- Clear test class/file structure
- Descriptive test method names (should_ReturnX_When_Y)
- Comments only for complex setup
- Complete, runnable test code

Respond in the user's language (Vietnamese or English).""",

    "code_review": """You are a Principal Software Engineer conducting code reviews.

REVIEW FOCUS:
1. **Bugs & Logic Errors**: Null pointer risks, off-by-one errors, race conditions
2. **Security**: SQL injection, XSS, authentication/authorization issues, secrets in code
3. **Performance**: N+1 queries, unnecessary loops, memory leaks, blocking operations
4. **Best Practices**: SOLID principles, design patterns, error handling
5. **Maintainability**: Code readability, naming conventions, complexity

OUTPUT FORMAT:
For each issue found:
- **Location**: File:line
- **Severity**: Critical / Major / Minor / Suggestion
- **Issue**: What's wrong
- **Recommendation**: How to fix
- **Example**: Code snippet if helpful

Be constructive, not critical. Explain WHY something is an issue.
Respond in the user's language (Vietnamese or English).""",

    "structural_analysis": """You are a Software Architect analyzing system design.

ANALYSIS FOCUS:
1. **Architecture Pattern**: Identify the architecture (Layered, Hexagonal, Microservices, etc.)
2. **Module Structure**: How code is organized into packages/modules
3. **Dependencies**: Internal and external dependencies, potential circular dependencies
4. **Design Patterns**: Patterns used (Factory, Repository, Strategy, etc.)
5. **Coupling & Cohesion**: How tightly coupled are components

OUTPUT FORMAT:
```
## Architecture Overview
[High-level description]

## Module Structure
[Package/folder organization]

## Key Components
[Main classes/services and their responsibilities]

## Dependencies
[Internal and external dependencies]

## Observations
[Strengths and potential improvements]
```

Respond in the user's language (Vietnamese or English).""",

    "search": """You are a Codebase Navigator helping developers find code.

CAPABILITIES:
- Locate class, function, method definitions
- Find usages and references
- Identify file paths and line numbers

OUTPUT FORMAT:
For each match found:
- **Symbol**: Name and type (class/function/method)
- **Location**: file/path:line_number
- **Context**: Brief description of what it does

Be precise with file paths and line numbers.
Respond in the user's language (Vietnamese or English).""",

    "debug": """You are a Senior Debugger with expertise in troubleshooting complex issues.

APPROACH:
1. **Understand the Error**: Analyze error messages, stack traces, logs
2. **Reproduce**: Identify conditions that trigger the issue
3. **Isolate**: Narrow down to the specific component/line
4. **Root Cause**: Find the actual cause, not just symptoms
5. **Fix**: Propose a solution that doesn't introduce new issues

OUTPUT FORMAT:
```
## Problem Analysis
[What's happening and why]

## Root Cause
[The actual source of the issue]

## Solution
[Step-by-step fix with code]

## Prevention
[How to avoid similar issues]
```

Respond in the user's language (Vietnamese or English).""",

    "refine": """You are a Refactoring Expert improving code quality.

PRINCIPLES:
- **SOLID**: Single responsibility, Open/closed, Liskov substitution, Interface segregation, Dependency inversion
- **DRY**: Don't Repeat Yourself
- **KISS**: Keep It Simple, Stupid
- **YAGNI**: You Aren't Gonna Need It

REFACTORING TECHNIQUES:
- Extract Method/Class
- Rename for clarity
- Remove dead code
- Simplify conditionals
- Replace magic numbers with constants
- Improve error handling

OUTPUT FORMAT:
1. **Current Issues**: What's wrong with the current code
2. **Proposed Changes**: What to refactor and why
3. **Refactored Code**: The improved version
4. **Benefits**: How this improves the code

Keep behavior unchanged. Make small, incremental changes.
Respond in the user's language (Vietnamese or English).""",

    "explain": """You are a Patient Coding Teacher explaining code to developers.

TEACHING APPROACH:
1. Start with the big picture (what does this code accomplish?)
2. Break down into smaller parts
3. Explain each part with simple language
4. Use analogies when helpful
5. Highlight key concepts and patterns

OUTPUT FORMAT:
```
## Overview
[What this code does in 1-2 sentences]

## How It Works
[Step-by-step explanation]

## Key Concepts
[Important patterns, techniques, or concepts used]

## Example
[If helpful, a simplified example]
```

Adjust explanation depth based on the question.
Respond in the user's language (Vietnamese or English).""",

    "code_gen": """You are a Senior Software Engineer implementing features.

CODING STANDARDS:
- Write clean, readable, maintainable code
- Follow language idioms and conventions
- Handle errors appropriately
- Use meaningful variable/function names
- Keep functions small and focused
- Add types/interfaces where applicable

IMPLEMENTATION APPROACH:
1. Understand the requirement fully
2. Consider edge cases upfront
3. Write self-documenting code
4. Handle errors gracefully
5. Consider performance implications

OUTPUT FORMAT:
- Complete, working code (not snippets)
- Brief explanation of design decisions
- Note any assumptions made

Respond in the user's language (Vietnamese or English).""",
}

DEFAULT_PROMPT = """You are a Senior Software Engineer helping developers.

Provide accurate, well-structured, and practical answers.
Consider best practices, performance, and maintainability.
Respond in the user's language (Vietnamese or English)."""

TOOL_INSTRUCTIONS = """

You have access to these tools:
- vtrip_read_file: Read file content (file_path, start_line, end_line)
- vtrip_search_symbol: Find class/function/method by name (name, type_filter)
- vtrip_grep: Full-text/regex content search across the repo (pattern, path_glob, ignore_case)
- vtrip_get_project_skeleton: Get project structure overview (include_methods)
- vtrip_index_with_deps: Index file with its dependencies (file_path, depth)
- vtrip_run_command: Execute shell command to run tests, lint, build (command, working_dir)
- vtrip_diff_preview: Preview changes before applying, shows unified diff (file_path, new_content)
- vtrip_apply_edits: Apply edits to multiple files (edits[], dry_run)
- vtrip_apply_edits_atomic: Multi-file edit as one transaction with rollback + conflict detection (edits[], dry_run)
- vtrip_git_status: Get git status (branch, staged, modified, untracked)
- vtrip_git_diff: Get git diff (file_path, staged)
- vtrip_git_log: Get recent commits (count, file_path)
- vtrip_git_commit: Create commit (message, files[])
- vtrip_git_branch: List/create/checkout branch (name, checkout)
- vtrip_run_tests: Run tests, auto-detect framework (test_file, test_name, framework)
- vtrip_lint_code: Run linter, optionally auto-fix (file_path, fix, linter)
- vtrip_rename_symbol: AST-aware rename across codebase (old_name, new_name, scope, dry_run)
- vtrip_extract_function: Extract line range into a function (file_path, start_line, end_line, new_function_name, dry_run)
- vtrip_inline_variable: Inline a variable into its uses (file_path, variable_name, dry_run)

Prefer to explore the codebase with tools rather than guessing: use vtrip_grep
for text/regex, vtrip_search_symbol for named definitions, then vtrip_read_file
to read the exact lines. Files read this way are always up to date.

Use tools when you need to:
- Read actual file content before making changes
- Find where a symbol is defined or used
- Understand project structure before analysis
- Verify code changes by running tests or linting
- Check git status and history before making commits
- Create branches and commits for your changes
- Refactor safely (rename/extract/inline) with a dry-run preview"""

# =============================================================================
# Tool Name Mapping (for models trained on different tool sets)
# =============================================================================

TOOL_NAME_MAP = {
    "ls": "vtrip_get_project_skeleton",
    "list_files": "vtrip_get_project_skeleton",
    "list_directory": "vtrip_get_project_skeleton",
    "read_file": "vtrip_read_file",
    "cat": "vtrip_read_file",
    "view_file": "vtrip_read_file",
    "search": "vtrip_search_symbol",
    "find_symbol": "vtrip_search_symbol",
    "grep": "vtrip_grep",
    "ripgrep": "vtrip_grep",
    "rg": "vtrip_grep",
    "search_text": "vtrip_grep",
    "find_in_files": "vtrip_grep",
    "content_search": "vtrip_grep",
    "index_file": "vtrip_index_with_deps",
    "run_command": "vtrip_run_command",
    "execute": "vtrip_run_command",
    "shell": "vtrip_run_command",
    "terminal": "vtrip_run_command",
    "run_tests": "vtrip_run_command",
    "test": "vtrip_run_command",
    "diff": "vtrip_diff_preview",
    "preview": "vtrip_diff_preview",
    "show_diff": "vtrip_diff_preview",
    "apply_edits": "vtrip_apply_edits",
    "edit_files": "vtrip_apply_edits",
    "write_files": "vtrip_apply_edits",
    "multi_edit": "vtrip_apply_edits",
    # Git tools
    "git_status": "vtrip_git_status",
    "status": "vtrip_git_status",
    "git_diff": "vtrip_git_diff",
    "git_log": "vtrip_git_log",
    "log": "vtrip_git_log",
    "git_commit": "vtrip_git_commit",
    "commit": "vtrip_git_commit",
    "git_branch": "vtrip_git_branch",
    "branch": "vtrip_git_branch",
}

# Argument name mapping per tool
ARG_NAME_MAP = {
    "vtrip_read_file": {
        "path": "file_path",
        "filePath": "file_path",
        "dirPath": "file_path",
        "filename": "file_path",
    },
    "vtrip_search_symbol": {
        "query": "name",
        "symbol": "name",
        "search": "name",
    },
}

# =============================================================================
# Configuration
# =============================================================================

MAX_TOOL_TURNS = int(os.environ.get("MAX_TOOL_TURNS", "5"))
MAX_INPUT_TOKENS = int(os.environ.get("MAX_INPUT_TOKENS", "24000"))
MAX_RETRIES = int(os.environ.get("LLM_MAX_RETRIES", "3"))
RETRY_BASE_DELAY = float(os.environ.get("LLM_RETRY_DELAY", "1.0"))

# Token budget reserved for injected RAG/file context (Phase 8.1)
CONTEXT_TOKEN_BUDGET = int(os.environ.get("CONTEXT_TOKEN_BUDGET", "8000"))
# LLM response cache for repeated final answers (Phase 8.4)
ENABLE_LLM_CACHE = os.environ.get("ENABLE_LLM_CACHE", "true").lower() in ("1", "true", "yes")


# =============================================================================
# Helper Functions
# =============================================================================

def _estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars per token for code/English)."""
    if not text:
        return 0
    return len(text) // 4


def _normalize_json(obj: Any) -> str:
    """Normalize JSON for comparison (sorted keys, no extra spaces)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


# Pattern to match <tool_call>...</tool_call> or <tool_name>...</tool_name> tags
_TOOL_CALL_TAG_PATTERN = re.compile(
    r"<tool_call>\s*\{.*?\}\s*</tool_call>|"  # <tool_call>{"name":...}</tool_call>
    r"<(vtrip_\w+)>.*?</\1>",  # <vtrip_read_file>...</vtrip_read_file>
    re.DOTALL | re.IGNORECASE
)


def _strip_tool_call_tags(content: str) -> str:
    """Remove <tool_call> and <vtrip_*> tags from content.

    Models sometimes output both native tool_calls AND text-based tags.
    We strip the text tags to avoid showing them to the user.
    """
    if not content:
        return content

    cleaned = _TOOL_CALL_TAG_PATTERN.sub("", content)
    # Clean up extra whitespace left behind
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _parse_json_safe(s: str) -> dict:
    """Parse JSON safely, return empty dict on error."""
    if not s:
        return {}
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return {}


def _to_openai_messages(state: AgentState, tools_disabled: bool = False) -> list[dict]:
    """Convert LangChain messages to OpenAI format, with truncation."""
    intent = state.get("intent", "code_gen")
    # Phase 16.4: prefer versioned YAML prompt (+ A/B variant); fall back to the
    # hardcoded persona when the intent is not defined in config/prompts.
    from server.agent.prompt_store import get_prompt_store

    variant = state.get("experiment_variant", "default")
    system_prompt = (
        get_prompt_store().get_intent(intent, variant)
        or INTENT_PROMPTS.get(intent, DEFAULT_PROMPT)
    )

    if tools_disabled:
        system_prompt += (
            "\n\nIMPORTANT: You have already used all available tool calls. "
            "DO NOT output any <tool_call> tags or attempt to call tools. "
            "Based on the information you have gathered, provide your final response directly."
        )
    else:
        system_prompt += TOOL_INSTRUCTIONS

    # Phase 8.1: inject assembled RAG/file context into the system prompt so
    # retrieved chunks actually reach the model (previously rag_chunks were
    # retrieved but never used in generation).
    if state.get("rag_chunks") or state.get("active_file"):
        ctx = build_optimal_context(
            state,
            repo_path=state.get("repo_path") or "",
            token_budget=CONTEXT_TOKEN_BUDGET,
        )
        if ctx["context"]:
            system_prompt += (
                "\n\n## Relevant Code Context\n"
                "Use the following retrieved context when answering. "
                "Cite file paths when you rely on it.\n\n"
                + ctx["context"]
            )
            logger.info(
                "generate: injected RAG context (%d parts, ~%d tokens)",
                ctx["parts_included"], ctx["tokens_used"],
            )

    out: list[dict] = [{"role": "system", "content": system_prompt}]

    # Collect messages (skip system messages from client)
    messages: list[dict] = []
    for msg in state.get("messages", []):
        mtype = getattr(msg, "type", None)

        if mtype == "system":
            continue  # Skip client system prompts

        elif mtype == "human":
            content = msg.content
            if isinstance(content, list):
                content = " ".join(
                    item.get("text", "") if isinstance(item, dict) else str(item)
                    for item in content
                )
            messages.append({"role": "user", "content": content or ""})

        elif mtype == "ai":
            item: dict = {"role": "assistant", "content": msg.content or ""}
            tc = (getattr(msg, "additional_kwargs", {}) or {}).get("tool_calls")
            if tc:
                item["tool_calls"] = tc
                item["content"] = None
            messages.append(item)

        elif mtype == "tool":
            tool_call_id = getattr(msg, "tool_call_id", "") or "unknown"
            messages.append({
                "role": "tool",
                "content": msg.content or "",
                "tool_call_id": tool_call_id,
            })

        else:
            content = getattr(msg, "content", str(msg))
            messages.append({"role": "user", "content": content or ""})

    # Truncate old messages to fit token budget (keep most recent)
    total_tokens = _estimate_tokens(system_prompt)
    kept_messages: list[dict] = []

    for msg in reversed(messages):
        content = msg.get("content") or ""
        msg_tokens = _estimate_tokens(content)

        if total_tokens + msg_tokens > MAX_INPUT_TOKENS:
            # Truncate this message if it's the last one we can include
            remaining = MAX_INPUT_TOKENS - total_tokens
            if remaining > 500 and content:
                half = (remaining * 3) // 2
                msg["content"] = content[:half] + "\n...[truncated]...\n" + content[-half:]
                kept_messages.append(msg)
            break

        total_tokens += msg_tokens
        kept_messages.append(msg)

    # Reverse to restore chronological order and add after system prompt
    out.extend(reversed(kept_messages))

    logger.info("_to_openai_messages: %d messages, ~%d tokens", len(out), total_tokens)
    return out


def _merge_tool_call_delta(acc: list[dict], delta_list: list) -> None:
    """Merge streaming tool_call deltas into accumulator."""
    for d in delta_list:
        if hasattr(d, "model_dump"):
            d = d.model_dump()

        idx = d.get("index", 0)

        # Expand accumulator if needed
        while len(acc) <= idx:
            acc.append({
                "id": "",
                "type": "function",
                "function": {"name": "", "arguments": ""},
            })

        slot = acc[idx]

        if d.get("id"):
            slot["id"] = d["id"]
        if d.get("type"):
            slot["type"] = d["type"]

        fn = d.get("function") or {}
        if fn.get("name"):
            slot["function"]["name"] += fn["name"]
        if fn.get("arguments"):
            slot["function"]["arguments"] += fn["arguments"]


def _map_and_validate_tool_calls(tool_calls: list[dict]) -> list[dict]:
    """Map tool names and validate arguments. Filter out invalid calls."""
    valid_calls: list[dict] = []

    for tc in tool_calls:
        fn = tc.get("function") or {}
        name = fn.get("name", "").strip()
        args = _parse_json_safe(fn.get("arguments", "{}"))

        # Skip empty tool calls
        if not name:
            logger.warning("Skipping tool call with empty name")
            continue

        # Map tool name if needed
        original_name = name
        if name in TOOL_NAME_MAP:
            name = TOOL_NAME_MAP[name]
            logger.info("Mapped tool: %s -> %s", original_name, name)

        # Skip unknown tools
        if name not in MCP_TOOL_NAMES:
            logger.warning("Skipping unknown tool: %s", name)
            continue

        # Map argument names
        arg_map = ARG_NAME_MAP.get(name, {})
        mapped_args = {}
        for k, v in args.items():
            mapped_key = arg_map.get(k, k)
            mapped_args[mapped_key] = v

        # Validate required arguments
        if name == "vtrip_read_file":
            file_path = mapped_args.get("file_path", "")
            if not file_path:
                logger.warning("Skipping vtrip_read_file: empty file_path")
                continue
            # Set defaults for optional args
            mapped_args.setdefault("start_line", 1)
            mapped_args.setdefault("end_line", 150)

        elif name == "vtrip_search_symbol":
            symbol_name = mapped_args.get("name", "")
            if not symbol_name:
                logger.warning("Skipping vtrip_search_symbol: empty name")
                continue
            mapped_args.setdefault("type_filter", "any")

        elif name == "vtrip_get_project_skeleton":
            mapped_args.setdefault("include_methods", True)

        elif name == "vtrip_index_with_deps":
            file_path = mapped_args.get("file_path", "")
            if not file_path:
                logger.warning("Skipping vtrip_index_with_deps: empty file_path")
                continue
            mapped_args.setdefault("depth", 2)

        valid_calls.append({
            "id": tc.get("id") or f"call_{len(valid_calls)}",
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(mapped_args),
            },
        })

    return valid_calls


def _normalize_call_key(name: str, arguments: str) -> str:
    """Create normalized call key for deduplication.

    Parses JSON once, sorts keys, and creates canonical form.
    Returns name:normalized_args string.
    """
    args = _parse_json_safe(arguments)
    return f"{name}:{_normalize_json(args)}"


def _get_already_called_tools(state: AgentState) -> set[str]:
    """Get normalized signatures of already-called tools.

    Single pass through messages, O(n) complexity.
    """
    already_called: set[str] = set()

    for msg in state.get("messages", []):
        if getattr(msg, "type", None) != "ai":
            continue
        tc_list = (getattr(msg, "additional_kwargs", {}) or {}).get("tool_calls", [])
        for tc in tc_list:
            fn = tc.get("function", {})
            name = fn.get("name", "")
            args_str = fn.get("arguments", "{}")
            call_key = _normalize_call_key(name, args_str)
            already_called.add(call_key)

    return already_called


def _deduplicate_tool_calls(
    tool_calls: list[dict],
    already_called: set[str],
) -> list[dict]:
    """Remove duplicate tool calls."""
    deduped: list[dict] = []

    for tc in tool_calls:
        fn = tc.get("function", {})
        name = fn.get("name", "")
        args_str = fn.get("arguments", "{}")
        call_key = _normalize_call_key(name, args_str)

        if call_key in already_called:
            logger.warning("Skipping duplicate tool call: %s", name)
            continue

        already_called.add(call_key)
        deduped.append(tc)

    return deduped


# =============================================================================
# Main Generate Function
# =============================================================================

async def generate(
    state: AgentState,
    vllm_client: AsyncOpenAI,
    model: str,
    sse_callback=None,
) -> dict:
    """Generate response, possibly with tool calls."""
    tool_turns_used = state.get("tool_turns_used", 0)

    # Check tool turn limit
    tools_disabled = tool_turns_used >= MAX_TOOL_TURNS
    if tools_disabled:
        logger.warning(
            "Tool turn limit reached (%d/%d), generating without tools",
            tool_turns_used, MAX_TOOL_TURNS
        )
        all_tools = None
    else:
        # Merge MCP tools with client tools (MCP takes priority)
        client_tools = state.get("client_tools") or []
        extra_tools = [
            t for t in client_tools
            if t.get("function", {}).get("name") not in MCP_TOOL_NAMES
        ]
        all_tools = MCP_TOOLS + extra_tools if MCP_TOOLS or extra_tools else None

    messages = _to_openai_messages(state, tools_disabled=tools_disabled)

    # Phase 8.4: serve a cached final answer for an identical recent context.
    # Only used for final responses (cache only stores turns with no tool calls).
    # Disabled when the answer depends on repo context (RAG chunks / active file),
    # since the cache key does not capture that context and could serve a stale
    # answer for a different repository state.
    intent = state.get("intent", "code_gen")
    context_sensitive = bool(state.get("rag_chunks") or state.get("active_file"))
    llm_cache = get_llm_cache() if (ENABLE_LLM_CACHE and not context_sensitive) else None
    cache_msgs = [m for m in messages if m.get("role") != "system"]
    if llm_cache is not None:
        cached = llm_cache.get(cache_msgs, all_tools, intent)
        if cached is not None:
            draft = cached.get("draft", "")
            if sse_callback and draft:
                await sse_callback("content", draft)
            return {
                "draft": draft,
                "pending_tool_calls": [],
                "tool_turns_used": tool_turns_used,
            }

    # Calculate max_tokens dynamically based on estimated input
    # Model context: ~32k, reserve enough for output
    estimated_input = sum(_estimate_tokens(m.get("content") or "") for m in messages)
    MODEL_CONTEXT = 32000
    max_tokens = min(8192, max(1024, MODEL_CONTEXT - estimated_input - 1000))

    logger.info("generate: estimated_input=%d, max_tokens=%d", estimated_input, max_tokens)

    # Build request kwargs
    kwargs: dict = {
        "model": model,
        "messages": messages,
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }

    # Only add tools if we have them
    if all_tools:
        kwargs["tools"] = all_tools
        tool_choice = state.get("tool_choice")
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice

    tool_names = [t["function"]["name"] for t in (all_tools or [])]
    logger.info("generate: tools=%s, tool_turns=%d/%d", tool_names, tool_turns_used, MAX_TOOL_TURNS)

    # Phase 11.3/11.4: fail fast + degrade gracefully when the vLLM circuit is
    # open instead of retrying a backend we already know is down.
    vllm_circuit = get_circuit_breaker("vllm")
    if not vllm_circuit.allow():
        logger.warning("vLLM circuit open — serving graceful-degradation fallback")
        draft = llm_unavailable_draft("vLLM circuit open")["draft"]
        if sse_callback:
            await sse_callback("content", draft)
        return {"draft": draft, "pending_tool_calls": [], "tool_turns_used": tool_turns_used,
                "degraded": True}

    # Stream response with retry logic
    content_buf: list[str] = []
    tool_calls_acc: list[dict] = []
    last_error: Exception | None = None

    for attempt in range(MAX_RETRIES):
        content_buf.clear()
        tool_calls_acc.clear()

        try:
            stream = await vllm_client.chat.completions.create(**kwargs)

            async for chunk in stream:
                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta

                # Collect content
                if getattr(delta, "content", None):
                    token = delta.content
                    content_buf.append(token)
                    if sse_callback:
                        await sse_callback("content", token)

                # Collect tool calls
                if getattr(delta, "tool_calls", None):
                    _merge_tool_call_delta(tool_calls_acc, delta.tool_calls)

            # Success - record on circuit breaker and break out of retry loop
            vllm_circuit.record_success()
            break

        except Exception as e:
            last_error = e
            vllm_circuit.record_failure()
            logger.warning(
                "vLLM generation failed (attempt %d/%d): %s",
                attempt + 1, MAX_RETRIES, e
            )

            if attempt < MAX_RETRIES - 1 and vllm_circuit.allow():
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                logger.info("Retrying in %.1fs...", delay)
                await asyncio.sleep(delay)
            else:
                logger.error("vLLM generation failed after %d attempts", MAX_RETRIES)
                # Graceful degradation rather than surfacing a raw stack/error.
                draft = llm_unavailable_draft(str(e))["draft"]
                if sse_callback:
                    await sse_callback("content", draft)
                return {
                    "draft": draft,
                    "pending_tool_calls": [],
                    "tool_turns_used": tool_turns_used,
                    "degraded": True,
                }

    # Process results
    draft = "".join(content_buf)

    # Strip <tool_call> tags from content (model may output both native + text tags)
    original_len = len(draft)
    draft = _strip_tool_call_tags(draft)
    if len(draft) != original_len:
        logger.info("generate: stripped tool_call tags from content (%d -> %d chars)", original_len, len(draft))

    # Fallback: if content is empty and no tool calls, provide a message
    if not draft and not tool_calls_acc:
        if tools_disabled:
            draft = (
                "Tôi đã thu thập đủ thông tin từ các công cụ. "
                "Tuy nhiên, tôi cần thêm context để hoàn thành yêu cầu. "
                "Vui lòng cung cấp thêm chi tiết hoặc thử lại với câu hỏi cụ thể hơn."
            )
            logger.warning("generate: empty response with tools disabled, using fallback message")
        else:
            logger.warning("generate: empty response, no content or tool calls")

    logger.info(
        "generate: content_len=%d, raw_tool_calls=%d",
        len(draft), len(tool_calls_acc)
    )

    # Validate and map tool calls
    valid_tool_calls = _map_and_validate_tool_calls(tool_calls_acc)

    # Deduplicate
    already_called = _get_already_called_tools(state)
    final_tool_calls = _deduplicate_tool_calls(valid_tool_calls, already_called)

    if final_tool_calls:
        logger.info(
            "generate: final_tool_calls=%d (raw=%d, valid=%d)",
            len(final_tool_calls), len(tool_calls_acc), len(valid_tool_calls)
        )

    # Increment tool_turns_used if we're returning tool calls
    new_tool_turns = tool_turns_used + 1 if final_tool_calls else tool_turns_used

    # Phase 8.4: cache only final answers (no pending tool calls).
    if llm_cache is not None and not final_tool_calls and draft:
        llm_cache.set(cache_msgs, all_tools, intent, {"draft": draft})

    return {
        "draft": draft,
        "pending_tool_calls": final_tool_calls,
        "tool_turns_used": new_tool_turns,
    }
