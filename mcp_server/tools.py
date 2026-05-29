"""MCP Tools — Section 5: read_file, search_symbol, run_command.

Core tools for AI coding agent:
- read_file: Read file content
- search_symbol: Find symbols in codebase
- run_command: Execute shell commands (tests, lint, build)
"""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
from pathlib import Path

from mcp_server.models import ExtractionMode, SKIP_DIRS, SKIP_EXTENSIONS
from mcp_server.plugins.registry import PluginRegistry

logger = logging.getLogger("mcp_server.tools")


def read_file(
    repo_path: str,
    file_path: str,
    start_line: int = 1,
    end_line: int = 150,
) -> dict:
    """Section 5, Tool: read_file.

    Read a contiguous range of lines from a file.
    Critical constraint: result must NEVER be uploaded to Qdrant.
    """
    abs_path = os.path.join(repo_path, file_path)

    if not os.path.isfile(abs_path):
        return {"error": f"File not found: {file_path}"}

    # Reject paths outside REPO_PATH
    real_repo = os.path.realpath(repo_path)
    real_file = os.path.realpath(abs_path)
    if not real_file.startswith(real_repo):
        return {"error": f"File not found: {file_path}"}

    try:
        with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
    except OSError as e:
        return {"error": f"Cannot read file: {e}"}

    total_lines = len(all_lines)
    start = max(1, start_line)
    end = min(total_lines, end_line)

    selected = all_lines[start - 1 : end]
    content = "".join(selected)

    return {
        "content": content,
        "start_line": start,
        "end_line": end,
        "total_lines": total_lines,
        "file_path": file_path,
    }


def search_symbol(
    repo_path: str,
    registry: PluginRegistry,
    name: str,
    type_filter: str = "any",
) -> list[dict]:
    """Section 5, Tool: search_symbol.

    Locate a class, function, or method by name anywhere in the repository.
    Uses the same skip rules as get_project_skeleton.
    """
    matches: list[dict] = []

    for root, dirs, files in os.walk(repo_path):
        # Prune blocked directories
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]

        for fname in files:
            # Skip blocked extensions
            ext = Path(fname).suffix.lower()
            if ext in SKIP_EXTENSIONS:
                continue

            full_path = os.path.join(root, fname)
            rel_path = os.path.relpath(full_path, repo_path).replace("\\", "/")

            plugin = registry.get_plugin(full_path)

            try:
                with open(full_path, "rb") as f:
                    source = f.read()
            except OSError:
                continue

            try:
                chunks = plugin.extract_chunks(rel_path, source, ExtractionMode.names_only)
            except Exception as e:
                logger.warning("Failed to parse %s: %s", rel_path, e)
                continue

            for chunk in chunks:
                if name.lower() not in chunk.symbol_name.lower():
                    continue

                # Apply type_filter
                if type_filter != "any":
                    type_map = {
                        "class": "grouping",
                        "function": "callable",
                        "method": "callable",
                    }
                    expected = type_map.get(type_filter)
                    if expected and chunk.chunk_type.value != expected:
                        continue

                matches.append({
                    "symbol_name": chunk.symbol_name,
                    "chunk_type": chunk.chunk_type.value,
                    "file_path": chunk.file_path,
                    "start_line": chunk.start_line,
                    "lang": chunk.lang,
                })

    return matches


# =============================================================================
# Code Execution Tool
# =============================================================================

# Allowed commands whitelist for safety
ALLOWED_COMMANDS = {
    # Build & Test
    "mvn", "gradle", "gradlew", "./gradlew",
    "npm", "npx", "yarn", "pnpm",
    "pytest", "python", "python3",
    "go", "cargo", "dotnet",
    "make",
    # Lint & Format
    "eslint", "prettier", "black", "ruff", "flake8", "mypy",
    "golint", "gofmt", "rustfmt", "checkstyle",
    # Git (read-only)
    "git",
    # Other
    "cat", "head", "tail", "grep", "find", "ls", "wc", "echo",
}

# Dangerous patterns to block
BLOCKED_PATTERNS = [
    "rm -rf", "rm -r", "rmdir",
    "> /dev", "| rm", "&& rm",
    "sudo", "su ",
    "curl | sh", "wget | sh",
    "eval", "exec",
    ":(){", "fork",
    "chmod 777", "chmod -R",
    "dd if=", "mkfs",
    "shutdown", "reboot", "halt",
    "passwd", "useradd", "userdel",
]

MAX_OUTPUT_SIZE = 50000  # 50KB max output
DEFAULT_TIMEOUT = 60  # 60 seconds


def run_command(
    repo_path: str,
    command: str,
    timeout: int = DEFAULT_TIMEOUT,
    working_dir: str | None = None,
) -> dict:
    """Execute a shell command in the repository.

    Safety measures:
    - Whitelist of allowed commands
    - Block dangerous patterns
    - Output size limit
    - Timeout
    - Runs in repo directory only

    Args:
        repo_path: Repository root path
        command: Command to execute
        timeout: Max execution time in seconds
        working_dir: Subdirectory to run in (relative to repo_path)

    Returns:
        {stdout, stderr, exit_code, truncated, command}
    """
    # Validate command is not empty
    if not command or not command.strip():
        return {"error": "Empty command", "exit_code": -1}

    # Check for dangerous patterns
    cmd_lower = command.lower()
    for pattern in BLOCKED_PATTERNS:
        if pattern in cmd_lower:
            return {
                "error": f"Blocked pattern detected: {pattern}",
                "exit_code": -1,
                "command": command,
            }

    # Parse command to check first word
    try:
        parts = shlex.split(command)
        if not parts:
            return {"error": "Invalid command", "exit_code": -1}
    except ValueError as e:
        return {"error": f"Command parse error: {e}", "exit_code": -1}

    # Check if base command is allowed
    base_cmd = os.path.basename(parts[0])
    if base_cmd not in ALLOWED_COMMANDS:
        return {
            "error": f"Command not in whitelist: {base_cmd}. Allowed: {sorted(ALLOWED_COMMANDS)}",
            "exit_code": -1,
            "command": command,
        }

    # Git: only allow read-only subcommands
    if base_cmd == "git":
        git_readonly = {"status", "log", "diff", "show", "branch", "blame", "ls-files"}
        if len(parts) > 1 and parts[1] not in git_readonly:
            return {
                "error": f"Git subcommand not allowed: {parts[1]}. Allowed: {git_readonly}",
                "exit_code": -1,
            }

    # Determine working directory
    cwd = repo_path
    if working_dir:
        cwd = os.path.join(repo_path, working_dir)
        # Validate it's within repo
        real_cwd = os.path.realpath(cwd)
        real_repo = os.path.realpath(repo_path)
        if not real_cwd.startswith(real_repo):
            return {"error": "working_dir must be within repository", "exit_code": -1}
        if not os.path.isdir(cwd):
            return {"error": f"Directory not found: {working_dir}", "exit_code": -1}

    logger.info("run_command: %s (cwd=%s)", command, cwd)

    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "CI": "true"},  # Signal CI environment
        )

        stdout = result.stdout
        stderr = result.stderr
        truncated = False

        # Truncate output if too large
        if len(stdout) > MAX_OUTPUT_SIZE:
            stdout = stdout[:MAX_OUTPUT_SIZE] + f"\n... [truncated, {len(result.stdout)} bytes total]"
            truncated = True
        if len(stderr) > MAX_OUTPUT_SIZE:
            stderr = stderr[:MAX_OUTPUT_SIZE] + f"\n... [truncated, {len(result.stderr)} bytes total]"
            truncated = True

        return {
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": result.returncode,
            "truncated": truncated,
            "command": command,
        }

    except subprocess.TimeoutExpired:
        return {
            "error": f"Command timed out after {timeout}s",
            "exit_code": -1,
            "command": command,
        }
    except Exception as e:
        return {
            "error": f"Execution failed: {e}",
            "exit_code": -1,
            "command": command,
        }


# =============================================================================
# Diff Preview Tool
# =============================================================================

def diff_preview(
    repo_path: str,
    file_path: str,
    new_content: str,
) -> dict:
    """Generate unified diff showing proposed changes.

    Args:
        repo_path: Repository root path
        file_path: Path relative to repo root
        new_content: Proposed new file content

    Returns:
        {diff, file_path, exists, lines_added, lines_removed}
    """
    import difflib

    abs_path = os.path.join(repo_path, file_path)

    # Validate path is within repo
    real_repo = os.path.realpath(repo_path)
    real_file = os.path.realpath(abs_path)
    if not real_file.startswith(real_repo):
        return {"error": f"Path outside repository: {file_path}"}

    # Read existing content (empty if new file)
    old_lines: list[str] = []
    exists = os.path.isfile(abs_path)

    if exists:
        try:
            with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                old_lines = f.read().splitlines(keepends=True)
        except OSError as e:
            return {"error": f"Cannot read file: {e}"}

    new_lines = new_content.splitlines(keepends=True)

    # Generate unified diff
    diff = list(difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=f"a/{file_path}",
        tofile=f"b/{file_path}",
        lineterm="",
    ))

    # Count changes
    lines_added = sum(1 for line in diff if line.startswith("+") and not line.startswith("+++"))
    lines_removed = sum(1 for line in diff if line.startswith("-") and not line.startswith("---"))

    return {
        "diff": "\n".join(diff) if diff else "(no changes)",
        "file_path": file_path,
        "exists": exists,
        "lines_added": lines_added,
        "lines_removed": lines_removed,
    }


# =============================================================================
# Multi-file Edit Tool
# =============================================================================

def apply_edits(
    repo_path: str,
    edits: list[dict],
    dry_run: bool = False,
) -> dict:
    """Apply multiple file edits atomically.

    Args:
        repo_path: Repository root path
        edits: List of {file_path, new_content} or {file_path, search, replace}
        dry_run: If True, only preview changes without applying

    Returns:
        {success, results: [{file_path, status, diff}], errors}
    """
    import difflib

    results = []
    errors = []
    real_repo = os.path.realpath(repo_path)

    # Validate all paths first
    for edit in edits:
        file_path = edit.get("file_path", "")
        if not file_path:
            errors.append({"file_path": "", "error": "Missing file_path"})
            continue

        abs_path = os.path.join(repo_path, file_path)
        real_file = os.path.realpath(abs_path)
        if not real_file.startswith(real_repo):
            errors.append({"file_path": file_path, "error": "Path outside repository"})

    if errors:
        return {"success": False, "results": [], "errors": errors}

    # Process edits
    for edit in edits:
        file_path = edit.get("file_path", "")
        abs_path = os.path.join(repo_path, file_path)

        # Read existing content
        old_content = ""
        exists = os.path.isfile(abs_path)
        if exists:
            try:
                with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                    old_content = f.read()
            except OSError as e:
                errors.append({"file_path": file_path, "error": str(e)})
                continue

        # Determine new content
        if "new_content" in edit:
            new_content = edit["new_content"]
        elif "search" in edit and "replace" in edit:
            if edit["search"] not in old_content:
                errors.append({
                    "file_path": file_path,
                    "error": f"Search string not found: {edit['search'][:50]}..."
                })
                continue
            new_content = old_content.replace(edit["search"], edit["replace"], 1)
        else:
            errors.append({"file_path": file_path, "error": "Missing new_content or search/replace"})
            continue

        # Generate diff
        old_lines = old_content.splitlines(keepends=True)
        new_lines = new_content.splitlines(keepends=True)
        diff = list(difflib.unified_diff(
            old_lines, new_lines,
            fromfile=f"a/{file_path}", tofile=f"b/{file_path}",
        ))

        if not dry_run and diff:
            # Ensure parent directory exists
            parent = os.path.dirname(abs_path)
            if parent and not os.path.exists(parent):
                os.makedirs(parent, exist_ok=True)

            # Write file
            try:
                with open(abs_path, "w", encoding="utf-8") as f:
                    f.write(new_content)
            except OSError as e:
                errors.append({"file_path": file_path, "error": str(e)})
                continue

        results.append({
            "file_path": file_path,
            "status": "preview" if dry_run else ("modified" if diff else "unchanged"),
            "diff": "".join(diff) if diff else "(no changes)",
        })

    return {
        "success": len(errors) == 0,
        "results": results,
        "errors": errors,
        "dry_run": dry_run,
    }


# =============================================================================
# Git Integration Tools
# =============================================================================

def git_status(repo_path: str) -> dict:
    """Get git status of the repository."""
    result = subprocess.run(
        ["git", "status", "--porcelain", "-b"],
        cwd=repo_path,
        capture_output=True,
        text=True,
        timeout=30,
    )

    if result.returncode != 0:
        return {"error": result.stderr or "git status failed"}

    lines = result.stdout.strip().split("\n")
    branch = ""
    staged = []
    modified = []
    untracked = []

    for line in lines:
        if line.startswith("##"):
            branch = line[3:].split("...")[0]
        elif line.startswith("A "):
            staged.append(line[3:])
        elif line.startswith("M "):
            staged.append(line[3:])
        elif line.startswith(" M"):
            modified.append(line[3:])
        elif line.startswith("??"):
            untracked.append(line[3:])

    return {
        "branch": branch,
        "staged": staged,
        "modified": modified,
        "untracked": untracked,
        "clean": len(staged) == 0 and len(modified) == 0,
    }


def git_diff(repo_path: str, file_path: str | None = None, staged: bool = False) -> dict:
    """Get git diff for file or entire repo."""
    cmd = ["git", "diff"]
    if staged:
        cmd.append("--cached")
    if file_path:
        cmd.append("--")
        cmd.append(file_path)

    result = subprocess.run(
        cmd,
        cwd=repo_path,
        capture_output=True,
        text=True,
        timeout=30,
    )

    if result.returncode != 0:
        return {"error": result.stderr or "git diff failed"}

    diff = result.stdout
    if len(diff) > MAX_OUTPUT_SIZE:
        diff = diff[:MAX_OUTPUT_SIZE] + "\n... [truncated]"

    return {
        "diff": diff or "(no changes)",
        "file_path": file_path,
        "staged": staged,
    }


def git_log(repo_path: str, count: int = 10, file_path: str | None = None) -> dict:
    """Get recent git commits."""
    cmd = ["git", "log", f"-{min(count, 50)}", "--oneline", "--no-decorate"]
    if file_path:
        cmd.append("--")
        cmd.append(file_path)

    result = subprocess.run(
        cmd,
        cwd=repo_path,
        capture_output=True,
        text=True,
        timeout=30,
    )

    if result.returncode != 0:
        return {"error": result.stderr or "git log failed"}

    commits = []
    for line in result.stdout.strip().split("\n"):
        if line:
            parts = line.split(" ", 1)
            commits.append({
                "hash": parts[0],
                "message": parts[1] if len(parts) > 1 else "",
            })

    return {"commits": commits, "file_path": file_path}


def git_commit(repo_path: str, message: str, files: list[str] | None = None) -> dict:
    """Stage files and create a commit."""
    # Stage files
    if files:
        for f in files:
            result = subprocess.run(
                ["git", "add", f],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                return {"error": f"Failed to stage {f}: {result.stderr}"}
    else:
        # Stage all changes
        result = subprocess.run(
            ["git", "add", "-A"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            return {"error": f"Failed to stage files: {result.stderr}"}

    # Create commit
    result = subprocess.run(
        ["git", "commit", "-m", message],
        cwd=repo_path,
        capture_output=True,
        text=True,
        timeout=30,
    )

    if result.returncode != 0:
        return {"error": result.stderr or "git commit failed"}

    # Get commit hash
    hash_result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_path,
        capture_output=True,
        text=True,
        timeout=10,
    )

    return {
        "success": True,
        "message": message,
        "hash": hash_result.stdout.strip() if hash_result.returncode == 0 else "",
        "output": result.stdout,
    }


def git_branch(repo_path: str, name: str | None = None, checkout: bool = False) -> dict:
    """List branches or create/checkout a branch."""
    if name is None:
        # List branches
        result = subprocess.run(
            ["git", "branch", "-a"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            return {"error": result.stderr}

        branches = []
        current = ""
        for line in result.stdout.strip().split("\n"):
            line = line.strip()
            if line.startswith("*"):
                current = line[2:]
                branches.append(current)
            elif line:
                branches.append(line)

        return {"branches": branches, "current": current}

    # Create or checkout branch
    if checkout:
        cmd = ["git", "checkout", "-b", name]
    else:
        cmd = ["git", "branch", name]

    result = subprocess.run(
        cmd,
        cwd=repo_path,
        capture_output=True,
        text=True,
        timeout=30,
    )

    if result.returncode != 0:
        # Try checkout existing branch
        if checkout:
            result = subprocess.run(
                ["git", "checkout", name],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                return {"success": True, "branch": name, "action": "checkout"}
        return {"error": result.stderr}

    return {"success": True, "branch": name, "action": "created" if not checkout else "created_and_checkout"}
