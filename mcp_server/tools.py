"""MCP Tools — Section 5: read_file, search_symbol, run_command.

Core tools for AI coding agent:
- read_file: Read file content
- search_symbol: Find symbols in codebase
- run_command: Execute shell commands (tests, lint, build)
"""

from __future__ import annotations

import fnmatch
import logging
import os
import re
import shlex
import subprocess
from pathlib import Path

from mcp_server.models import ExtractionMode, SKIP_DIRS, SKIP_EXTENSIONS
from mcp_server.plugins.registry import PluginRegistry

logger = logging.getLogger("mcp_server.tools")

# grep_content limits
GREP_MAX_FILE_BYTES = 2_000_000   # skip files larger than ~2MB
GREP_MAX_RESULTS_CAP = 200        # hard cap regardless of caller request


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


def _matches_skip_extension(fname: str) -> bool:
    """True if a filename matches SKIP_EXTENSIONS (handles multi-part like .min.js)."""
    lower = fname.lower()
    if Path(lower).suffix in SKIP_EXTENSIONS:
        return True
    # Multi-part suffix chain, e.g. "a.min.js" -> ".min.js"
    suffixes = "".join(Path(lower).suffixes)
    return any(suffixes.endswith(skip) for skip in SKIP_EXTENSIONS)


def grep_content(
    repo_path: str,
    pattern: str,
    path_glob: str | None = None,
    ignore_case: bool = False,
    max_results: int = 100,
) -> dict:
    """Full-text / regex content search across the repository (agentic search).

    Pure-Python, ripgrep-style. Reads files fresh from the local filesystem so
    results are never stale, and the working copy stays on the user's machine
    (no upload to Qdrant). Skips SKIP_DIRS, SKIP_EXTENSIONS, binary and
    oversized files.

    Args:
        repo_path: Repository root path
        pattern: Regular expression to search for
        path_glob: Optional glob to restrict files (e.g. "**/*.py", "src/*.go")
        ignore_case: Case-insensitive match
        max_results: Max matching lines to return (capped at 200)

    Returns:
        {matches: [{file_path, line_number, line}], total, files_scanned,
         truncated, pattern}
    """
    if not pattern:
        return {"error": "Empty pattern"}

    try:
        regex = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
    except re.error as e:
        return {"error": f"Invalid regex: {e}"}

    limit = max(1, min(max_results, GREP_MAX_RESULTS_CAP))
    real_repo = os.path.realpath(repo_path)

    matches: list[dict] = []
    files_scanned = 0
    truncated = False

    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]

        for fname in files:
            if _matches_skip_extension(fname):
                continue

            full_path = os.path.join(root, fname)
            rel_path = os.path.relpath(full_path, repo_path).replace("\\", "/")

            if path_glob and not (
                fnmatch.fnmatch(rel_path, path_glob)
                or fnmatch.fnmatch(fname, path_glob)
            ):
                continue

            # Path traversal guard (symlinks escaping the repo)
            if not os.path.realpath(full_path).startswith(real_repo):
                continue

            try:
                if os.path.getsize(full_path) > GREP_MAX_FILE_BYTES:
                    continue
            except OSError:
                continue

            try:
                with open(full_path, "r", encoding="utf-8", errors="strict") as f:
                    files_scanned += 1
                    for lineno, line in enumerate(f, 1):
                        if regex.search(line):
                            matches.append({
                                "file_path": rel_path,
                                "line_number": lineno,
                                "line": line.rstrip("\n")[:400],
                            })
                            if len(matches) >= limit:
                                truncated = True
                                break
            except (OSError, UnicodeDecodeError):
                # Unreadable or binary file — skip
                continue

            if truncated:
                break
        if truncated:
            break

    return {
        "matches": matches,
        "total": len(matches),
        "files_scanned": files_scanned,
        "truncated": truncated,
        "pattern": pattern,
    }


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
    # Phase 10.1: delegate to CommandSandbox (whitelist by category, dangerous
    # pattern regex, env isolation, path-traversal guard, audit log). The legacy
    # ALLOWED_COMMANDS/BLOCKED_PATTERNS path is superseded by the sandbox.
    from mcp_server.sandbox import CommandSandbox, SandboxConfig, get_sandbox

    sandbox = get_sandbox()
    # Honor a caller-specified timeout without mutating the shared singleton.
    if timeout and timeout != sandbox.config.timeout_seconds:
        sandbox = CommandSandbox(
            SandboxConfig(timeout_seconds=timeout, max_output_bytes=MAX_OUTPUT_SIZE)
        )

    result = sandbox.execute(command, repo_path, working_dir)

    if result.success:
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.exit_code,
            "truncated": result.truncated,
            "command": result.command,
        }
    return {
        "error": result.error,
        "exit_code": -1,
        "command": result.command,
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


# =============================================================================
# Test Execution Tool (Phase 7.2)
# =============================================================================

TEST_COMMANDS = {
    "pytest": "pytest --tb=short -v",
    "jest": "npx jest --colors",
    "mocha": "npx mocha",
    "junit": "mvn test -Dtest=",
    "gradle": "./gradlew test",
    "go": "go test -v",
    "cargo": "cargo test",
}

TEST_FILE_PATTERNS = {
    "pytest": ["test_*.py", "*_test.py", "tests/*.py"],
    "jest": ["*.test.js", "*.test.ts", "*.spec.js", "*.spec.ts"],
    "mocha": ["test/*.js", "*.test.js"],
    "junit": ["*Test.java", "*Tests.java"],
    "go": ["*_test.go"],
    "cargo": ["tests/*.rs"],
}


def _detect_test_framework(repo_path: str) -> str:
    """Auto-detect test framework from project files."""
    # Check for pytest
    if os.path.exists(os.path.join(repo_path, "pytest.ini")) or \
       os.path.exists(os.path.join(repo_path, "pyproject.toml")):
        return "pytest"

    # Check for Jest
    pkg_json = os.path.join(repo_path, "package.json")
    if os.path.exists(pkg_json):
        try:
            with open(pkg_json, "r") as f:
                import json
                pkg = json.load(f)
                if "jest" in pkg.get("devDependencies", {}) or \
                   "jest" in pkg.get("dependencies", {}):
                    return "jest"
                if "mocha" in pkg.get("devDependencies", {}):
                    return "mocha"
        except Exception:
            pass

    # Check for Go
    if os.path.exists(os.path.join(repo_path, "go.mod")):
        return "go"

    # Check for Cargo
    if os.path.exists(os.path.join(repo_path, "Cargo.toml")):
        return "cargo"

    # Check for Gradle
    if os.path.exists(os.path.join(repo_path, "build.gradle")) or \
       os.path.exists(os.path.join(repo_path, "build.gradle.kts")):
        return "gradle"

    # Check for Maven/JUnit
    if os.path.exists(os.path.join(repo_path, "pom.xml")):
        return "junit"

    # Default to pytest
    return "pytest"


def _parse_test_output(output: str, framework: str) -> dict:
    """Parse test output to extract results."""
    import re

    result = {
        "total": 0,
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "failure_details": [],
    }

    if framework == "pytest":
        # Parse pytest summary line: "5 passed, 2 failed, 1 skipped"
        match = re.search(r"(\d+) passed", output)
        if match:
            result["passed"] = int(match.group(1))
        match = re.search(r"(\d+) failed", output)
        if match:
            result["failed"] = int(match.group(1))
        match = re.search(r"(\d+) skipped", output)
        if match:
            result["skipped"] = int(match.group(1))
        result["total"] = result["passed"] + result["failed"] + result["skipped"]

        # Extract failure details
        failures = re.findall(r"FAILED ([\w/.:]+)", output)
        result["failure_details"] = failures

    elif framework == "jest":
        # Parse Jest output
        match = re.search(r"Tests:\s+(\d+) passed", output)
        if match:
            result["passed"] = int(match.group(1))
        match = re.search(r"Tests:\s+\d+ passed,\s+(\d+) failed", output)
        if match:
            result["failed"] = int(match.group(1))
        result["total"] = result["passed"] + result["failed"]

    elif framework == "go":
        # Parse Go test output
        passed = output.count("--- PASS:")
        failed = output.count("--- FAIL:")
        result["passed"] = passed
        result["failed"] = failed
        result["total"] = passed + failed

    elif framework == "cargo":
        # Parse Cargo test output
        match = re.search(r"(\d+) passed; (\d+) failed", output)
        if match:
            result["passed"] = int(match.group(1))
            result["failed"] = int(match.group(2))
            result["total"] = result["passed"] + result["failed"]

    elif framework == "junit":
        # Parse Maven/JUnit output: "Tests run: 5, Failures: 1, Errors: 0, Skipped: 1"
        match = re.search(r"Tests run:\s*(\d+),\s*Failures:\s*(\d+),\s*Errors:\s*(\d+),\s*Skipped:\s*(\d+)", output)
        if match:
            total = int(match.group(1))
            failures = int(match.group(2))
            errors = int(match.group(3))
            skipped = int(match.group(4))
            result["total"] = total
            result["failed"] = failures + errors
            result["skipped"] = skipped
            result["passed"] = total - failures - errors - skipped

        # Extract failure details from Maven output
        failures = re.findall(r"(?:FAILURE!|Failed tests:)\s*([\w.]+)", output)
        result["failure_details"] = failures

    elif framework == "mocha":
        # Parse Mocha output: "5 passing", "2 failing"
        match = re.search(r"(\d+) passing", output)
        if match:
            result["passed"] = int(match.group(1))
        match = re.search(r"(\d+) failing", output)
        if match:
            result["failed"] = int(match.group(1))
        match = re.search(r"(\d+) pending", output)
        if match:
            result["skipped"] = int(match.group(1))
        result["total"] = result["passed"] + result["failed"] + result["skipped"]

    elif framework == "gradle":
        # Parse Gradle test output: "5 tests completed, 2 failed"
        # Or: "BUILD SUCCESSFUL" with test counts
        match = re.search(r"(\d+) tests completed", output)
        if match:
            result["total"] = int(match.group(1))
        match = re.search(r"(\d+) failed", output)
        if match:
            result["failed"] = int(match.group(1))
        match = re.search(r"(\d+) skipped", output)
        if match:
            result["skipped"] = int(match.group(1))
        result["passed"] = result["total"] - result["failed"] - result["skipped"]

        # Extract failure details
        failures = re.findall(r"(?:FAILED|> .+) > (\w+)", output)
        result["failure_details"] = failures

    return result


def run_tests(
    repo_path: str,
    test_file: str | None = None,
    test_name: str | None = None,
    framework: str = "auto",
    timeout: int = 300,
) -> dict:
    """Run tests and parse results.

    Args:
        repo_path: Repository root path
        test_file: Specific test file to run (optional)
        test_name: Specific test name/pattern (optional)
        framework: Test framework (auto, pytest, jest, junit, go, cargo)
        timeout: Test timeout in seconds

    Returns:
        {stdout, stderr, exit_code, tests_run, tests_passed, tests_failed, failures}
    """
    # Auto-detect framework
    if framework == "auto":
        framework = _detect_test_framework(repo_path)

    if framework not in TEST_COMMANDS:
        return {"error": f"Unsupported test framework: {framework}"}

    # Build command
    cmd = TEST_COMMANDS[framework]

    if test_file:
        if framework == "pytest":
            cmd += f" {test_file}"
        elif framework == "jest":
            cmd += f" {test_file}"
        elif framework == "junit":
            # Extract class name from file
            class_name = os.path.splitext(os.path.basename(test_file))[0]
            cmd += class_name
        elif framework == "gradle":
            # Gradle: --tests "ClassName" or --tests "ClassName.methodName"
            class_name = os.path.splitext(os.path.basename(test_file))[0]
            cmd += f" --tests '{class_name}'"
        elif framework == "go":
            cmd += f" ./{os.path.dirname(test_file)}/..."

    if test_name:
        if framework == "pytest":
            cmd += f" -k '{test_name}'"
        elif framework == "jest":
            cmd += f" -t '{test_name}'"
        elif framework == "gradle":
            cmd += f" --tests '*{test_name}*'"
        elif framework == "go":
            cmd += f" -run '{test_name}'"

    logger.info("run_tests: %s (framework=%s)", cmd, framework)

    # Run tests
    result = run_command(repo_path, cmd, timeout=timeout)

    if "error" in result and "exit_code" not in result:
        return result

    # Parse output
    stdout = result.get("stdout", "")
    parsed = _parse_test_output(stdout, framework)

    return {
        **result,
        "framework": framework,
        "tests_run": parsed["total"],
        "tests_passed": parsed["passed"],
        "tests_failed": parsed["failed"],
        "tests_skipped": parsed.get("skipped", 0),
        "failures": parsed["failure_details"],
    }


# =============================================================================
# Linting Tool (Phase 7.3)
# =============================================================================

LINT_COMMANDS = {
    "ruff": "ruff check",
    "flake8": "flake8",
    "pylint": "pylint",
    "eslint": "npx eslint",
    "prettier": "npx prettier --check",
    "golint": "golint",
    "gofmt": "gofmt -l",
    "rustfmt": "cargo fmt --check",
}


def _detect_linter(repo_path: str) -> str:
    """Auto-detect linter from project files."""
    # Python linters
    if os.path.exists(os.path.join(repo_path, "ruff.toml")) or \
       os.path.exists(os.path.join(repo_path, ".ruff.toml")):
        return "ruff"

    pyproject = os.path.join(repo_path, "pyproject.toml")
    if os.path.exists(pyproject):
        try:
            with open(pyproject, "r") as f:
                content = f.read()
                if "[tool.ruff]" in content:
                    return "ruff"
        except Exception:
            pass

    if os.path.exists(os.path.join(repo_path, ".flake8")):
        return "flake8"

    # JS/TS linters
    if os.path.exists(os.path.join(repo_path, ".eslintrc.js")) or \
       os.path.exists(os.path.join(repo_path, ".eslintrc.json")):
        return "eslint"

    # Go
    if os.path.exists(os.path.join(repo_path, "go.mod")):
        return "gofmt"

    # Rust
    if os.path.exists(os.path.join(repo_path, "Cargo.toml")):
        return "rustfmt"

    # Default
    return "ruff"


def _parse_lint_output(output: str, linter: str) -> list[dict]:
    """Parse linter output to extract issues."""
    import re

    issues = []

    if linter in ("ruff", "flake8", "pylint"):
        # Format: file.py:10:5: E501 line too long
        pattern = r"([^:]+):(\d+):(\d+): (\w+) (.+)"
        for match in re.finditer(pattern, output):
            issues.append({
                "file": match.group(1),
                "line": int(match.group(2)),
                "column": int(match.group(3)),
                "code": match.group(4),
                "message": match.group(5),
            })

    elif linter == "eslint":
        # Format: file.js:10:5: error/warning message (rule)
        pattern = r"([^:]+):(\d+):(\d+): (error|warning) (.+) \((.+)\)"
        for match in re.finditer(pattern, output):
            issues.append({
                "file": match.group(1),
                "line": int(match.group(2)),
                "column": int(match.group(3)),
                "severity": match.group(4),
                "message": match.group(5),
                "rule": match.group(6),
            })

    elif linter in ("gofmt", "golint"):
        # gofmt just lists files that need formatting
        for line in output.strip().split("\n"):
            if line.strip():
                issues.append({
                    "file": line.strip(),
                    "message": "needs formatting",
                })

    return issues


def lint_code(
    repo_path: str,
    file_path: str | None = None,
    fix: bool = False,
    linter: str = "auto",
) -> dict:
    """Run linter and return issues.

    Args:
        repo_path: Repository root path
        file_path: Specific file to lint (optional)
        fix: Auto-fix issues if supported
        linter: Linter to use (auto, ruff, flake8, eslint, etc.)

    Returns:
        {linter, issues, fixed, stdout, stderr}
    """
    # Auto-detect linter
    if linter == "auto":
        linter = _detect_linter(repo_path)

    if linter not in LINT_COMMANDS:
        return {"error": f"Unsupported linter: {linter}"}

    # Build command
    cmd = LINT_COMMANDS[linter]

    if fix:
        if linter == "ruff":
            cmd += " --fix"
        elif linter == "eslint":
            cmd += " --fix"
        elif linter == "prettier":
            cmd = cmd.replace("--check", "--write")
        elif linter == "gofmt":
            cmd = "gofmt -w"
        elif linter == "rustfmt":
            cmd = "cargo fmt"

    if file_path:
        cmd += f" {file_path}"
    else:
        # Lint common directories
        if linter in ("ruff", "flake8", "pylint"):
            cmd += " ."
        elif linter == "eslint":
            cmd += " src/"

    logger.info("lint_code: %s (linter=%s, fix=%s)", cmd, linter, fix)

    result = run_command(repo_path, cmd, timeout=120)

    if "error" in result and "exit_code" not in result:
        return result

    # Parse issues
    stdout = result.get("stdout", "")
    issues = _parse_lint_output(stdout, linter)

    return {
        "linter": linter,
        "issues": issues,
        "issue_count": len(issues),
        "fixed": fix,
        "stdout": stdout,
        "stderr": result.get("stderr", ""),
        "exit_code": result.get("exit_code", 0),
    }
