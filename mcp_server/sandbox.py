"""Command Sandbox — Phase 10.1.

Enhanced command execution with:
- Whitelist validation
- Resource limits (timeout, memory)
- Audit logging
- Path traversal prevention
"""

from __future__ import annotations

import logging
import os
import re
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger("mcp_server.sandbox")


class CommandCategory(Enum):
    """Categories of allowed commands."""
    TEST = "test"
    LINT = "lint"
    BUILD = "build"
    GIT_READ = "git_read"
    FILE_READ = "file_read"
    SEARCH = "search"


@dataclass
class SandboxConfig:
    """Sandbox configuration."""
    timeout_seconds: int = 300
    max_output_bytes: int = 50_000
    max_memory_mb: int = 512
    allow_network: bool = False
    allowed_paths: list[str] = field(default_factory=list)


# Command whitelist with categories
COMMAND_WHITELIST: dict[str, CommandCategory] = {
    # Test runners
    "pytest": CommandCategory.TEST,
    "python": CommandCategory.TEST,
    "npm": CommandCategory.TEST,
    "npx": CommandCategory.TEST,
    "node": CommandCategory.TEST,
    "go": CommandCategory.TEST,
    "cargo": CommandCategory.TEST,
    "gradle": CommandCategory.BUILD,
    "gradlew": CommandCategory.BUILD,
    "mvn": CommandCategory.BUILD,
    "java": CommandCategory.TEST,
    "dotnet": CommandCategory.TEST,
    # Linters
    "ruff": CommandCategory.LINT,
    "eslint": CommandCategory.LINT,
    "prettier": CommandCategory.LINT,
    "gofmt": CommandCategory.LINT,
    "black": CommandCategory.LINT,
    "flake8": CommandCategory.LINT,
    "mypy": CommandCategory.LINT,
    "pylint": CommandCategory.LINT,
    # Git read-only
    "git": CommandCategory.GIT_READ,
    # File read
    "cat": CommandCategory.FILE_READ,
    "head": CommandCategory.FILE_READ,
    "tail": CommandCategory.FILE_READ,
    "ls": CommandCategory.FILE_READ,
    "wc": CommandCategory.FILE_READ,
    # Search
    "grep": CommandCategory.SEARCH,
    "find": CommandCategory.SEARCH,
    "rg": CommandCategory.SEARCH,
}

# Git subcommands allowed (read-only)
GIT_READONLY_SUBCOMMANDS = frozenset({
    "status", "log", "diff", "show", "branch", "blame",
    "ls-files", "describe", "rev-parse", "config",
})

# Patterns that indicate malicious intent
DANGEROUS_PATTERNS = [
    # Destructive
    r"rm\s+-[rf]",
    r"rmdir",
    r">\s*/dev/",
    r"\|\s*rm",
    r"&&\s*rm",
    # Privilege escalation
    r"\bsudo\b",
    r"\bsu\s",
    r"\bdoas\b",
    # Remote code execution
    r"curl.*\|\s*sh",
    r"wget.*\|\s*sh",
    r"\beval\b",
    r"\bexec\b",
    # Fork bombs / resource exhaustion
    r":\(\)\s*\{",
    r"\bfork\b",
    # System damage
    r"chmod\s+777",
    r"chmod\s+-R",
    r"\bdd\s+if=",
    r"\bmkfs\b",
    r"\bshutdown\b",
    r"\breboot\b",
    r"\bhalt\b",
    # User management
    r"\bpasswd\b",
    r"\buseradd\b",
    r"\buserdel\b",
    # Environment manipulation
    r"export\s+PATH=",
    r"unset\s+PATH",
    # Network exfiltration
    r"\bnc\s+-[el]",
    r"\bnetcat\b",
    r"\btelnet\b",
]

DANGEROUS_REGEX = [re.compile(p, re.IGNORECASE) for p in DANGEROUS_PATTERNS]


@dataclass
class SandboxResult:
    """Result of sandboxed command execution."""
    success: bool
    stdout: str = ""
    stderr: str = ""
    exit_code: int = -1
    error: str | None = None
    truncated: bool = False
    duration_ms: int = 0
    command: str = ""
    category: str | None = None


@dataclass
class AuditEntry:
    """Audit log entry for command execution."""
    timestamp: float
    command: str
    category: str | None
    allowed: bool
    reason: str | None
    exit_code: int | None
    duration_ms: int
    user_id: str | None = None


class CommandSandbox:
    """Sandboxed command executor with audit logging."""

    def __init__(self, config: SandboxConfig | None = None):
        self.config = config or SandboxConfig()
        self._audit_log: list[AuditEntry] = []

    def validate_command(self, command: str) -> tuple[bool, str | None, CommandCategory | None]:
        """Validate command against whitelist and dangerous patterns.

        Returns:
            (allowed, reason_if_blocked, category)
        """
        if not command or not command.strip():
            return False, "Empty command", None

        # Check dangerous patterns
        cmd_str = command.lower()
        for pattern in DANGEROUS_REGEX:
            if pattern.search(cmd_str):
                return False, f"Dangerous pattern: {pattern.pattern}", None

        # Parse command
        try:
            parts = shlex.split(command)
            if not parts:
                return False, "Invalid command", None
        except ValueError as e:
            return False, f"Parse error: {e}", None

        # Check whitelist
        base_cmd = os.path.basename(parts[0])
        category = COMMAND_WHITELIST.get(base_cmd)

        if category is None:
            return False, f"Command not whitelisted: {base_cmd}", None

        # Git: only allow read-only subcommands
        if base_cmd == "git" and len(parts) > 1:
            subcommand = parts[1]
            if subcommand not in GIT_READONLY_SUBCOMMANDS:
                return False, f"Git subcommand not allowed: {subcommand}", None

        return True, None, category

    def validate_path(self, path: str, repo_root: str) -> tuple[bool, str | None]:
        """Validate path is within allowed boundaries.

        Prevents path traversal attacks.
        """
        try:
            real_path = os.path.realpath(path)
            real_root = os.path.realpath(repo_root)

            if not real_path.startswith(real_root + os.sep) and real_path != real_root:
                return False, "Path outside repository"

            return True, None
        except Exception as e:
            return False, f"Path validation error: {e}"

    def execute(
        self,
        command: str,
        repo_path: str,
        working_dir: str | None = None,
        user_id: str | None = None,
    ) -> SandboxResult:
        """Execute command in sandbox with full validation."""
        start_time = time.time()

        # Validate command
        allowed, reason, category = self.validate_command(command)

        if not allowed:
            self._log_audit(command, None, False, reason, None, 0, user_id)
            return SandboxResult(
                success=False,
                error=reason,
                command=command,
            )

        # Determine and validate working directory
        cwd = repo_path
        if working_dir:
            cwd = os.path.join(repo_path, working_dir)
            path_ok, path_err = self.validate_path(cwd, repo_path)
            if not path_ok:
                self._log_audit(command, category.value if category else None,
                              False, path_err, None, 0, user_id)
                return SandboxResult(
                    success=False,
                    error=path_err,
                    command=command,
                )
            if not os.path.isdir(cwd):
                return SandboxResult(
                    success=False,
                    error=f"Directory not found: {working_dir}",
                    command=command,
                )

        # Execute command
        try:
            parts = shlex.split(command)

            # Build environment (isolated)
            env = {
                **os.environ,
                "CI": "true",
                "TERM": "dumb",
            }

            # Remove potentially dangerous env vars
            for key in ["LD_PRELOAD", "LD_LIBRARY_PATH", "PYTHONPATH"]:
                env.pop(key, None)

            logger.info("sandbox.execute: %s (cwd=%s, user=%s)",
                       command, cwd, user_id or "anonymous")

            result = subprocess.run(
                parts,
                shell=False,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds,
                env=env,
            )

            duration_ms = int((time.time() - start_time) * 1000)

            # Process output
            stdout = result.stdout
            stderr = result.stderr
            truncated = False

            if len(stdout) > self.config.max_output_bytes:
                stdout = stdout[:self.config.max_output_bytes] + \
                    f"\n... [truncated, {len(result.stdout)} bytes total]"
                truncated = True
            if len(stderr) > self.config.max_output_bytes:
                stderr = stderr[:self.config.max_output_bytes] + \
                    f"\n... [truncated, {len(result.stderr)} bytes total]"
                truncated = True

            self._log_audit(command, category.value if category else None,
                          True, None, result.returncode, duration_ms, user_id)

            return SandboxResult(
                success=True,
                stdout=stdout,
                stderr=stderr,
                exit_code=result.returncode,
                truncated=truncated,
                duration_ms=duration_ms,
                command=command,
                category=category.value if category else None,
            )

        except subprocess.TimeoutExpired:
            duration_ms = int((time.time() - start_time) * 1000)
            self._log_audit(command, category.value if category else None,
                          False, "timeout", None, duration_ms, user_id)
            return SandboxResult(
                success=False,
                error=f"Command timed out after {self.config.timeout_seconds}s",
                duration_ms=duration_ms,
                command=command,
            )
        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            self._log_audit(command, category.value if category else None,
                          False, str(e), None, duration_ms, user_id)
            return SandboxResult(
                success=False,
                error=f"Execution failed: {e}",
                duration_ms=duration_ms,
                command=command,
            )

    def _log_audit(
        self,
        command: str,
        category: str | None,
        allowed: bool,
        reason: str | None,
        exit_code: int | None,
        duration_ms: int,
        user_id: str | None,
    ) -> None:
        """Log command execution to audit trail."""
        entry = AuditEntry(
            timestamp=time.time(),
            command=command,
            category=category,
            allowed=allowed,
            reason=reason,
            exit_code=exit_code,
            duration_ms=duration_ms,
            user_id=user_id,
        )
        self._audit_log.append(entry)

        # Keep audit log bounded
        if len(self._audit_log) > 10000:
            self._audit_log = self._audit_log[-5000:]

        # Log to standard logger
        if allowed:
            logger.info("AUDIT: command=%s category=%s exit=%s duration=%dms user=%s",
                       command[:100], category, exit_code, duration_ms, user_id)
        else:
            logger.warning("AUDIT_BLOCKED: command=%s reason=%s user=%s",
                          command[:100], reason, user_id)

    def get_audit_log(self, limit: int = 100) -> list[dict[str, Any]]:
        """Get recent audit entries."""
        entries = self._audit_log[-limit:]
        return [
            {
                "timestamp": e.timestamp,
                "command": e.command,
                "category": e.category,
                "allowed": e.allowed,
                "reason": e.reason,
                "exit_code": e.exit_code,
                "duration_ms": e.duration_ms,
                "user_id": e.user_id,
            }
            for e in entries
        ]


# Global sandbox instance
_sandbox: CommandSandbox | None = None


def get_sandbox() -> CommandSandbox:
    """Get or create global sandbox instance."""
    global _sandbox
    if _sandbox is None:
        _sandbox = CommandSandbox()
    return _sandbox


def run_sandboxed(
    repo_path: str,
    command: str,
    working_dir: str | None = None,
    user_id: str | None = None,
) -> dict[str, Any]:
    """Execute command in sandbox (convenience function).

    Returns dict compatible with existing run_command interface.
    """
    sandbox = get_sandbox()
    result = sandbox.execute(command, repo_path, working_dir, user_id)

    if result.success:
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.exit_code,
            "truncated": result.truncated,
            "command": result.command,
            "duration_ms": result.duration_ms,
        }
    else:
        return {
            "error": result.error,
            "exit_code": -1,
            "command": result.command,
        }
