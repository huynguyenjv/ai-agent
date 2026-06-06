"""RBAC — Phase R11 (was Phase 10.4, optional).

Maps an API key → role → permissions, and filters which tools the agent is
allowed to use. Because tools execute client-side, the enforceable control point
server-side is *which tools we advertise* to the model.

Default role is `developer` (full coding) so existing single-key setups are
unchanged. Configure per-key roles via API_KEY_ROLES (JSON) or a single
API_KEY_ROLE.
"""

from __future__ import annotations

import json
import logging
import os
from enum import Enum

logger = logging.getLogger("server.auth_rbac")


class Permission(str, Enum):
    READ = "read"            # read files, search, grep, skeleton, git-read
    WRITE = "write"          # edits, commits, branches
    EXECUTE = "execute"      # run_command/tests/lint
    REVIEW = "review"        # code review tools
    ADMIN = "admin"


ROLE_PERMISSIONS: dict[str, set[Permission]] = {
    "viewer": {Permission.READ},
    "developer": {Permission.READ, Permission.WRITE, Permission.EXECUTE, Permission.REVIEW},
    "admin": {Permission.READ, Permission.WRITE, Permission.EXECUTE, Permission.REVIEW, Permission.ADMIN},
}

# Tool → required permission
TOOL_PERMISSION: dict[str, Permission] = {
    "vtrip_read_file": Permission.READ,
    "vtrip_search_symbol": Permission.READ,
    "vtrip_grep": Permission.READ,
    "vtrip_get_project_skeleton": Permission.READ,
    "vtrip_index_with_deps": Permission.READ,
    "vtrip_git_status": Permission.READ,
    "vtrip_git_diff": Permission.READ,
    "vtrip_git_log": Permission.READ,
    "vtrip_diff_preview": Permission.WRITE,
    "vtrip_apply_edits": Permission.WRITE,
    "vtrip_apply_edits_atomic": Permission.WRITE,
    "vtrip_rename_symbol": Permission.WRITE,
    "vtrip_extract_function": Permission.WRITE,
    "vtrip_inline_variable": Permission.WRITE,
    "vtrip_git_commit": Permission.WRITE,
    "vtrip_git_branch": Permission.WRITE,
    "vtrip_run_command": Permission.EXECUTE,
    "vtrip_run_tests": Permission.EXECUTE,
    "vtrip_lint_code": Permission.EXECUTE,
}

DEFAULT_ROLE = os.environ.get("DEFAULT_ROLE", "developer")


def role_for_key(api_key: str | None) -> str:
    """Resolve role for an API key. API_KEY_ROLES (JSON {key: role}) > API_KEY_ROLE > default."""
    if api_key:
        roles_json = os.environ.get("API_KEY_ROLES")
        if roles_json:
            try:
                mapping = json.loads(roles_json)
                if api_key in mapping:
                    return mapping[api_key]
            except json.JSONDecodeError:
                logger.warning("API_KEY_ROLES is not valid JSON")
    single = os.environ.get("API_KEY_ROLE")
    return single or DEFAULT_ROLE


def permissions_for(role: str) -> set[Permission]:
    return ROLE_PERMISSIONS.get(role, ROLE_PERMISSIONS["developer"])


def allowed_tool_names(perms: set[Permission]) -> set[str]:
    """Tool names permitted for a permission set (Permission.ADMIN ⇒ all)."""
    if Permission.ADMIN in perms:
        return set(TOOL_PERMISSION.keys())
    return {name for name, p in TOOL_PERMISSION.items() if p in perms}


def has_permission(role: str, perm: Permission) -> bool:
    return perm in permissions_for(role)
