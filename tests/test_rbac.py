"""R11 — RBAC role → permission → tool filtering."""

from __future__ import annotations

from server.auth_rbac import (
    Permission,
    role_for_key,
    permissions_for,
    allowed_tool_names,
    has_permission,
)


class TestRbac:
    def test_default_role_is_developer(self, monkeypatch):
        monkeypatch.delenv("API_KEY_ROLE", raising=False)
        monkeypatch.delenv("API_KEY_ROLES", raising=False)
        monkeypatch.delenv("DEFAULT_ROLE", raising=False)
        assert role_for_key("anykey") == "developer"

    def test_single_role_env(self, monkeypatch):
        monkeypatch.setenv("API_KEY_ROLE", "viewer")
        assert role_for_key("k") == "viewer"

    def test_per_key_roles(self, monkeypatch):
        monkeypatch.setenv("API_KEY_ROLES", '{"k1": "admin", "k2": "viewer"}')
        assert role_for_key("k1") == "admin"
        assert role_for_key("k2") == "viewer"

    def test_viewer_is_read_only(self):
        tools = allowed_tool_names(permissions_for("viewer"))
        assert "vtrip_read_file" in tools
        assert "vtrip_grep" in tools
        assert "vtrip_apply_edits" not in tools       # WRITE
        assert "vtrip_run_command" not in tools        # EXECUTE

    def test_developer_has_write_and_execute(self):
        tools = allowed_tool_names(permissions_for("developer"))
        assert "vtrip_apply_edits_atomic" in tools
        assert "vtrip_run_tests" in tools

    def test_admin_gets_all(self):
        from server.auth_rbac import TOOL_PERMISSION
        tools = allowed_tool_names(permissions_for("admin"))
        assert tools == set(TOOL_PERMISSION.keys())

    def test_has_permission(self):
        assert has_permission("viewer", Permission.READ)
        assert not has_permission("viewer", Permission.WRITE)
