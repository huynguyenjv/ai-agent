"""Phase 10.3 / 10.5 / 10.6 — secret scanning, audit logging, input validation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from server.validation import (
    validate_chat_request,
    is_safe_relative_path,
    ValidationError,
    MAX_MESSAGES,
)
from server.utils import secret_scanner
from server.audit import (
    AuditLogger,
    SQLiteAuditStorage,
    AuditEvent,
    AuditEventType,
)


def _req(messages, active_file=None):
    return SimpleNamespace(
        messages=[SimpleNamespace(content=c) for c in messages],
        active_file=active_file,
    )


# --------------------------------------------------------------------------- #
# 10.6 Input validation
# --------------------------------------------------------------------------- #
class TestInputValidation:
    def test_normal_request_ok(self):
        validate_chat_request(_req(["hello", "world"]))  # no raise

    def test_empty_rejected(self):
        with pytest.raises(ValidationError):
            validate_chat_request(_req([]))

    def test_too_many_messages(self):
        with pytest.raises(ValidationError):
            validate_chat_request(_req(["x"] * (MAX_MESSAGES + 1)))

    def test_oversized_message(self):
        with pytest.raises(ValidationError):
            validate_chat_request(_req(["a" * 200_001]))

    def test_unsafe_active_file_traversal(self):
        with pytest.raises(ValidationError):
            validate_chat_request(_req(["hi"], active_file="../../etc/passwd"))

    def test_path_safety(self):
        assert is_safe_relative_path("src/main.py")
        assert not is_safe_relative_path("../secrets")
        assert not is_safe_relative_path("/etc/passwd")
        assert not is_safe_relative_path("C:/Windows/system32")
        assert is_safe_relative_path("")


# --------------------------------------------------------------------------- #
# 10.3 Secret scanning
# --------------------------------------------------------------------------- #
class TestSecretScanner:
    def test_detects_aws_key(self):
        findings = secret_scanner.scan("key = AKIAIOSFODNN7EXAMPLE end")
        assert any(f.kind == "aws_access_key_id" for f in findings)

    def test_detects_assigned_password(self):
        assert secret_scanner.has_secrets('DB_PASSWORD="sup3rs3cretvalue"')

    def test_detects_private_key_block(self):
        text = "-----BEGIN PRIVATE KEY-----\nMIIBVwIBADAN\n-----END PRIVATE KEY-----"
        assert secret_scanner.has_secrets(text)

    def test_redact_replaces_secret(self):
        text = "token AKIAIOSFODNN7EXAMPLE here"
        redacted, findings = secret_scanner.redact(text)
        assert "AKIAIOSFODNN7EXAMPLE" not in redacted
        assert "[REDACTED]" in redacted
        assert findings

    def test_clean_text_no_findings(self):
        assert secret_scanner.scan("just a normal sentence about code") == []

    def test_preview_never_leaks_full_secret(self):
        findings = secret_scanner.scan("key = AKIAIOSFODNN7EXAMPLE")
        for f in findings:
            assert "AKIAIOSFODNN7EXAMPLE" not in f.preview


# --------------------------------------------------------------------------- #
# 10.5 Audit logging
# --------------------------------------------------------------------------- #
class TestAuditLogging:
    def _logger(self):
        return AuditLogger(SQLiteAuditStorage(":memory:"))

    def test_record_and_recent(self):
        audit = self._logger()
        audit.security_violation("prompt_injection", actor="1.2.3.4", threat_level="critical")
        rows = audit.recent()
        assert len(rows) == 1
        assert rows[0]["event_type"] == AuditEventType.SECURITY_VIOLATION.value
        assert rows[0]["outcome"] == "blocked"
        assert rows[0]["detail"]["threat_level"] == "critical"

    def test_tool_and_auth_events(self):
        audit = self._logger()
        audit.tool_execution("vtrip_run_command", actor="key1", outcome="ok")
        audit.auth("verify_api_key", actor="9.9.9.9", outcome="error")
        rows = audit.recent()
        assert {r["event_type"] for r in rows} == {
            AuditEventType.TOOL_EXECUTION.value,
            AuditEventType.AUTH.value,
        }

    def test_clear_old_keeps_recent(self):
        audit = self._logger()
        audit.tool_execution("x", actor="a")
        assert audit.clear_old(days=90) == 0  # nothing older than 90d
        assert len(audit.recent()) == 1

    def test_logging_is_best_effort(self):
        class Boom(SQLiteAuditStorage):
            def record(self, event):
                raise RuntimeError("db down")

        audit = AuditLogger(Boom(":memory:"))
        # Must not raise into the caller
        audit.log(AuditEvent(AuditEventType.AUTH.value, "login", actor="x"))
