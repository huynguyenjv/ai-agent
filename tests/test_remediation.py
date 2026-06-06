"""Remediation tests — R1 sandbox hardening, R2 eval gate, R13 prompt migration."""

from __future__ import annotations

import json

from mcp_server.sandbox import CommandSandbox


# --------------------------------------------------------------------------- #
# R1 — sandbox hardening (close interpreter RCE vectors)
# --------------------------------------------------------------------------- #
class TestSandboxHardening:
    def setup_method(self):
        self.sb = CommandSandbox()

    def test_blocks_python_inline_code(self):
        ok, reason, _ = self.sb.validate_command('python -c "import os"')
        assert not ok and "inline code" in reason.lower()

    def test_blocks_node_eval(self):
        ok, reason, _ = self.sb.validate_command('node -e "process.exit(1)"')
        assert not ok

    def test_blocks_pip_install(self):
        ok, reason, _ = self.sb.validate_command("pip install requests")
        assert not ok and "argument pattern" in reason.lower()

    def test_blocks_npx(self):
        ok, reason, _ = self.sb.validate_command("npx some-package")
        assert not ok

    def test_allows_pytest(self):
        ok, _, _ = self.sb.validate_command("pytest tests/")
        assert ok

    def test_allows_python_m_pytest(self):
        ok, _, _ = self.sb.validate_command("python -m pytest")
        assert ok

    def test_exec_disabled_blocks_all(self, monkeypatch):
        monkeypatch.setenv("EXEC_ENABLED", "false")
        ok, reason, _ = self.sb.validate_command("pytest")
        assert not ok and "disabled" in reason.lower()


# --------------------------------------------------------------------------- #
# R2 — eval gate
# --------------------------------------------------------------------------- #
class TestEvalGate:
    def test_no_baseline_passes(self):
        from eval.ci_gate import evaluate

        passed, detail = evaluate({"avg_score": 0.8}, None)
        assert passed and detail["status"] == "no_baseline"

    def test_main_no_baseline_returns_zero(self, tmp_path):
        from eval.ci_gate import main

        r = tmp_path / "results.json"
        r.write_text(json.dumps({"avg_score": 0.8}), encoding="utf-8")
        assert main(["--results", str(r)]) == 0


# --------------------------------------------------------------------------- #
# R13 — all intents migrated to YAML
# --------------------------------------------------------------------------- #
class TestPromptMigration:
    def test_all_intents_resolve_from_yaml(self):
        from server.agent.prompt_store import reset_prompt_store

        store = reset_prompt_store("config/prompts")
        for intent in (
            "code_gen", "unit_test", "code_review", "structural_analysis",
            "search", "debug", "refine", "explain",
        ):
            assert store.get_intent(intent), f"missing intent in YAML: {intent}"
