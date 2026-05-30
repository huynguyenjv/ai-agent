"""Tests for code intelligence tools: test execution, linting, refactoring."""

import pytest

from mcp_server.tools import (
    _detect_test_framework,
    _parse_test_output,
    _detect_linter,
    _parse_lint_output,
)
from mcp_server.tools_refactor import (
    _detect_variables,
    _build_function,
    _build_function_call,
)


class TestTestFrameworkDetection:
    """Tests for test framework detection."""

    def test_default_pytest(self, tmp_path):
        # Empty directory defaults to pytest
        assert _detect_test_framework(str(tmp_path)) == "pytest"

    def test_detect_pytest_ini(self, tmp_path):
        (tmp_path / "pytest.ini").write_text("[pytest]")
        assert _detect_test_framework(str(tmp_path)) == "pytest"

    def test_detect_go_mod(self, tmp_path):
        (tmp_path / "go.mod").write_text("module example")
        assert _detect_test_framework(str(tmp_path)) == "go"

    def test_detect_cargo(self, tmp_path):
        (tmp_path / "Cargo.toml").write_text("[package]")
        assert _detect_test_framework(str(tmp_path)) == "cargo"


class TestParseTestOutput:
    """Tests for test output parsing."""

    def test_parse_pytest_success(self):
        output = "===== 5 passed in 1.23s ====="
        result = _parse_test_output(output, "pytest")
        assert result["passed"] == 5
        assert result["failed"] == 0
        assert result["total"] == 5

    def test_parse_pytest_mixed(self):
        output = "===== 3 passed, 2 failed, 1 skipped in 2.5s ====="
        result = _parse_test_output(output, "pytest")
        assert result["passed"] == 3
        assert result["failed"] == 2
        assert result["skipped"] == 1
        assert result["total"] == 6

    def test_parse_pytest_failures(self):
        output = "FAILED tests/test_foo.py::test_bar\nFAILED tests/test_baz.py::test_qux"
        result = _parse_test_output(output, "pytest")
        assert len(result["failure_details"]) == 2

    def test_parse_go_tests(self):
        output = "--- PASS: TestFoo\n--- PASS: TestBar\n--- FAIL: TestBaz"
        result = _parse_test_output(output, "go")
        assert result["passed"] == 2
        assert result["failed"] == 1

    def test_parse_cargo_tests(self):
        output = "test result: ok. 10 passed; 2 failed; 0 ignored"
        result = _parse_test_output(output, "cargo")
        assert result["passed"] == 10
        assert result["failed"] == 2

    def test_parse_junit_tests(self):
        output = "Tests run: 10, Failures: 2, Errors: 1, Skipped: 1"
        result = _parse_test_output(output, "junit")
        assert result["total"] == 10
        assert result["passed"] == 6  # 10 - 2 - 1 - 1
        assert result["failed"] == 3  # 2 failures + 1 error
        assert result["skipped"] == 1

    def test_parse_mocha_tests(self):
        output = "  5 passing (2s)\n  2 failing\n  1 pending"
        result = _parse_test_output(output, "mocha")
        assert result["passed"] == 5
        assert result["failed"] == 2
        assert result["skipped"] == 1


class TestLinterDetection:
    """Tests for linter detection."""

    def test_default_ruff(self, tmp_path):
        assert _detect_linter(str(tmp_path)) == "ruff"

    def test_detect_ruff_toml(self, tmp_path):
        (tmp_path / "ruff.toml").write_text("")
        assert _detect_linter(str(tmp_path)) == "ruff"

    def test_detect_flake8(self, tmp_path):
        (tmp_path / ".flake8").write_text("")
        assert _detect_linter(str(tmp_path)) == "flake8"

    def test_detect_eslint(self, tmp_path):
        (tmp_path / ".eslintrc.json").write_text("{}")
        assert _detect_linter(str(tmp_path)) == "eslint"

    def test_detect_gofmt(self, tmp_path):
        (tmp_path / "go.mod").write_text("module example")
        assert _detect_linter(str(tmp_path)) == "gofmt"


class TestParseLintOutput:
    """Tests for lint output parsing."""

    def test_parse_ruff_output(self):
        output = "src/main.py:10:5: E501 line too long\nsrc/utils.py:20:1: F401 unused import"
        issues = _parse_lint_output(output, "ruff")
        assert len(issues) == 2
        assert issues[0]["file"] == "src/main.py"
        assert issues[0]["line"] == 10
        assert issues[0]["code"] == "E501"

    def test_parse_gofmt_output(self):
        output = "main.go\nutils.go"
        issues = _parse_lint_output(output, "gofmt")
        assert len(issues) == 2
        assert issues[0]["file"] == "main.go"


class TestRefactoringHelpers:
    """Tests for refactoring helper functions."""

    def test_detect_variables(self):
        code = "x = foo + bar\nresult = calculate(x)"
        vars = _detect_variables(code, ".py")
        assert "x" in vars["all"]
        assert "foo" in vars["all"]
        assert "bar" in vars["all"]
        assert "result" in vars["all"]
        # Keywords should be filtered
        assert "if" not in vars["all"]

    def test_build_function_python(self):
        code = "x = 1\nreturn x"
        func = _build_function("my_func", code, {}, ".py")
        assert "def my_func():" in func
        assert "x = 1" in func

    def test_build_function_javascript(self):
        code = "let x = 1;\nreturn x;"
        func = _build_function("myFunc", code, {}, ".js")
        assert "function myFunc()" in func

    def test_build_function_call_python(self):
        call = _build_function_call("my_func", {}, ".py")
        assert call == "my_func()"

    def test_build_function_call_js(self):
        call = _build_function_call("myFunc", {}, ".js")
        assert call == "myFunc();"
