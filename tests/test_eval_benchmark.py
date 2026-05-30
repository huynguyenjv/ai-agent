"""Tests for evaluation benchmark suite."""

import pytest

from eval.benchmark import (
    BenchmarkCase,
    BenchmarkResult,
    BENCHMARK_CASES,
    score_output,
    evaluate_intent,
    compare_to_baseline,
)


class TestBenchmarkCases:
    """Tests for benchmark case definitions."""

    def test_cases_not_empty(self):
        assert len(BENCHMARK_CASES) > 0

    def test_cases_have_required_fields(self):
        for case in BENCHMARK_CASES:
            assert case.id
            assert case.expected_intent
            assert isinstance(case.expected_contains, list)
            assert isinstance(case.tags, list)


class TestScoreOutput:
    """Tests for output scoring."""

    def test_perfect_match(self):
        case = BenchmarkCase(
            id="test",
            input="test",
            expected_intent="code_gen",
            expected_contains=["def ", "return"],
            expected_min_length=10,
        )
        output = "def foo():\n    return 42"

        result = score_output(output, case)

        assert result["contains_score"] == 1.0
        assert result["length_ok"] is True
        assert result["score"] > 0.8

    def test_partial_match(self):
        case = BenchmarkCase(
            id="test",
            input="test",
            expected_intent="code_gen",
            expected_contains=["def ", "return", "class"],
        )
        output = "def foo():\n    return 42"

        result = score_output(output, case)

        # 2 out of 3 contains
        assert result["contains_hits"] == 2
        assert result["contains_score"] == pytest.approx(2/3)

    def test_not_contains_penalty(self):
        case = BenchmarkCase(
            id="test",
            input="test",
            expected_intent="code_gen",
            expected_contains=[],
            expected_not_contains=["import os", "import sys"],
        )
        output = "import os\nprint('hello')"

        result = score_output(output, case)

        # 1 forbidden found
        assert result["not_contains_hits"] == 1
        assert result["not_contains_score"] == 0.5


class TestEvaluateIntent:
    """Tests for intent evaluation."""

    def test_exact_match(self):
        assert evaluate_intent("code_gen", "code_gen") is True

    def test_case_insensitive(self):
        assert evaluate_intent("CODE_GEN", "code_gen") is True

    def test_alias_match(self):
        assert evaluate_intent("generate", "code_gen") is True
        assert evaluate_intent("unittest", "unit_test") is True

    def test_mismatch(self):
        assert evaluate_intent("code_gen", "explain") is False


class TestCompareBaseline:
    """Tests for baseline comparison."""

    def test_improvement(self):
        baseline = {"avg_score": 0.7, "results": []}
        current = {"avg_score": 0.8, "results": []}

        comparison = compare_to_baseline(current, baseline)

        assert comparison["improved"] is True
        assert comparison["delta"] == pytest.approx(0.1)

    def test_regression(self):
        baseline = {
            "avg_score": 0.8,
            "results": [{"case_id": "test1", "score": 0.9}],
        }
        current = {
            "avg_score": 0.7,
            "results": [{"case_id": "test1", "score": 0.6}],
        }

        comparison = compare_to_baseline(current, baseline)

        assert comparison["improved"] is False
        assert comparison["regression_detected"] is True
        assert len(comparison["regressions"]) == 1

    def test_new_cases(self):
        baseline = {"avg_score": 0.8, "results": []}
        current = {
            "avg_score": 0.8,
            "results": [{"case_id": "new_case", "score": 0.9}],
        }

        comparison = compare_to_baseline(current, baseline)

        assert "new_case" in comparison["new_cases"]
