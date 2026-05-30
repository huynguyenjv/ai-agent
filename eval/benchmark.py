"""Offline evaluation benchmark suite.

Provides benchmark cases and scoring functions to evaluate
agent quality and detect regressions.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

logger = logging.getLogger("eval.benchmark")


@dataclass
class BenchmarkCase:
    """A single benchmark test case."""
    id: str
    input: str
    expected_intent: str
    expected_contains: list[str] = field(default_factory=list)
    expected_not_contains: list[str] = field(default_factory=list)
    expected_min_length: int = 50
    tags: list[str] = field(default_factory=list)
    weight: float = 1.0


@dataclass
class BenchmarkResult:
    """Result of a single benchmark case."""
    case_id: str
    passed: bool
    score: float
    intent_correct: bool
    contains_score: float
    not_contains_score: float
    length_ok: bool
    latency_ms: float
    output_preview: str
    error: str | None = None


# =============================================================================
# Benchmark Cases
# =============================================================================

BENCHMARK_CASES = [
    # Code Generation
    BenchmarkCase(
        id="code_gen_python_simple",
        input="Write a Python function to calculate factorial",
        expected_intent="code_gen",
        expected_contains=["def ", "factorial", "return"],
        expected_not_contains=["import os", "import sys"],
        tags=["python", "simple"],
    ),
    BenchmarkCase(
        id="code_gen_python_recursion",
        input="Write a recursive Python function to compute Fibonacci numbers",
        expected_intent="code_gen",
        expected_contains=["def ", "fib", "return", "if"],
        tags=["python", "recursion"],
    ),
    BenchmarkCase(
        id="code_gen_java_class",
        input="Create a Java class UserService with a method to find user by ID",
        expected_intent="code_gen",
        expected_contains=["class", "UserService", "findUserById", "public"],
        tags=["java", "oop"],
    ),
    BenchmarkCase(
        id="code_gen_typescript_interface",
        input="Create a TypeScript interface for a Product with id, name, and price",
        expected_intent="code_gen",
        expected_contains=["interface", "Product", "id", "name", "price"],
        tags=["typescript", "interface"],
    ),

    # Unit Tests
    BenchmarkCase(
        id="unit_test_python",
        input="Write pytest tests for a Calculator class with add and subtract methods",
        expected_intent="unit_test",
        expected_contains=["def test_", "assert", "Calculator"],
        tags=["python", "testing"],
    ),
    BenchmarkCase(
        id="unit_test_java",
        input="Write JUnit tests for UserService.createUser() method",
        expected_intent="unit_test",
        expected_contains=["@Test", "assert", "UserService"],
        tags=["java", "testing"],
    ),
    BenchmarkCase(
        id="unit_test_javascript",
        input="Write Jest tests for a login function",
        expected_intent="unit_test",
        expected_contains=["test(", "expect(", "login"],
        tags=["javascript", "testing"],
    ),

    # Code Review
    BenchmarkCase(
        id="code_review_security",
        input="Review this code for security issues: user_input = request.GET['query']; cursor.execute(f'SELECT * FROM users WHERE name = {user_input}')",
        expected_intent="code_review",
        expected_contains=["SQL injection", "parameterized", "security"],
        tags=["security", "review"],
    ),
    BenchmarkCase(
        id="code_review_quality",
        input="Review this function for code quality: def f(x,y,z): return x+y+z if x>0 else y-z",
        expected_intent="code_review",
        expected_contains=["naming", "readable"],
        tags=["quality", "review"],
    ),

    # Explanation
    BenchmarkCase(
        id="explain_code",
        input="Explain what this code does: def binary_search(arr, target): l, r = 0, len(arr)-1; while l <= r: m = (l+r)//2; if arr[m] == target: return m; elif arr[m] < target: l = m+1; else: r = m-1; return -1",
        expected_intent="explain",
        expected_contains=["binary", "search", "middle", "half"],
        tags=["explanation"],
    ),
    BenchmarkCase(
        id="explain_concept",
        input="Explain dependency injection in simple terms",
        expected_intent="explain",
        expected_contains=["dependency", "inject", "decoupl"],
        tags=["concept", "explanation"],
    ),

    # Refactoring
    BenchmarkCase(
        id="refactor_extract",
        input="Refactor this code to extract repeated logic into a function: x = a*2+1; y = b*2+1; z = c*2+1",
        expected_intent="refactor",
        expected_contains=["def ", "return"],
        tags=["refactor"],
    ),

    # Search
    BenchmarkCase(
        id="search_symbol",
        input="Find where UserRepository is defined",
        expected_intent="search",
        expected_contains=[],  # Search may not produce text
        tags=["search"],
    ),

    # Multi-language
    BenchmarkCase(
        id="code_gen_go",
        input="Write a Go function to reverse a string",
        expected_intent="code_gen",
        expected_contains=["func", "string", "return"],
        tags=["go"],
    ),
    BenchmarkCase(
        id="code_gen_rust",
        input="Write a Rust function to check if a number is prime",
        expected_intent="code_gen",
        expected_contains=["fn ", "bool", "return"],
        tags=["rust"],
    ),

    # Edge Cases
    BenchmarkCase(
        id="edge_empty_input",
        input="",
        expected_intent="code_gen",  # Default
        expected_contains=[],
        expected_min_length=0,
        tags=["edge"],
    ),
    BenchmarkCase(
        id="edge_gibberish",
        input="asdf jkl; qwerty",
        expected_intent="code_gen",
        expected_contains=[],
        expected_min_length=0,
        tags=["edge"],
    ),
]


# =============================================================================
# Scoring Functions
# =============================================================================

def score_output(output: str, case: BenchmarkCase) -> dict[str, Any]:
    """Score an output against a benchmark case.

    Args:
        output: Generated output text
        case: Benchmark case

    Returns:
        Scoring breakdown dict
    """
    output_lower = output.lower()

    # Contains score (how many expected items found)
    contains_hits = 0
    for expected in case.expected_contains:
        if expected.lower() in output_lower:
            contains_hits += 1

    contains_total = len(case.expected_contains) or 1
    contains_score = contains_hits / contains_total

    # Not contains score (none of the forbidden items)
    not_contains_hits = 0
    for forbidden in case.expected_not_contains:
        if forbidden.lower() in output_lower:
            not_contains_hits += 1

    not_contains_total = len(case.expected_not_contains) or 1
    not_contains_score = 1.0 - (not_contains_hits / not_contains_total)

    # Length check
    length_ok = len(output) >= case.expected_min_length

    # Combined score
    score = (
        contains_score * 0.5 +
        not_contains_score * 0.3 +
        (1.0 if length_ok else 0.5) * 0.2
    )

    return {
        "contains_score": contains_score,
        "contains_hits": contains_hits,
        "contains_total": len(case.expected_contains),
        "not_contains_score": not_contains_score,
        "not_contains_hits": not_contains_hits,
        "length_ok": length_ok,
        "output_length": len(output),
        "score": score,
    }


def evaluate_intent(actual_intent: str, expected_intent: str) -> bool:
    """Check if intent classification is correct.

    Args:
        actual_intent: Classified intent
        expected_intent: Expected intent

    Returns:
        True if match
    """
    # Normalize intents
    actual = actual_intent.lower().strip()
    expected = expected_intent.lower().strip()

    # Exact match
    if actual == expected:
        return True

    # Aliases
    aliases = {
        "code_gen": ["codegen", "generate", "code_generation"],
        "unit_test": ["unittest", "test", "testing"],
        "explain": ["explanation", "describe"],
        "code_review": ["review", "codereview"],
        "refactor": ["refactoring"],
        "search": ["find", "locate"],
    }

    for canonical, alias_list in aliases.items():
        if expected == canonical and actual in alias_list:
            return True
        if actual == canonical and expected in alias_list:
            return True

    return False


# =============================================================================
# Benchmark Runner
# =============================================================================

async def run_benchmark(
    invoke_fn: Callable,
    cases: list[BenchmarkCase] | None = None,
    tags_filter: list[str] | None = None,
) -> dict[str, Any]:
    """Run benchmark suite.

    Args:
        invoke_fn: Async function that takes input string and returns (output, intent)
        cases: Benchmark cases (defaults to BENCHMARK_CASES)
        tags_filter: Only run cases with these tags

    Returns:
        Benchmark results summary
    """
    if cases is None:
        cases = BENCHMARK_CASES

    # Filter by tags
    if tags_filter:
        cases = [c for c in cases if any(t in c.tags for t in tags_filter)]

    results: list[BenchmarkResult] = []

    for case in cases:
        start_time = time.time()

        try:
            output, intent = await invoke_fn(case.input)
            latency_ms = (time.time() - start_time) * 1000

            # Score output
            scoring = score_output(output, case)
            intent_correct = evaluate_intent(intent, case.expected_intent)

            # Adjust score with intent correctness
            final_score = scoring["score"] * (1.0 if intent_correct else 0.7)

            result = BenchmarkResult(
                case_id=case.id,
                passed=final_score >= 0.7,
                score=final_score,
                intent_correct=intent_correct,
                contains_score=scoring["contains_score"],
                not_contains_score=scoring["not_contains_score"],
                length_ok=scoring["length_ok"],
                latency_ms=latency_ms,
                output_preview=output[:200] if output else "",
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            result = BenchmarkResult(
                case_id=case.id,
                passed=False,
                score=0.0,
                intent_correct=False,
                contains_score=0.0,
                not_contains_score=0.0,
                length_ok=False,
                latency_ms=latency_ms,
                output_preview="",
                error=str(e),
            )

        results.append(result)
        logger.info("Benchmark %s: score=%.2f, passed=%s",
                    case.id, result.score, result.passed)

    # Summary
    passed_count = sum(1 for r in results if r.passed)
    avg_score = sum(r.score for r in results) / len(results) if results else 0
    avg_latency = sum(r.latency_ms for r in results) / len(results) if results else 0

    return {
        "timestamp": datetime.now().isoformat(),
        "total_cases": len(results),
        "passed": passed_count,
        "failed": len(results) - passed_count,
        "pass_rate": passed_count / len(results) if results else 0,
        "avg_score": avg_score,
        "avg_latency_ms": avg_latency,
        "results": [
            {
                "case_id": r.case_id,
                "passed": r.passed,
                "score": r.score,
                "intent_correct": r.intent_correct,
                "latency_ms": r.latency_ms,
                "error": r.error,
            }
            for r in results
        ],
    }


def compare_to_baseline(
    current: dict[str, Any],
    baseline: dict[str, Any],
    regression_threshold: float = 0.05,
) -> dict[str, Any]:
    """Compare benchmark results to baseline.

    Args:
        current: Current benchmark results
        baseline: Baseline benchmark results
        regression_threshold: Max allowed score drop

    Returns:
        Comparison summary
    """
    current_score = current.get("avg_score", 0)
    baseline_score = baseline.get("avg_score", 0)
    delta = current_score - baseline_score

    regressions = []

    # Compare individual cases
    baseline_by_id = {r["case_id"]: r for r in baseline.get("results", [])}

    for result in current.get("results", []):
        case_id = result["case_id"]
        if case_id in baseline_by_id:
            baseline_case = baseline_by_id[case_id]
            case_delta = result["score"] - baseline_case["score"]

            if case_delta < -regression_threshold:
                regressions.append({
                    "case_id": case_id,
                    "baseline_score": baseline_case["score"],
                    "current_score": result["score"],
                    "delta": case_delta,
                })

    return {
        "baseline_score": baseline_score,
        "current_score": current_score,
        "delta": delta,
        "improved": delta > 0,
        "regression_detected": len(regressions) > 0,
        "regressions": regressions,
        "new_cases": [
            r["case_id"] for r in current.get("results", [])
            if r["case_id"] not in baseline_by_id
        ],
    }


def save_baseline(results: dict[str, Any], filepath: str) -> None:
    """Save benchmark results as baseline.

    Args:
        results: Benchmark results
        filepath: Output file path
    """
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved baseline to %s", filepath)


def load_baseline(filepath: str) -> dict[str, Any] | None:
    """Load baseline from file.

    Args:
        filepath: Baseline file path

    Returns:
        Baseline dict or None
    """
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Failed to load baseline: %s", e)
        return None
