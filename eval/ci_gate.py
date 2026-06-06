"""Eval gate for CI (R2).

Compares a benchmark results file against a baseline and exits non-zero if a
regression is detected, so CI can block merges that degrade quality.

Usage:
    python -m eval.ci_gate --results results.json --baseline baseline.json
"""

from __future__ import annotations

import argparse
import json
import sys


def evaluate(results: dict, baseline: dict | None) -> tuple[bool, dict]:
    """Return (passed, detail). passed=False when a regression is detected."""
    try:
        from eval.benchmark import compare_to_baseline
    except Exception:
        compare_to_baseline = None

    if baseline is None:
        return True, {"status": "no_baseline", "note": "baseline missing — gate skipped"}

    if compare_to_baseline is not None:
        cmp = compare_to_baseline(results, baseline)
        passed = not cmp.get("regression_detected", False)
        return passed, cmp

    # Fallback: compare aggregate average score if present
    r = float(results.get("avg_score", results.get("average", 0)) or 0)
    b = float(baseline.get("avg_score", baseline.get("average", 0)) or 0)
    return r >= b - 0.02, {"results_avg": r, "baseline_avg": b}


def _load(path: str | None) -> dict | None:
    if not path:
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Eval gate for CI")
    ap.add_argument("--results", required=True)
    ap.add_argument("--baseline")
    args = ap.parse_args(argv)

    results = _load(args.results) or {}
    baseline = _load(args.baseline)

    passed, detail = evaluate(results, baseline)
    print(json.dumps({"passed": passed, "detail": detail}, indent=2, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
