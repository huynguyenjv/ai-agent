"""Model Comparison — Phase 16.6.

Runs the same set of cases across multiple models and compares quality, latency
and cost, then recommends the best quality/cost trade-off. Model invocation and
scoring are injected (async callables) so this is model-agnostic and testable.
"""

from __future__ import annotations

import logging
import time
from typing import Awaitable, Callable

logger = logging.getLogger("eval.model_comparison")

# invoke(model, query) -> response text
InvokeFn = Callable[[str, str], Awaitable[str]]
# score(query, response) -> float in [0, 10]
ScoreFn = Callable[[str, str], Awaitable[float]]


async def compare_models(
    models: list[str],
    cases: list[dict],
    invoke: InvokeFn,
    score: ScoreFn,
    cost_per_1k_tokens: dict[str, float] | None = None,
) -> dict:
    """Evaluate each model over `cases` ({query: ...}).

    Returns per-model {avg_score, avg_latency_ms, avg_tokens, est_cost} and a
    `recommendation` (best score-per-cost).
    """
    cost_per_1k_tokens = cost_per_1k_tokens or {}
    per_model: dict[str, dict] = {}

    for model in models:
        scores: list[float] = []
        latencies: list[float] = []
        token_counts: list[int] = []

        for case in cases:
            query = case.get("query", "")
            start = time.monotonic()
            try:
                response = await invoke(model, query)
            except Exception as e:
                logger.warning("model %s failed on a case: %s", model, e)
                response = ""
            latencies.append((time.monotonic() - start) * 1000)
            token_counts.append(len(response) // 4)  # rough estimate
            try:
                scores.append(await score(query, response))
            except Exception:
                scores.append(0.0)

        n = len(cases) or 1
        avg_tokens = sum(token_counts) / n
        price = cost_per_1k_tokens.get(model, 0.0)
        per_model[model] = {
            "avg_score": round(sum(scores) / n, 3),
            "avg_latency_ms": round(sum(latencies) / n, 1),
            "avg_tokens": round(avg_tokens, 1),
            "est_cost_per_request": round((avg_tokens / 1000) * price, 6),
        }

    recommendation = _recommend(per_model)
    return {"models": per_model, "recommendation": recommendation}


def _recommend(per_model: dict[str, dict]) -> str | None:
    """Pick the model with the best score-per-cost (falls back to best score)."""
    if not per_model:
        return None

    def value(m: dict) -> float:
        cost = m["est_cost_per_request"]
        return m["avg_score"] / cost if cost > 0 else m["avg_score"]

    return max(per_model.items(), key=lambda kv: value(kv[1]))[0]
