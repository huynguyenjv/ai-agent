"""A/B Testing — Phase 16.2.

Deterministic variant assignment (stable per user/session) + per-variant
outcome tracking. Used to A/B test prompt variants (see prompt_store) or other
behaviours without redeploying.
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading

logger = logging.getLogger("server.experiment")


class Experiment:
    """A single experiment: weighted variants summing to ~100."""

    def __init__(self, name: str, variants: dict[str, int], default: str = "default"):
        if not variants:
            variants = {default: 100}
        self.name = name
        self.variants = variants
        self.default = default

    def assign(self, unit_id: str) -> str:
        """Deterministically assign a variant for a stable unit id (user/session)."""
        if not unit_id:
            return self.default
        digest = hashlib.sha256(f"{self.name}:{unit_id}".encode()).hexdigest()
        bucket = int(digest[:8], 16) % 100
        cumulative = 0
        for variant, weight in self.variants.items():
            cumulative += weight
            if bucket < cumulative:
                return variant
        return self.default


class ExperimentManager:
    """Holds experiments and tracks per-variant outcome counters."""

    def __init__(self):
        self._experiments: dict[str, Experiment] = {}
        self._stats: dict[tuple[str, str], dict[str, float]] = {}
        self._lock = threading.Lock()

    def register(self, name: str, variants: dict[str, int], default: str = "default") -> None:
        with self._lock:
            self._experiments[name] = Experiment(name, variants, default)

    def get_variant(self, name: str, unit_id: str) -> str:
        with self._lock:
            exp = self._experiments.get(name)
        return exp.assign(unit_id) if exp else "default"

    def record_outcome(self, name: str, variant: str, *, success: bool, score: float | None = None) -> None:
        key = (name, variant)
        with self._lock:
            s = self._stats.setdefault(key, {"n": 0, "success": 0, "score_sum": 0.0, "scored": 0})
            s["n"] += 1
            if success:
                s["success"] += 1
            if score is not None:
                s["score_sum"] += score
                s["scored"] += 1

    def results(self, name: str) -> dict[str, dict]:
        with self._lock:
            out: dict[str, dict] = {}
            for (exp_name, variant), s in self._stats.items():
                if exp_name != name:
                    continue
                n = s["n"] or 1
                out[variant] = {
                    "n": s["n"],
                    "success_rate": round(s["success"] / n, 4),
                    "avg_score": round(s["score_sum"] / s["scored"], 4) if s["scored"] else None,
                }
            return out


_manager: ExperimentManager | None = None
_lock = threading.Lock()


def get_experiment_manager() -> ExperimentManager:
    global _manager
    if _manager is None:
        with _lock:
            if _manager is None:
                _manager = ExperimentManager()
                # Default experiment: prompt variant for code_gen, off unless enabled.
                if os.environ.get("ENABLE_AB_PROMPT", "false").lower() in ("1", "true", "yes"):
                    _manager.register("code_gen_prompt", {"default": 50, "concise": 50})
                    logger.info("A/B experiment 'code_gen_prompt' enabled (default/concise)")
    return _manager


def reset_experiment_manager() -> None:
    global _manager
    with _lock:
        _manager = None
