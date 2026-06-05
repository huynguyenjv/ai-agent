"""Phase 16 — AI engineering maturity (prompt versioning, A/B, judge, quality,
trace→dataset, model comparison)."""

from __future__ import annotations

import json

from server.agent.prompt_store import reset_prompt_store
from server.experiment import ExperimentManager
from server.metrics.quality import QualityTracker
from eval.llm_judge import parse_judge_output, judge_response, judge_batch
from eval.trace_collector import TraceCollector
from eval.model_comparison import compare_models


# --------------------------------------------------------------------------- #
# 16.4 Prompt versioning
# --------------------------------------------------------------------------- #
class TestPromptStore:
    def test_loads_intent_and_variant(self):
        store = reset_prompt_store("config/prompts")
        default = store.get_intent("code_gen")
        concise = store.get_intent("code_gen", "concise")
        assert default and "Senior Software Engineer" in default
        assert concise and concise != default

    def test_unknown_intent_returns_none(self):
        store = reset_prompt_store("config/prompts")
        assert store.get_intent("does_not_exist") is None

    def test_version_and_variants(self):
        store = reset_prompt_store("config/prompts")
        assert store.version() == "1.0.0"
        assert "concise" in store.variants_for("code_gen")


# --------------------------------------------------------------------------- #
# 16.2 A/B testing
# --------------------------------------------------------------------------- #
class TestExperiment:
    def test_assignment_is_deterministic(self):
        m = ExperimentManager()
        m.register("exp", {"a": 50, "b": 50})
        v1 = m.get_variant("exp", "user-123")
        v2 = m.get_variant("exp", "user-123")
        assert v1 == v2
        assert v1 in ("a", "b")

    def test_weights_route_all_to_one(self):
        m = ExperimentManager()
        m.register("exp", {"a": 100, "b": 0})
        assert all(m.get_variant("exp", f"u{i}") == "a" for i in range(20))

    def test_unknown_experiment_default(self):
        m = ExperimentManager()
        assert m.get_variant("nope", "u") == "default"

    def test_outcome_tracking(self):
        m = ExperimentManager()
        m.register("exp", {"a": 100})
        m.record_outcome("exp", "a", success=True, score=8.0)
        m.record_outcome("exp", "a", success=False, score=4.0)
        res = m.results("exp")["a"]
        assert res["n"] == 2
        assert res["success_rate"] == 0.5
        assert res["avg_score"] == 6.0


# --------------------------------------------------------------------------- #
# 16.1 LLM-as-Judge
# --------------------------------------------------------------------------- #
class TestLlmJudge:
    def test_parse_valid(self):
        r = parse_judge_output('{"correctness":8,"completeness":7,"quality":9,"clarity":8,"overall":8}')
        assert r.parsed and r.overall == 8

    def test_parse_derives_overall(self):
        r = parse_judge_output('{"correctness":8,"completeness":8,"quality":8,"clarity":8}')
        assert r.parsed and r.overall == 8.0

    def test_parse_invalid(self):
        assert parse_judge_output("not json").parsed is False

    async def test_judge_response_with_fake_model(self):
        async def fake_complete(prompt):
            return '{"correctness":9,"completeness":9,"quality":9,"clarity":9,"overall":9}'

        r = await judge_response("task", "resp", fake_complete)
        assert r.overall == 9

    async def test_judge_batch_avg(self):
        async def fake_complete(prompt):
            return '{"overall": 7}'

        out = await judge_batch([{"task": "t", "response": "r"}] * 3, fake_complete)
        assert out["count"] == 3 and out["avg_overall"] == 7


# --------------------------------------------------------------------------- #
# 16.5 Online quality metrics
# --------------------------------------------------------------------------- #
class TestQualityTracker:
    def test_rates(self):
        q = QualityTracker()
        q.record(intent="code_gen", satisfied=True, completed=True, code_offered=True, code_accepted=True, response_length=100)
        q.record(intent="code_gen", satisfied=False, retried=True, code_offered=True, code_accepted=False, response_length=200)
        snap = q.snapshot()
        assert snap["total"] == 2
        assert snap["satisfaction_rate"] == 0.5
        assert snap["retry_rate"] == 0.5
        assert snap["code_acceptance_rate"] == 0.5
        assert snap["by_intent"]["code_gen"]["n"] == 2


# --------------------------------------------------------------------------- #
# 16.3 Trace → dataset
# --------------------------------------------------------------------------- #
class TestTraceCollector:
    def test_collect_and_export_redacts(self, tmp_path):
        tc = TraceCollector(output_dir=str(tmp_path))
        tc.add_positive(
            messages=[{"role": "user", "content": "key AKIAIOSFODNN7EXAMPLE"}],
            response="here is code",
            intent="code_gen",
        )
        tc.add_negative(messages=[{"role": "user", "content": "x"}], response="bad")
        assert tc.stats() == {"total": 2, "positive": 1, "negative": 1}

        path = tc.export_jsonl(version="test")
        lines = [json.loads(l) for l in open(path, encoding="utf-8")]
        assert len(lines) == 2
        # secret redacted, never written raw
        blob = json.dumps(lines)
        assert "AKIAIOSFODNN7EXAMPLE" not in blob
        assert "[REDACTED]" in blob

    def test_invalid_label(self, tmp_path):
        import pytest
        tc = TraceCollector(output_dir=str(tmp_path))
        with pytest.raises(ValueError):
            tc.add([], "r", "maybe")


# --------------------------------------------------------------------------- #
# 16.6 Model comparison
# --------------------------------------------------------------------------- #
class TestModelComparison:
    async def test_compare_and_recommend(self):
        async def invoke(model, query):
            return "x" * (40 if model == "big" else 20)

        async def score(query, response):
            return 9.0 if len(response) > 30 else 5.0

        out = await compare_models(
            models=["big", "small"],
            cases=[{"query": "q1"}, {"query": "q2"}],
            invoke=invoke,
            score=score,
            cost_per_1k_tokens={"big": 1.0, "small": 0.1},
        )
        assert out["models"]["big"]["avg_score"] == 9.0
        assert out["models"]["small"]["avg_score"] == 5.0
        # recommendation exists (best score-per-cost)
        assert out["recommendation"] in ("big", "small")
