"""R9 — /agents opt-in thorough mode (parse + routing + e2e)."""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from server.routers.chat import _parse_agents_directive
from server.agent.graph import _route_after_verify


class TestParseDirective:
    def test_strips_and_flags(self):
        msgs, flag = _parse_agents_directive([HumanMessage(content="/agents refactor X")])
        assert flag is True
        assert msgs[0].content == "refactor X"

    def test_case_insensitive(self):
        msgs, flag = _parse_agents_directive([HumanMessage(content="/Agents do it")])
        assert flag is True

    def test_no_directive(self):
        msgs, flag = _parse_agents_directive([HumanMessage(content="normal request")])
        assert flag is False
        assert msgs[0].content == "normal request"


class TestRouting:
    def test_multi_agent_always_reviews(self):
        state = {"verification_passed": True, "complexity": "simple", "multi_agent": True}
        assert _route_after_verify(state) == "critic"

    def test_simple_single_agent_skips_review(self):
        state = {"verification_passed": True, "complexity": "simple"}
        assert _route_after_verify(state) == "post_process"

    def test_failed_verify_retries(self):
        assert _route_after_verify({"verification_passed": False}) == "generate"


class TestMultiAgentE2E:
    def test_agents_prefix_runs(self, monkeypatch):
        monkeypatch.setenv("DEV_MODE", "true")
        monkeypatch.delenv("ENABLE_RAG", raising=False)
        monkeypatch.setattr("server.auth.API_KEY", "test-key")
        from fastapi.testclient import TestClient
        from server.app import create_app

        with TestClient(create_app()) as c:
            r = c.post(
                "/v1/chat/completions",
                headers={"X-Api-Key": "test-key"},
                json={"messages": [{"role": "user", "content": "/agents write hello world"}]},
            )
            assert r.status_code == 200
            assert "dev-mode mock response" in r.text
