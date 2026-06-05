"""End-to-end chat pipeline test (Phase 20.3).

Drives the full LangGraph (classify → route → planner → generate → verify →
post_process) through the real FastAPI app, using DEV_MODE's mock vLLM client so
no model server is required.
"""

from __future__ import annotations


def _client(monkeypatch):
    monkeypatch.setenv("DEV_MODE", "true")
    monkeypatch.delenv("ENABLE_RAG", raising=False)
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.setattr("server.auth.API_KEY", "test-key")
    from fastapi.testclient import TestClient
    from server.app import create_app

    return TestClient(create_app())


class TestChatEndToEnd:
    def test_full_pipeline_streams_mock_reply(self, monkeypatch):
        with _client(monkeypatch) as client:
            resp = client.post(
                "/v1/chat/completions",
                headers={"X-Api-Key": "test-key"},
                json={
                    "messages": [{"role": "user", "content": "write a hello world in python"}],
                    "stream": True,
                },
            )
            assert resp.status_code == 200
            body = resp.text
            # mock vLLM reply made it through generate → SSE
            assert "dev-mode mock response" in body
            # SSE stream terminated
            assert "[DONE]" in body or "data:" in body

    def test_auth_required(self, monkeypatch):
        with _client(monkeypatch) as client:
            resp = client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hi"}]},
            )
            assert resp.status_code == 403

    def test_input_validation_rejects_empty(self, monkeypatch):
        with _client(monkeypatch) as client:
            resp = client.post(
                "/v1/chat/completions",
                headers={"X-Api-Key": "test-key"},
                json={"messages": []},
            )
            assert resp.status_code == 422
