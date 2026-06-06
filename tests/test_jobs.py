"""R4 — async job manager + /jobs endpoints."""

from __future__ import annotations

import asyncio

from server.jobs import get_job_manager, reset_job_manager


async def _wait_done(jm, jid, tries=100):
    for _ in range(tries):
        j = jm.get(jid)
        if j and j["status"] in ("completed", "failed"):
            return j
        await asyncio.sleep(0.01)
    return jm.get(jid)


class TestJobManager:
    async def test_completes_and_returns_result(self):
        reset_job_manager()
        jm = get_job_manager()

        async def work():
            await asyncio.sleep(0)
            return {"indexed": 7}

        jid = jm.submit("index", work())
        job = await _wait_done(jm, jid)
        assert job["status"] == "completed"
        assert job["result"] == {"indexed": 7}

    async def test_failure_captured(self):
        reset_job_manager()
        jm = get_job_manager()

        async def boom():
            raise ValueError("kaboom")

        jid = jm.submit("bad", boom())
        job = await _wait_done(jm, jid)
        assert job["status"] == "failed"
        assert "kaboom" in job["error"]

    async def test_unknown_job_is_none(self):
        reset_job_manager()
        assert get_job_manager().get("nope") is None


class TestJobEndpoints:
    def _client(self, monkeypatch):
        monkeypatch.setenv("DEV_MODE", "true")
        monkeypatch.delenv("ENABLE_RAG", raising=False)
        monkeypatch.setattr("server.auth.API_KEY", "test-key")
        from fastapi.testclient import TestClient
        from server.app import create_app

        return TestClient(create_app())

    def test_get_missing_job_404(self, monkeypatch):
        with self._client(monkeypatch) as c:
            r = c.get("/jobs/doesnotexist", headers={"X-Api-Key": "test-key"})
            assert r.status_code == 404

    def test_jobs_requires_auth(self, monkeypatch):
        with self._client(monkeypatch) as c:
            r = c.get("/jobs")
            assert r.status_code == 403
