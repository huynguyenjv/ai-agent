"""Integration smoke test for /v1/chat/completions native tool-call streaming."""
import os

os.environ.setdefault("API_KEY", "test-key")

import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock, MagicMock

import server.auth as auth_mod
auth_mod.API_KEY = "test-key"


class FakeChoice:
    def __init__(self, delta, finish_reason=None):
        self.delta = delta
        self.finish_reason = finish_reason


class FakeChunk:
    def __init__(self, delta, finish_reason=None):
        self.choices = [FakeChoice(delta, finish_reason)]


class FakeStream:
    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        self._it = iter(self._chunks)
        return self

    async def __anext__(self):
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration


@pytest.mark.asyncio
async def test_chat_endpoint_streams_content():
    from server.app import create_app

    app = create_app()

    # Manually populate app.state since TestClient doesn't run the full lifespan for streaming
    from openai import AsyncOpenAI  # noqa: F401
    app.state.vllm_client = MagicMock()
    app.state.vllm_model = "test-model"
    app.state.qdrant = MagicMock()
    app.state.embedder = MagicMock()

    d1 = MagicMock(content="Hi", tool_calls=None)
    d2 = MagicMock(content=" there", tool_calls=None)
    stream = FakeStream([FakeChunk(d1), FakeChunk(d2, finish_reason="stop")])
    app.state.vllm_client.chat.completions.create = AsyncMock(return_value=stream)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        r = await c.post(
            "/v1/chat/completions",
            headers={"X-Api-Key": "test-key"},
            json={"messages": [{"role": "user", "content": "say hi"}], "stream": True},
        )
        body = r.text

    assert r.status_code == 200
    assert "Hi" in body or "there" in body
    assert "[DONE]" in body
