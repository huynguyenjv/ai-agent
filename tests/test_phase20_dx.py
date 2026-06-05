"""Phase 20 — dev mode (20.2) + CLI (20.4) unit tests."""

from __future__ import annotations

from argparse import Namespace

from server.dev_mode import build_mock_vllm_client, is_dev_mode, MOCK_REPLY


class TestDevMode:
    def test_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv("DEV_MODE", raising=False)
        assert is_dev_mode() is False

    async def test_mock_streams_reply(self):
        client = build_mock_vllm_client()
        stream = await client.chat.completions.create(stream=True, model="m", messages=[])
        out = []
        async for chunk in stream:
            out.append(chunk.choices[0].delta.content)
        assert "".join(out).strip() == MOCK_REPLY

    async def test_mock_non_streaming(self):
        client = build_mock_vllm_client()
        resp = await client.chat.completions.create(stream=False, model="m", messages=[])
        assert resp.choices[0].message.content == MOCK_REPLY

    async def test_mock_models_list(self):
        client = build_mock_vllm_client()
        assert await client.models.list()


class TestCli:
    def test_config_unreachable_returns_nonzero(self):
        import cli

        rc = cli.cmd_config(Namespace(url="http://127.0.0.1:1", api_key="dev-secret-key"))
        assert rc == 1

    def test_agents_flag_prefixes_message(self, monkeypatch):
        import cli

        captured = {}

        class FakeStream:
            def __enter__(self):
                return self
            def __exit__(self, *a):
                return False
            status_code = 200
            def iter_lines(self):
                return iter(["data: [DONE]"])

        def fake_stream(method, url, json=None, headers=None, timeout=None):
            captured["content"] = json["messages"][0]["content"]
            return FakeStream()

        monkeypatch.setattr(cli.httpx, "stream", fake_stream)
        cli.cmd_chat(Namespace(url="http://x", api_key="k", message="do it", agents=True))
        assert captured["content"].startswith("/agents ")
