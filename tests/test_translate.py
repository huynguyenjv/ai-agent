"""Translate API (batch contract) — translate_batch unit + endpoint."""

from __future__ import annotations

from server.agent.translate import translate_batch


# ----- fake vLLM client ----------------------------------------------------- #
class _Msg:
    def __init__(self, content): self.content = content


class _Choice:
    def __init__(self, content): self.message = _Msg(content)


class _Resp:
    def __init__(self, content): self.choices = [_Choice(content)]


class _Completions:
    def __init__(self, responses): self._responses = list(responses); self.calls = 0

    async def create(self, **kw):
        r = self._responses[min(self.calls, len(self._responses) - 1)]
        self.calls += 1
        return _Resp(r)


class FakeVLLM:
    def __init__(self, responses):
        self.chat = type("C", (), {"completions": _Completions(responses)})()


ITEMS = [
    {"ref": {"id": 123, "field": "name"}, "text": "Quản trị viên"},
    {"ref": {"id": 123, "field": "desc"}, "text": "Người quản lý hệ thống"},
]


class TestTranslateBatch:
    async def test_happy_two_items_two_langs(self):
        vllm = FakeVLLM([
            '{"0":"Administrator","1":"System manager"}',   # en
            '{"0":"관리자","1":"시스템 관리자"}',              # ko
        ])
        results, errors = await translate_batch(
            vllm, "qwen", ITEMS, "vi", ["en", "ko"])
        assert errors == []
        assert results[0]["ref"] == {"id": 123, "field": "name"}
        assert results[0]["translations"] == {"en": "Administrator", "ko": "관리자"}
        assert results[1]["translations"]["en"] == "System manager"

    async def test_retry_on_bad_json(self):
        vllm = FakeVLLM(["not json at all", '{"0":"OK"}'])
        results, errors = await translate_batch(
            vllm, "qwen", [ITEMS[0]], "vi", ["en"])
        assert errors == []
        assert results[0]["translations"]["en"] == "OK"
        assert vllm.chat.completions.calls == 2   # retried once

    async def test_both_attempts_fail_go_to_errors(self):
        vllm = FakeVLLM(["junk", "still junk"])
        results, errors = await translate_batch(
            vllm, "qwen", [ITEMS[0]], "vi", ["en"])
        assert results[0]["translations"] == {}
        assert errors == [{"index": 0, "lang": "en", "reason": "parse_failed"}]

    async def test_partial_failure_keeps_rest(self):
        # en ok, ko junk both times → ko error, en still returned
        vllm = FakeVLLM(['{"0":"Administrator"}', "junk", "junk"])
        results, errors = await translate_batch(
            vllm, "qwen", [ITEMS[0]], "vi", ["en", "ko"])
        assert results[0]["translations"] == {"en": "Administrator"}
        assert errors == [{"index": 0, "lang": "ko", "reason": "parse_failed"}]

    async def test_marketing_style_uses_qwen(self):
        vllm = FakeVLLM(['{"0":"Welcome to paradise"}'])
        results, errors = await translate_batch(
            vllm, "qwen", [ITEMS[0]], "vi", ["en"], style="marketing", context="travel")
        assert results[0]["translations"]["en"] == "Welcome to paradise"

    async def test_faithful_uses_nllb(self, monkeypatch):
        class FakeNLLB:
            async def translate(self, text, src, tgt):
                return f"NLLB[{src}->{tgt}]"

        monkeypatch.setattr("server.translation.get_translation_client", lambda: FakeNLLB())
        results, errors = await translate_batch(
            None, "qwen", [ITEMS[0]], "vi", ["en"], style="faithful")
        assert errors == []
        assert results[0]["translations"]["en"] == "NLLB[vi->en]"


# ----- endpoint ------------------------------------------------------------- #
def _client(monkeypatch):
    monkeypatch.setenv("DEV_MODE", "true")
    monkeypatch.delenv("ENABLE_RAG", raising=False)
    monkeypatch.setattr("server.auth.API_KEY", "test-key")
    from fastapi.testclient import TestClient
    from server.app import create_app
    return TestClient(create_app())


class TestEndpoint:
    H = {"X-Api-Key": "test-key"}
    BODY = {"sourceLang": "vi", "targetLangs": ["en"],
            "items": [{"ref": {"id": 1}, "text": "Xin chào"}]}

    def test_happy_path_shape(self, monkeypatch):
        async def fake_batch(*a, **k):
            return ([{"ref": {"id": 1}, "translations": {"en": "Hello"}}], [])
        monkeypatch.setattr("server.routers.translate.translate_batch", fake_batch)
        with _client(monkeypatch) as c:
            r = c.post("/v1/translate", headers=self.H, json=self.BODY)
            assert r.status_code == 200
            body = r.json()
            assert body["results"][0]["translations"]["en"] == "Hello"
            assert body["errors"] == []
            assert "model" in body

    def test_auth_required(self, monkeypatch):
        with _client(monkeypatch) as c:
            assert c.post("/v1/translate", json=self.BODY).status_code == 403

    def test_missing_source_422(self, monkeypatch):
        with _client(monkeypatch) as c:
            r = c.post("/v1/translate", headers=self.H,
                       json={"targetLangs": ["en"], "items": [{"text": "hi"}]})
            assert r.status_code == 422

    def test_empty_items_422(self, monkeypatch):
        with _client(monkeypatch) as c:
            r = c.post("/v1/translate", headers=self.H,
                       json={"sourceLang": "vi", "targetLangs": ["en"], "items": []})
            assert r.status_code == 422

    def test_too_many_items_422(self, monkeypatch):
        with _client(monkeypatch) as c:
            r = c.post("/v1/translate", headers=self.H, json={
                "sourceLang": "vi", "targetLangs": ["en"],
                "items": [{"text": "x"}] * 51})
            assert r.status_code == 422
