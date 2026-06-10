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


# ----- prompt loader -------------------------------------------------------- #
class TestTranslatePrompts:
    def test_absent_file_returns_none(self, tmp_path):
        from server.agent.translate_prompts import reset_translate_prompts
        store = reset_translate_prompts(str(tmp_path))
        assert store.get("marketing_system") is None

    def test_present_file_returns_value(self, tmp_path):
        from server.agent.translate_prompts import reset_translate_prompts
        (tmp_path / "translate.yaml").write_text(
            "prompts:\n  marketing_system: |\n    Hello copywriter\n", encoding="utf-8")
        store = reset_translate_prompts(str(tmp_path))
        assert store.get("marketing_system") == "Hello copywriter"
        assert store.get("missing_key") is None


# ----- hybrid marketing helpers --------------------------------------------- #
class TestNllbDraftBatch:
    async def test_success_returns_drafts(self, monkeypatch):
        from server.agent.translate import _nllb_draft_batch

        class FakeNLLB:
            async def translate(self, text, src, tgt):
                return f"draft[{text}->{tgt}]"
        monkeypatch.setattr("server.translation.get_translation_client", lambda: FakeNLLB())
        drafts, failed = await _nllb_draft_batch(ITEMS, "vi", "en")
        assert failed is False
        assert drafts[0] == "draft[Quản trị viên->en]"
        assert drafts[1].startswith("draft[")

    async def test_failure_returns_failed_true(self, monkeypatch):
        from server.agent.translate import _nllb_draft_batch

        class FakeNLLB:
            async def translate(self, text, src, tgt):
                raise RuntimeError("nllb down")
        monkeypatch.setattr("server.translation.get_translation_client", lambda: FakeNLLB())
        drafts, failed = await _nllb_draft_batch([ITEMS[0]], "vi", "en")
        assert failed is True


class TestQwenTranscreate:
    async def test_uses_draft_and_returns_polished(self):
        from server.agent.translate import _qwen_transcreate_from_draft
        vllm = FakeVLLM(['{"0":"Polished EN"}'])
        out = await _qwen_transcreate_from_draft(
            vllm, "qwen", [ITEMS[0]], {0: "rough draft"}, "vi", "en", None, None)
        assert out == {"0": "Polished EN"}

    async def test_retries_once_on_bad_json(self):
        from server.agent.translate import _qwen_transcreate_from_draft
        vllm = FakeVLLM(["not json", '{"0":"OK"}'])
        out = await _qwen_transcreate_from_draft(
            vllm, "qwen", [ITEMS[0]], {0: "d"}, "vi", "en", None, None)
        assert out == {"0": "OK"}
        assert vllm.chat.completions.calls == 2

    async def test_prompt_includes_original_and_draft(self):
        from server.agent.translate import _build_transcreate_prompt
        prompt = _build_transcreate_prompt(
            [ITEMS[0]], {0: "rough draft"}, "vi", "en",
            context="travel", glossary=["Vtrip"])
        assert "Quản trị viên" in prompt          # original
        assert "rough draft" in prompt            # draft
        assert "travel" in prompt                 # context
        assert "Vtrip" in prompt                  # glossary
        assert '{"0":"..."}' in prompt            # literal JSON example survived


class TestMarketingHybrid:
    def _nllb(self, monkeypatch, fn):
        class FakeNLLB:
            async def translate(self, text, src, tgt):
                return fn(text, src, tgt)
        monkeypatch.setattr("server.translation.get_translation_client", lambda: FakeNLLB())

    async def test_both_ok_uses_qwen(self, monkeypatch):
        self._nllb(monkeypatch, lambda t, s, g: "NLLB draft")
        vllm = FakeVLLM(['{"0":"Polished!"}'])
        results, errors = await translate_batch(
            vllm, "qwen", [ITEMS[0]], "vi", ["en"], style="marketing")
        assert errors == []
        assert results[0]["translations"]["en"] == "Polished!"

    async def test_nllb_ok_qwen_parsefail_falls_back_to_draft(self, monkeypatch):
        self._nllb(monkeypatch, lambda t, s, g: "NLLB draft")
        vllm = FakeVLLM(["junk", "still junk"])
        results, errors = await translate_batch(
            vllm, "qwen", [ITEMS[0]], "vi", ["en"], style="marketing")
        assert errors == []
        assert results[0]["translations"]["en"] == "NLLB draft"

    async def test_nllb_down_qwen_direct(self, monkeypatch):
        def boom(t, s, g):
            raise RuntimeError("nllb down")
        self._nllb(monkeypatch, boom)
        vllm = FakeVLLM(['{"0":"Qwen direct"}'])
        results, errors = await translate_batch(
            vllm, "qwen", [ITEMS[0]], "vi", ["en"], style="marketing")
        assert errors == []
        assert results[0]["translations"]["en"] == "Qwen direct"

    async def test_both_fail_goes_to_errors(self, monkeypatch):
        def boom(t, s, g):
            raise RuntimeError("nllb down")
        self._nllb(monkeypatch, boom)

        class _BoomCompletions:
            calls = 0
            async def create(self, **kw):
                raise RuntimeError("vllm down")

        class BoomVLLM:
            def __init__(self):
                self.chat = type("C", (), {"completions": _BoomCompletions()})()

        results, errors = await translate_batch(
            BoomVLLM(), "qwen", [ITEMS[0]], "vi", ["en"], style="marketing")
        assert results[0]["translations"] == {}
        assert errors == [{"index": 0, "lang": "en", "reason": "parse_failed"}]

    async def test_multilang_partial_en_polished_ko_draft(self, monkeypatch):
        self._nllb(monkeypatch, lambda t, s, g: f"draft-{g}")
        # en: 1 call parses; ko: 2 calls both junk -> falls back to draft-ko
        vllm = FakeVLLM(['{"0":"EN polished"}', "junk", "junk"])
        results, errors = await translate_batch(
            vllm, "qwen", [ITEMS[0]], "vi", ["en", "ko"], style="marketing")
        assert errors == []
        assert results[0]["translations"]["en"] == "EN polished"
        assert results[0]["translations"]["ko"] == "draft-ko"


class TestQwenDirectUsesLoader:
    async def test_system_prompt_from_yaml_when_present(self, monkeypatch, tmp_path):
        from server.agent.translate_prompts import reset_translate_prompts
        (tmp_path / "translate.yaml").write_text(
            'prompts:\n  llm_system: |\n    CUSTOM LLM SYSTEM\n', encoding="utf-8")
        reset_translate_prompts(str(tmp_path))

        captured = {}
        class _Comp:
            async def create(self, **kw):
                captured["system"] = kw["messages"][0]["content"]
                return _Resp('{"0":"x"}')
        class _VLLM:
            def __init__(self): self.chat = type("C", (), {"completions": _Comp()})()

        from server.agent.translate import _qwen_translate_lang
        await _qwen_translate_lang(_VLLM(), "qwen", [ITEMS[0]], "vi", "en", None, "llm", None)
        reset_translate_prompts()  # restore default singleton for other tests
        assert captured["system"] == "CUSTOM LLM SYSTEM"

    async def test_falls_back_to_hardcoded_when_absent(self, monkeypatch, tmp_path):
        from server.agent.translate_prompts import reset_translate_prompts
        reset_translate_prompts(str(tmp_path))  # empty dir -> no yaml -> None

        captured = {}
        class _Comp:
            async def create(self, **kw):
                captured["system"] = kw["messages"][0]["content"]
                return _Resp('{"0":"x"}')
        class _VLLM:
            def __init__(self): self.chat = type("C", (), {"completions": _Comp()})()

        from server.agent.translate import _qwen_translate_lang, _system_prompt
        await _qwen_translate_lang(_VLLM(), "qwen", [ITEMS[0]], "vi", "en", None, "marketing", None)
        reset_translate_prompts()
        assert captured["system"] == _system_prompt("marketing")
