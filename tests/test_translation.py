"""Translation API tests — ai-agent router (mock backend) + service (mock mode)."""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

from server.translation import to_flores


# --------------------------------------------------------------------------- #
# Language code resolution (dynamic — FLORES pass-through)
# --------------------------------------------------------------------------- #
class TestToFlores:
    def test_alias(self):
        assert to_flores("vi") == "vie_Latn"
        assert to_flores("en") == "eng_Latn"

    def test_flores_passthrough(self):
        assert to_flores("vie_Latn") == "vie_Latn"
        assert to_flores("yue_Hant") == "yue_Hant"   # any of ~200, no hardcoded list

    def test_unknown_short_code_passed_through(self):
        # left for the service to validate (dynamic)
        assert to_flores("xx") == "xx"

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            to_flores("")


# --------------------------------------------------------------------------- #
# /v1/translate endpoint
# --------------------------------------------------------------------------- #
class _FakeClient:
    async def translate(self, text, source_lang, target_lang):
        if not source_lang:
            raise ValueError("language code is required")
        return f"T[{source_lang}->{target_lang}]:{text}"

    async def languages(self):
        return ["vie_Latn", "eng_Latn"]


def _client(monkeypatch, *, enabled=True, fake=True):
    monkeypatch.setenv("DEV_MODE", "true")
    monkeypatch.delenv("ENABLE_RAG", raising=False)
    if enabled:
        monkeypatch.setenv("ENABLE_TRANSLATE", "true")
    else:
        monkeypatch.delenv("ENABLE_TRANSLATE", raising=False)
    monkeypatch.setattr("server.auth.API_KEY", "test-key")
    if fake:
        monkeypatch.setattr("server.routers.translate.get_translation_client", lambda: _FakeClient())
    from fastapi.testclient import TestClient
    from server.app import create_app

    return TestClient(create_app())


class TestTranslateEndpoint:
    H = {"X-Api-Key": "test-key"}

    def test_disabled_returns_503(self, monkeypatch):
        with _client(monkeypatch, enabled=False, fake=False) as c:
            r = c.post("/v1/translate", headers=self.H,
                       json={"text": "hi", "source_lang": "vi", "target_lang": "en"})
            assert r.status_code == 503

    def test_translate_ok(self, monkeypatch):
        with _client(monkeypatch) as c:
            r = c.post("/v1/translate", headers=self.H,
                       json={"text": "xin chao", "source_lang": "vi", "target_lang": "en"})
            assert r.status_code == 200
            body = r.json()
            assert body["translated"] == "T[vi->en]:xin chao"
            assert body["model"] == "nllb-200-3.3B"

    def test_empty_text_422(self, monkeypatch):
        with _client(monkeypatch) as c:
            r = c.post("/v1/translate", headers=self.H,
                       json={"text": "  ", "source_lang": "vi", "target_lang": "en"})
            assert r.status_code == 422

    def test_bad_language_400(self, monkeypatch):
        with _client(monkeypatch) as c:
            r = c.post("/v1/translate", headers=self.H,
                       json={"text": "hi", "source_lang": "", "target_lang": "en"})
            assert r.status_code == 400

    def test_auth_required(self, monkeypatch):
        with _client(monkeypatch) as c:
            r = c.post("/v1/translate",
                       json={"text": "hi", "source_lang": "vi", "target_lang": "en"})
            assert r.status_code == 403

    def test_languages(self, monkeypatch):
        with _client(monkeypatch) as c:
            r = c.get("/v1/translate/languages", headers=self.H)
            assert r.status_code == 200
            assert "vie_Latn" in r.json()["languages"]


# --------------------------------------------------------------------------- #
# translation-service (MOCK mode — no NLLB needed)
# --------------------------------------------------------------------------- #
def _load_service(monkeypatch):
    monkeypatch.setenv("TRANSLATE_MOCK", "true")
    path = pathlib.Path(__file__).parent.parent / "translation-service" / "server.py"
    spec = importlib.util.spec_from_file_location("trsvc", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestService:
    def test_split_sentences(self, monkeypatch):
        mod = _load_service(monkeypatch)
        parts = mod._split_sentences("Hello world. How are you? Fine!")
        assert len(parts) == 3

    def test_mock_translate(self, monkeypatch):
        mod = _load_service(monkeypatch)
        out = mod.translate(mod.TranslateReq(text="hi", source="vie_Latn", target="eng_Latn"))
        assert "mock" in out["translation"]

    def test_mock_languages(self, monkeypatch):
        mod = _load_service(monkeypatch)
        assert "vie_Latn" in mod.languages()["languages"]
