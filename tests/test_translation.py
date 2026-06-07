"""NLLB translation-service + FLORES mapping tests (faithful path)."""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

from server.translation import to_flores


class TestToFlores:
    def test_alias(self):
        assert to_flores("vi") == "vie_Latn"
        assert to_flores("en") == "eng_Latn"

    def test_flores_passthrough(self):
        assert to_flores("vie_Latn") == "vie_Latn"
        assert to_flores("yue_Hant") == "yue_Hant"

    def test_unknown_short_code_passed_through(self):
        assert to_flores("xx") == "xx"

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            to_flores("")


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
