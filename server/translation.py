"""Translation client — multi-model router to a separate NLLB-200 backend.

The coding model is served by vLLM (decoder-only). NLLB-200-3.3B is an
encoder-decoder seq2seq model, so it runs in a dedicated `translation-service`
(CTranslate2). This module is a thin router: it forwards FLORES-200 language
codes to that service (reusing the circuit breaker) — the service is the source
of truth for which languages are supported (dynamic, ~200 languages).

Opt-in via ENABLE_TRANSLATE; backend at TRANSLATION_URL.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger("server.translation")

TRANSLATION_URL = os.environ.get("TRANSLATION_URL", "http://localhost:8100")
TRANSLATION_TIMEOUT = float(os.environ.get("TRANSLATION_TIMEOUT", "60"))

# Optional convenience aliases (short code → FLORES). NOT a whitelist — any
# FLORES code is accepted and validated by the service (dynamic ~200 languages).
ALIASES: dict[str, str] = {
    "vi": "vie_Latn", "en": "eng_Latn", "zh": "zho_Hans", "zh-tw": "zho_Hant",
    "ja": "jpn_Jpan", "ko": "kor_Hang", "fr": "fra_Latn", "de": "deu_Latn",
    "es": "spa_Latn", "th": "tha_Thai", "id": "ind_Latn", "ru": "rus_Cyrl",
}


def enable_translate() -> bool:
    return os.environ.get("ENABLE_TRANSLATE", "false").lower() in ("1", "true", "yes")


def to_flores(code: str) -> str:
    """Resolve a language code to a FLORES-200 code.

    - already a FLORES code (contains '_') → pass through (supports all ~200)
    - known alias → mapped
    - otherwise → returned as-is for the service to validate
    """
    if not code:
        raise ValueError("language code is required")
    if "_" in code:
        return code
    return ALIASES.get(code.lower(), code)


class TranslationClient:
    def __init__(self, url: str = TRANSLATION_URL):
        self._url = url.rstrip("/")

    async def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        src = to_flores(source_lang)
        tgt = to_flores(target_lang)
        if not text or not text.strip():
            return ""

        import httpx

        from server.circuit_breaker import get_circuit_breaker

        cb = get_circuit_breaker("translation")

        async def _call() -> str:
            async with httpx.AsyncClient(timeout=TRANSLATION_TIMEOUT) as client:
                resp = await client.post(
                    f"{self._url}/translate",
                    json={"text": text, "source": src, "target": tgt},
                )
                resp.raise_for_status()
                return resp.json()["translation"]

        return await cb.call(_call)

    async def languages(self) -> list[str]:
        """Proxy the service's dynamic language list (sourced from the model)."""
        import httpx

        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{self._url}/languages")
            resp.raise_for_status()
            return resp.json().get("languages", [])


_client: TranslationClient | None = None


def get_translation_client() -> TranslationClient:
    global _client
    if _client is None:
        _client = TranslationClient()
    return _client


def reset_translation_client() -> None:
    global _client
    _client = None
