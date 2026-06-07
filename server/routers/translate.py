"""Translation endpoint — POST /v1/translate (multi-model router to NLLB).

Gated by ENABLE_TRANSLATE; forwards to the dedicated translation-service.
"""

from __future__ import annotations

import logging
import os

import httpx
from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import BaseModel

from server.auth import verify_api_key
from server.translation import (
    enable_translate,
    get_translation_client,
)

logger = logging.getLogger("server.translate")

router = APIRouter()

MAX_TRANSLATE_CHARS = int(os.environ.get("MAX_TRANSLATE_CHARS", "20000"))


class TranslateRequest(BaseModel):
    text: str
    source_lang: str
    target_lang: str


def _require_enabled() -> None:
    if not enable_translate():
        raise HTTPException(status_code=503, detail="Translation disabled (ENABLE_TRANSLATE=false)")


@router.post("/v1/translate")
async def translate(
    body: TranslateRequest,
    req: Request,
    x_api_key: str = Header(None),
    authorization: str = Header(None),
) -> dict:
    verify_api_key(req, x_api_key, authorization)
    _require_enabled()

    if not body.text or not body.text.strip():
        raise HTTPException(status_code=422, detail="text must not be empty")
    if len(body.text) > MAX_TRANSLATE_CHARS:
        raise HTTPException(status_code=422, detail=f"text too long (> {MAX_TRANSLATE_CHARS} chars)")

    try:
        translated = await get_translation_client().translate(
            body.text, body.source_lang, body.target_lang
        )
    except ValueError as e:                       # bad language code
        raise HTTPException(status_code=400, detail=str(e))
    except httpx.HTTPStatusError as e:            # service rejected (e.g. unsupported lang)
        code = e.response.status_code
        detail = e.response.text[:200]
        raise HTTPException(status_code=400 if code == 400 else 502, detail=detail)
    except Exception as e:                         # service down / circuit open
        logger.warning("translation backend error: %s", e)
        raise HTTPException(status_code=502, detail="Translation backend unavailable")

    return {
        "translated": translated,
        "source_lang": body.source_lang,
        "target_lang": body.target_lang,
        "model": "nllb-200-3.3B",
    }


@router.get("/v1/translate/languages")
async def languages(
    req: Request,
    x_api_key: str = Header(None),
    authorization: str = Header(None),
) -> dict:
    verify_api_key(req, x_api_key, authorization)
    _require_enabled()
    try:
        langs = await get_translation_client().languages()
    except Exception as e:
        logger.warning("translation languages error: %s", e)
        raise HTTPException(status_code=502, detail="Translation backend unavailable")
    return {"languages": langs}
