"""Translate API — POST /v1/translate (design approved 2026-06-07).

Batch, multi-target, ref echo-back, explicit source language, optional context +
glossary. Stateless: calls vLLM directly (style llm/marketing) or the NLLB
service (style faithful).
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import BaseModel

from server.auth import verify_api_key
from server.agent.translate import translate_batch

logger = logging.getLogger("server.translate")

router = APIRouter()

TRANSLATE_MAX_ITEMS = int(os.environ.get("TRANSLATE_MAX_ITEMS", "50"))
TRANSLATE_MAX_CHARS = int(os.environ.get("TRANSLATE_MAX_CHARS", "20000"))
TRANSLATE_TIMEOUT_SECS = float(os.environ.get("TRANSLATE_TIMEOUT_SECS", "60"))
_VALID_STYLES = {"llm", "marketing", "faithful"}


class TranslateItem(BaseModel):
    ref: Any = None          # arbitrary, echoed back untouched
    text: str


class TranslateRequest(BaseModel):
    sourceLang: str          # required (explicit — no auto)
    targetLangs: list[str]
    items: list[TranslateItem]
    context: str | None = None
    style: str = "llm"
    glossary: list[str] | None = None


@router.post("/v1/translate")
async def translate(
    body: TranslateRequest,
    req: Request,
    x_api_key: str = Header(None),
    authorization: str = Header(None),
) -> dict:
    verify_api_key(req, x_api_key, authorization)

    # ---- validation (422) ----
    if not body.sourceLang or not body.sourceLang.strip():
        raise HTTPException(422, "sourceLang is required")
    if not body.items:
        raise HTTPException(422, "items must not be empty")
    if not body.targetLangs:
        raise HTTPException(422, "targetLangs must not be empty")
    if len(body.items) > TRANSLATE_MAX_ITEMS:
        raise HTTPException(422, f"too many items (> {TRANSLATE_MAX_ITEMS})")
    total_chars = sum(len(it.text or "") for it in body.items)
    if total_chars > TRANSLATE_MAX_CHARS:
        raise HTTPException(422, f"total text too long (> {TRANSLATE_MAX_CHARS} chars)")

    style = body.style if body.style in _VALID_STYLES else "llm"
    items = [{"ref": it.ref, "text": it.text} for it in body.items]

    try:
        results, errors = await asyncio.wait_for(
            translate_batch(
                req.app.state.vllm_client,
                req.app.state.vllm_model,
                items,
                body.sourceLang,
                body.targetLangs,
                context=body.context,
                style=style,
                glossary=body.glossary,
            ),
            timeout=TRANSLATE_TIMEOUT_SECS,
        )
    except asyncio.TimeoutError:
        raise HTTPException(504, "translation timed out")
    except Exception as e:
        logger.warning("translate failed: %s", e)
        raise HTTPException(502, "translation backend error")

    return {
        "results": results,
        "errors": errors,
        "model": "nllb-200-3.3B" if style == "faithful" else req.app.state.vllm_model,
    }
