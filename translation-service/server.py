"""Translation service — NLLB-200-3.3B via CTranslate2.

A thin FastAPI wrapper. Languages are dynamic (sourced from the NLLB tokenizer,
~200 FLORES codes). Set TRANSLATE_MOCK=true to run without the model (dev/CI).
"""

from __future__ import annotations

import logging
import os
import re

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("translation-service")

MODEL_DIR = os.environ.get("MODEL_DIR", "/models/nllb-200-3.3B-ct2")
DEVICE = os.environ.get("DEVICE", "cuda")               # cpu | cuda
COMPUTE_TYPE = os.environ.get("COMPUTE_TYPE", "int8_float16")  # GPU: int8_float16; CPU: int8
TOKENIZER_NAME = os.environ.get("TOKENIZER_NAME", "facebook/nllb-200-3.3B")
MOCK = os.environ.get("TRANSLATE_MOCK", "false").lower() in ("1", "true", "yes")
BEAM_SIZE = int(os.environ.get("BEAM_SIZE", "4"))               # 4-5 cho chất lượng cao hơn
MAX_DECODING = int(os.environ.get("MAX_DECODING_LENGTH", "512"))
LENGTH_PENALTY = float(os.environ.get("LENGTH_PENALTY", "1.0"))  # >1 ưu tiên câu dài/đủ ý
REPETITION_PENALTY = float(os.environ.get("REPETITION_PENALTY", "1.1"))  # chống lặp từ
NO_REPEAT_NGRAM = int(os.environ.get("NO_REPEAT_NGRAM", "3"))    # cấm lặp cụm 3-gram; 0=tắt

app = FastAPI(title="translation-service", version="1.0.0")

_translator = None
_tokenizer = None
_langs: set[str] = set()


def _load() -> None:
    global _translator, _tokenizer, _langs
    if _translator is not None or MOCK:
        return
    import ctranslate2
    import transformers

    logger.info("Loading NLLB CTranslate2 model from %s (device=%s, %s)",
                MODEL_DIR, DEVICE, COMPUTE_TYPE)
    _translator = ctranslate2.Translator(MODEL_DIR, device=DEVICE, compute_type=COMPUTE_TYPE)
    _tokenizer = transformers.AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    # DYNAMIC: ~200 FLORES language codes straight from the tokenizer.
    # API differs across transformers versions: newer NllbTokenizer drops
    # `additional_special_tokens`, so fall back to lang_code_to_id / all_special_tokens.
    _lang_source = (
        getattr(_tokenizer, "additional_special_tokens", None)
        or getattr(_tokenizer, "lang_code_to_id", None)
        or _tokenizer.all_special_tokens
    )
    _langs = {t for t in _lang_source if "_" in t}
    logger.info("Model loaded; %d languages", len(_langs))


def _split_sentences(text: str) -> list[str]:
    """NLLB is sentence-level (<512 tokens); split then translate piecewise."""
    parts = re.split(r"(?<=[.!?。！？\n])\s+", text.strip())
    return [p for p in parts if p.strip()]


class TranslateReq(BaseModel):
    text: str
    source: str
    target: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": (_translator is not None) or MOCK, "mock": MOCK}


@app.get("/languages")
def languages() -> dict:
    if MOCK:
        return {"languages": ["vie_Latn", "eng_Latn"]}
    _load()
    return {"languages": sorted(_langs)}


@app.post("/translate")
def translate(req: TranslateReq) -> dict:
    if MOCK:
        return {"translation": f"[mock {req.source}->{req.target}] {req.text}"}

    _load()
    if _langs and (req.source not in _langs or req.target not in _langs):
        raise HTTPException(status_code=400,
                            detail=f"unsupported language: {req.source} / {req.target}")

    _tokenizer.src_lang = req.source
    outputs: list[str] = []
    for sentence in _split_sentences(req.text):
        src_tokens = _tokenizer.convert_ids_to_tokens(_tokenizer.encode(sentence))
        result = _translator.translate_batch(
            [src_tokens],
            target_prefix=[[req.target]],
            beam_size=BEAM_SIZE,
            max_decoding_length=MAX_DECODING,
            length_penalty=LENGTH_PENALTY,
            repetition_penalty=REPETITION_PENALTY,
            no_repeat_ngram_size=NO_REPEAT_NGRAM,
        )
        hyp = result[0].hypotheses[0][1:]  # drop the target-language token
        outputs.append(_tokenizer.decode(_tokenizer.convert_tokens_to_ids(hyp)))

    return {"translation": " ".join(outputs)}
