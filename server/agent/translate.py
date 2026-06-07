"""Batch translation — Translate API (design approved 2026-06-07).

Stateless: calls vLLM (Qwen) directly, no agent graph (like review_analyze).
Multi-target, batch items, ref echo-back, explicit source language, optional
domain context + glossary.

Styles:
- "llm"       : accurate + fluent translation via Qwen (default)
- "marketing" : transcreation via Qwen (evocative, keeps all facts)
- "faithful"  : deterministic via the NLLB service
"""

from __future__ import annotations

import json
import logging
import os
import re

logger = logging.getLogger("server.agent.translate")

TRANSLATE_TEMPERATURE = float(os.environ.get("TRANSLATE_TEMPERATURE", "0.2"))
TRANSLATE_MARKETING_TEMPERATURE = float(os.environ.get("TRANSLATE_MARKETING_TEMPERATURE", "0.5"))
TRANSLATE_MAX_TOKENS = int(os.environ.get("TRANSLATE_MAX_TOKENS", "4096"))

LANG_NAMES = {
    "vi": "Vietnamese", "en": "English", "ko": "Korean", "ja": "Japanese",
    "zh": "Chinese", "fr": "French", "de": "German", "es": "Spanish",
    "th": "Thai", "id": "Indonesian", "ru": "Russian", "pt": "Portuguese",
}


def _lang_name(code: str) -> str:
    return LANG_NAMES.get((code or "").lower(), code)


def _system_prompt(style: str) -> str:
    if style == "marketing":
        return (
            "You are an expert travel & marketing copywriter and translator. "
            "Transcreate the text into natural, evocative marketing copy in the "
            "target language. Keep ALL facts, numbers and proper nouns unchanged. "
            "Be creative but accurate — never invent details."
        )
    return (
        "You are a professional translator. Translate accurately and fluently into "
        "the target language, preserving meaning, proper nouns and numbers. Do not "
        "add or omit information."
    )


def _build_prompt(items, source_lang, target_lang, context, glossary) -> str:
    lines = [f"Translate from {_lang_name(source_lang)} to {_lang_name(target_lang)}."]
    if context:
        lines.append(f"Domain/context: {context}")
    if glossary:
        lines.append("Keep these terms unchanged (do NOT translate): " + ", ".join(glossary))
    lines.append(
        'Return ONLY a JSON object mapping each item index (as a string) to its '
        'translation, e.g. {"0":"...","1":"..."}. No prose, no code fences.'
    )
    lines.append("Items:")
    for idx, it in enumerate(items):
        lines.append(f"{idx}: {it['text']}")
    return "\n".join(lines)


def _extract_json_map(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None


async def _qwen_translate_lang(vllm_client, model, items, source_lang, target_lang,
                               context, style, glossary) -> dict:
    """Translate all items into one target language via Qwen. Returns {index_str: text}."""
    temperature = TRANSLATE_MARKETING_TEMPERATURE if style == "marketing" else TRANSLATE_TEMPERATURE
    base = [
        {"role": "system", "content": _system_prompt(style)},
        {"role": "user", "content": _build_prompt(items, source_lang, target_lang, context, glossary)},
    ]

    async def _call(messages) -> dict | None:
        resp = await vllm_client.chat.completions.create(
            model=model, messages=messages, temperature=temperature,
            max_tokens=TRANSLATE_MAX_TOKENS, stream=False,
        )
        return _extract_json_map(resp.choices[0].message.content or "")

    parsed = await _call(base)
    if parsed is None:  # retry once with a stricter nudge
        parsed = await _call(base + [
            {"role": "user", "content": 'Output ONLY valid JSON like {"0":"..."} — nothing else.'}
        ])
    return parsed or {}


async def translate_batch(vllm_client, model, items, source_lang, target_langs,
                          context=None, style="llm", glossary=None):
    """Translate items into target_langs. Returns (results, errors).

    results: [{"ref": <echoed>, "translations": {lang: text}}]  (same order as items)
    errors:  [{"index": i, "lang": l, "reason": "..."}]
    """
    results = [{"ref": it.get("ref"), "translations": {}} for it in items]
    errors: list[dict] = []

    for lang in target_langs:
        if style == "faithful":
            from server.translation import get_translation_client

            client = get_translation_client()
            for idx, it in enumerate(items):
                try:
                    results[idx]["translations"][lang] = await client.translate(
                        it["text"], source_lang, lang)
                except Exception as e:
                    errors.append({"index": idx, "lang": lang, "reason": str(e)[:120]})
        else:
            try:
                per = await _qwen_translate_lang(
                    vllm_client, model, items, source_lang, lang, context, style, glossary)
            except Exception as e:
                logger.warning("qwen translate failed (lang=%s): %s", lang, e)
                per = {}
            for idx in range(len(items)):
                val = per.get(str(idx))
                if val:
                    results[idx]["translations"][lang] = val
                else:
                    errors.append({"index": idx, "lang": lang, "reason": "parse_failed"})

    return results, errors
