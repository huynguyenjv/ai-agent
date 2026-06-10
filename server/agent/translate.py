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

_DEFAULT_TRANSCREATE_USER = (
    "The items below were machine-translated into {target_lang}. For each item you "
    "are given the ORIGINAL source text and the machine DRAFT.\n"
    "Rewrite each item as fluent, evocative marketing copy in {target_lang}:\n"
    "- Fix awkwardness or errors in the draft using the original as the source of truth.\n"
    "- Keep ALL facts, numbers, prices, proper nouns and glossary terms unchanged.\n"
    "- Never invent details that are not in the original.\n"
    "{context_line}\n"
    "{glossary_line}\n"
    'Return ONLY a JSON object mapping each item index (as a string) to its '
    'rewritten text, e.g. {"0":"..."}. No prose, no code fences.\n'
    "Items:\n{items_block}"
)


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


async def _nllb_draft_batch(items, source_lang, target_lang):
    """Translate the whole batch into target_lang via NLLB.

    Returns (drafts {index:int -> text}, failed:bool). On ANY NLLB error the
    whole language is considered failed (caller falls back to Qwen-direct).
    """
    from server.translation import get_translation_client

    client = get_translation_client()
    drafts: dict[int, str] = {}
    try:
        for idx, it in enumerate(items):
            drafts[idx] = await client.translate(it["text"], source_lang, target_lang)
        return drafts, False
    except Exception as e:
        logger.warning("nllb draft failed (lang=%s): %s", target_lang, e)
        return drafts, True


def _build_transcreate_prompt(items, drafts, source_lang, target_lang, context, glossary) -> str:
    """Build the marketing transcreate user prompt from template + fallback.

    Uses str.replace (not str.format) so literal { } in the JSON example survive.
    """
    from server.agent.translate_prompts import get_translate_prompts

    tmpl = get_translate_prompts().get("marketing_transcreate_user") or _DEFAULT_TRANSCREATE_USER
    context_line = f"Domain/context: {context}" if context else ""
    glossary_line = (
        "Keep these terms unchanged (do NOT translate): " + ", ".join(glossary)
        if glossary else ""
    )
    items_block = "\n".join(
        f'{idx}: original="{it["text"]}" | draft="{drafts.get(idx, "")}"'
        for idx, it in enumerate(items)
    )
    out = tmpl
    for key, val in (
        ("{target_lang}", _lang_name(target_lang)),
        ("{context_line}", context_line),
        ("{glossary_line}", glossary_line),
        ("{items_block}", items_block),
    ):
        out = out.replace(key, val)
    return out


async def _qwen_transcreate_from_draft(vllm_client, model, items, drafts,
                                       source_lang, target_lang, context, glossary) -> dict:
    """Qwen rewrites the NLLB draft into marketing copy. Returns {index_str: text}."""
    from server.agent.translate_prompts import get_translate_prompts

    store = get_translate_prompts()
    system = store.get("marketing_system") or _system_prompt("marketing")
    nudge = store.get("json_nudge") or 'Output ONLY valid JSON like {"0":"..."} — nothing else.'
    user = _build_transcreate_prompt(items, drafts, source_lang, target_lang, context, glossary)
    base = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    async def _call(messages) -> dict | None:
        resp = await vllm_client.chat.completions.create(
            model=model, messages=messages,
            temperature=TRANSLATE_MARKETING_TEMPERATURE,
            max_tokens=TRANSLATE_MAX_TOKENS, stream=False,
        )
        return _extract_json_map(resp.choices[0].message.content or "")

    parsed = await _call(base)
    if parsed is None:
        parsed = await _call(base + [{"role": "user", "content": nudge}])
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
