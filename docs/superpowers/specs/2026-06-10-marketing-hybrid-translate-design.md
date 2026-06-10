# Design — Hybrid `marketing` translate (NLLB → Qwen)

Date: 2026-06-10
Status: Approved (brainstorming)
Scope: `server/agent/translate.py` + new prompt config + tests. `llm`/`faithful`/router unchanged.

---

## 1. Problem

The `marketing` style currently calls **Qwen only** (`_qwen_translate_lang`, the
`else` branch in `translate_batch`). Qwen must return a JSON map
`{"0":"...","1":"..."}` which the code parses with `_extract_json_map`.

Observed failure: Qwen produces clean JSON reliably when translating to **English**,
but for other target languages (Korean/Japanese/Chinese/…) it more often emits
malformed JSON (prose around it, literal newlines inside strings, full-width
punctuation). Result: those languages hit `reason: "parse_failed"` and drop out,
while `en` succeeds. So "other languages fail, English is fine."

NLLB (the `faithful` backend) does **not** have this problem — it reliably covers
~200 FLORES languages and returns plain text, not JSON. But it produces literal,
non-marketing prose.

## 2. Goal

Make `marketing` a **hybrid pipeline**:

1. **NLLB** translates each item into the target language — broad language coverage,
   correct meaning, no JSON-parsing fragility.
2. **Qwen** transcreates that draft (seeing *both* the original source and the NLLB
   draft) into smooth, on-brand marketing copy in the target language.

This fixes the "other languages fail" problem (NLLB always provides a correct draft
as a floor) while keeping Qwen's fluency/marketing voice.

Only the `marketing` style changes. `llm` and `faithful` keep their current behavior.
The endpoint/router contract (`POST /v1/translate`, request/response shapes) is
unchanged.

## 3. Pipeline (per target language)

For each `lang` in `targetLangs`, for the whole batch of items:

```
Step 1 — NLLB draft:  source(vi) → lang   (via get_translation_client)
           success → draft[i] for every item
           NLLB fails/timeouts → skip NLLB; Qwen translates directly
                                  (== old marketing behavior)

Step 2 — Qwen transcreate:  input = (original source text + NLLB draft) per item
           system prompt: travel/marketing copywriter (from config)
           user prompt:   "here is a machine translation into <lang>; rewrite it as
                          fluent marketing copy, keep all facts/numbers/proper-nouns
                          and glossary terms unchanged, do not invent details.
                          Original: ...  Draft: ..."  (+ context, + glossary)
           returns JSON {"0":"...","1":"..."}  (batch — 1 call per language)
           parse OK            → use Qwen output (polished)
           parse fail / error  → use NLLB draft (correct meaning, less polished)

Step 3 — assemble: results[i]["translations"][lang] = chosen text
```

### 3.1 Graceful-degradation matrix (chosen behavior)

| NLLB | Qwen | Returned for that (item, lang) |
|------|------|--------------------------------|
| ✅ | ✅ | Qwen transcreation (best) |
| ✅ | ❌ | NLLB draft (correct, unpolished) |
| ❌ | ✅ | Qwen direct translation (== old marketing) |
| ❌ | ❌ | entry in `errors[]` for that (index, lang) |

"❌ Qwen" covers both a backend error (connection/timeout) **and** a JSON parse
failure after the one retry — both fall back to the NLLB draft when a draft exists.

This is why the original parse-fail problem disappears: a parse failure no longer
empties the language; it falls through to the NLLB draft.

## 4. Output-step format — chosen: Cách A (batch JSON + fallback)

Qwen transcreation runs **once per language for the whole batch** and returns a JSON
index→text map, same shape as today. On parse failure it falls back to the NLLB
draft per item.

Rejected alternative (Cách B): one Qwen call per item returning plain text.
Eliminates JSON parsing entirely but costs `items × langs` calls. Not needed —
the NLLB-draft fallback already removes the parse-fail data loss, so batch (1
call/lang) is kept for cost/latency.

## 5. Prompt externalization

System/user prompt templates move out of Python into
**`config/prompts/translate.yaml`** so they can be edited in detail without a rebuild.

Format — YAML block scalars (`|`), single file holding all translate prompts:

```yaml
version: 1
prompts:
  llm_system: |
    You are a professional translator. ...
  marketing_system: |
    You are an expert travel & marketing copywriter and translator. ...
  marketing_transcreate_user: |
    The text below is a machine translation into {target_lang}.
    Rewrite each item as fluent, evocative marketing copy in {target_lang}.
    Keep ALL facts, numbers, prices, proper nouns and glossary terms unchanged.
    Never invent details.
    {context_line}
    {glossary_line}
    Return ONLY a JSON object mapping each index to its rewritten text,
    e.g. {"0":"...","1":"..."}. No prose, no code fences.
    Items (original | machine-draft):
    {items_block}
  json_nudge: |
    Output ONLY valid JSON like {"0":"..."} — nothing else.
```

### 5.1 Loader

`PromptStore` is hardcoded to `intents.yaml` with an intents-specific schema, so it
is **not** reused directly. Add a small dedicated loader mirroring its pattern:

- New module `server/agent/translate_prompts.py` (or a `TranslatePrompts` class):
  - loads `config/prompts/translate.yaml` from `PROMPT_CONFIG_DIR` (default
    `config/prompts`),
  - hot-reload on mtime change (same approach as `PromptStore`),
  - `get(key) -> str | None`; returns `None` when the file/key is absent.
- `server/agent/translate.py` keeps the **current hardcoded strings as fallback
  defaults**: if the loader returns `None`, use the in-code default. So deleting the
  YAML file never breaks translation — exactly the `prompt_store` fallback contract.

## 6. Code changes

All in `server/agent/translate.py` unless noted:

1. `translate_batch`: in the `else` (non-faithful) branch, split `llm` vs `marketing`.
   - `llm` → unchanged (`_qwen_translate_lang`).
   - `marketing` → new hybrid path (Step 1–3 above).
2. New `_nllb_draft_batch(items, source_lang, lang) -> (drafts: dict[int,str], failed: bool)`
   — translates the batch via `get_translation_client`; on any failure returns
   `failed=True` so the caller falls back to Qwen-direct.
3. New `_qwen_transcreate_from_draft(vllm, model, items, drafts, source_lang, lang, context, glossary) -> dict`
   — builds the transcreate prompt from `translate.yaml` (with hardcoded fallback),
   calls Qwen, parses JSON with the existing `_extract_json_map`, retries once with
   the nudge; returns `{index_str: text}` (may be partial/empty).
4. New module `server/agent/translate_prompts.py` — the loader (§5.1).
5. New file `config/prompts/translate.yaml` — the prompts (§5).
6. `_system_prompt` / `_build_prompt` stay as the hardcoded fallback source for `llm`
   and direct-Qwen marketing.

Unchanged: `server/routers/translate.py`, request/response models, validation,
timeout (60s covers NLLB+Qwen since both run within the per-request budget), the
`model` field in the response (still vLLM model name for marketing — the user-facing
output is Qwen's).

## 7. Testing

Unit tests on `translate_batch` / helpers with a `FakeVLLM` (existing pattern in
`tests/test_translate.py`) and a fake NLLB client (monkeypatch
`server.translation.get_translation_client`, as in `test_faithful_uses_nllb`):

1. **Both succeed** — NLLB draft returned, Qwen polishes; result = Qwen text.
2. **NLLB ok, Qwen parse-fail** — result falls back to NLLB draft; no `errors[]`.
3. **NLLB fails, Qwen ok** — result = Qwen direct translation (old behavior).
4. **Both fail** — `errors[]` has `{index, lang, reason}` for that language.
5. **Multi-language partial** — e.g. `en` polished, `ko` falls back to NLLB draft; both present.
6. **Prompt loader** — returns YAML value when file present; returns `None`/uses
   hardcoded fallback when file/key absent.
7. **`llm` unchanged** — existing `llm` tests still pass (regression).

## 8. Assumptions to verify during implementation

- `get_translation_client().translate(text, src, lang)` accepts the same short codes
  the endpoint receives (`vi`, `en`, `ko`); `to_flores` maps them. Confirm `marketing`
  passes short codes through unchanged (it does today).
- A single request now may issue NLLB calls (N items × langs, sentence-split) **plus**
  one Qwen call per language. Confirm this stays within `TRANSLATE_TIMEOUT_SECS` (60s)
  for a max batch; if not, note it (not addressed here — out of scope).
- The circuit breaker `translation` (NLLB) opening simply makes Step 1 "fail" → Qwen
  direct path. No new circuit logic needed.

## 9. Out of scope (YAGNI)

- Changing `llm` or `faithful`.
- Per-item Qwen calls (Cách B).
- New request/response fields, new endpoint, new style name.
- Caching NLLB drafts, parallelizing languages, tuning the 60s budget.
