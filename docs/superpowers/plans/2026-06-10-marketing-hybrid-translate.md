# Hybrid `marketing` Translate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the `marketing` translate style a hybrid pipeline — NLLB produces a correct draft in the target language, then Qwen transcreates it into fluent marketing copy — with graceful degradation so no language silently drops.

**Architecture:** Only `server/agent/translate.py` gains logic. For `marketing`, each target language goes NLLB-draft → Qwen-transcreate(source + draft) → merge (Qwen wins, NLLB draft is the fallback floor). `llm` and `faithful` are untouched. Prompts move to `config/prompts/translate.yaml`, loaded by a small dedicated loader with hardcoded fallbacks (so a missing file never breaks translation).

**Tech Stack:** Python, FastAPI, pytest (async), PyYAML, OpenAI-compatible vLLM client, NLLB translation-service client.

**Spec:** `docs/superpowers/specs/2026-06-10-marketing-hybrid-translate-design.md`

---

## File Structure

- **Create** `server/agent/translate_prompts.py` — loader for `config/prompts/translate.yaml` (hot-reload + fallback-to-None). Mirrors `server/agent/prompt_store.py` but for the translate prompt file.
- **Create** `config/prompts/translate.yaml` — externalized prompts (`llm_system`, `marketing_system`, `marketing_transcreate_user`, `json_nudge`).
- **Modify** `server/agent/translate.py` — add `_nllb_draft_batch`, `_build_transcreate_prompt`, `_qwen_transcreate_from_draft`, `_marketing_translate_lang`, a default transcreate template constant, and split the `marketing` branch in `translate_batch`.
- **Modify** `tests/test_translate.py` — add hybrid-marketing tests (the 4-cell matrix + multi-lang) and a loader test.

---

## Task 1: Translate prompt loader

**Files:**
- Create: `server/agent/translate_prompts.py`
- Test: `tests/test_translate.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_translate.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_translate.py::TestTranslatePrompts -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'server.agent.translate_prompts'`

- [ ] **Step 3: Write minimal implementation**

Create `server/agent/translate_prompts.py`:

```python
"""Translate prompt loader.

Loads externalized translate prompts from config/prompts/translate.yaml with
hot-reload. Keys absent from the file (or a missing file) return None so callers
fall back to the hardcoded defaults in server/agent/translate.py. Mirrors the
prompt_store.py pattern but for the stateless translate endpoint.
"""

from __future__ import annotations

import logging
import os
import threading

import yaml

logger = logging.getLogger("server.agent.translate_prompts")

DEFAULT_DIR = os.environ.get("PROMPT_CONFIG_DIR", "config/prompts")
_FILE = "translate.yaml"


class TranslatePrompts:
    def __init__(self, config_dir: str = DEFAULT_DIR):
        self._path = os.path.join(config_dir, _FILE)
        self._lock = threading.Lock()
        self._mtime: float = 0.0
        self._data: dict = {}
        self._load()

    def _load(self) -> None:
        try:
            mtime = os.path.getmtime(self._path)
        except OSError:
            self._data = {}
            return
        if mtime == self._mtime and self._data:
            return
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                self._data = yaml.safe_load(f) or {}
            self._mtime = mtime
            logger.info("TranslatePrompts loaded %s", self._path)
        except Exception as e:
            logger.error("TranslatePrompts failed to load %s: %s", self._path, e)

    def _reload_if_changed(self) -> None:
        try:
            if os.path.getmtime(self._path) != self._mtime:
                self._load()
        except OSError:
            pass

    def get(self, key: str) -> str | None:
        """Return the prompt for `key`, or None if file/key absent or blank."""
        with self._lock:
            self._reload_if_changed()
            val = (self._data.get("prompts") or {}).get(key)
            return val.strip() if isinstance(val, str) and val.strip() else None


_store: TranslatePrompts | None = None
_store_lock = threading.Lock()


def get_translate_prompts() -> TranslatePrompts:
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = TranslatePrompts()
    return _store


def reset_translate_prompts(config_dir: str = DEFAULT_DIR) -> TranslatePrompts:
    """Reset singleton (tests)."""
    global _store
    with _store_lock:
        _store = TranslatePrompts(config_dir)
        return _store
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_translate.py::TestTranslatePrompts -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add server/agent/translate_prompts.py tests/test_translate.py
git commit -m "feat(translate): add translate.yaml prompt loader with fallback"
```

---

## Task 2: Prompt config file

**Files:**
- Create: `config/prompts/translate.yaml`

This file is data, not logic — no test of its own (Task 1 already proves the loader,
Task 4 proves the transcreate prompt is used). Just create it and confirm the loader
reads it.

- [ ] **Step 1: Create the file**

Create `config/prompts/translate.yaml`:

```yaml
# Externalized translate prompts. Edited without a rebuild (hot-reloaded).
# If this file or a key is missing, server/agent/translate.py uses its hardcoded
# defaults — so deleting this never breaks translation.
#
# Placeholders in marketing_transcreate_user are replaced by str.replace (NOT
# str.format), so literal { } in the JSON example below are safe to keep as-is.
version: 1
prompts:
  llm_system: |
    You are a professional translator. Translate accurately and fluently into
    the target language, preserving meaning, proper nouns and numbers. Do not
    add or omit information.

  marketing_system: |
    You are an expert travel & marketing copywriter and translator.
    Transcreate the text into natural, evocative marketing copy in the target
    language. Keep ALL facts, numbers and proper nouns unchanged. Be creative
    but accurate — never invent details.

  marketing_transcreate_user: |
    The items below were machine-translated into {target_lang}. For each item you
    are given the ORIGINAL source text and the machine DRAFT.
    Rewrite each item as fluent, evocative marketing copy in {target_lang}:
    - Fix any awkwardness or errors in the draft using the original as the source of truth.
    - Keep ALL facts, numbers, prices, dates, proper nouns and glossary terms unchanged.
    - Never invent details that are not in the original.
    {context_line}
    {glossary_line}
    Return ONLY a JSON object mapping each item index (as a string) to its
    rewritten text, e.g. {"0":"...","1":"..."}. No prose, no code fences.
    Items:
    {items_block}

  json_nudge: |
    Output ONLY valid JSON like {"0":"..."} — nothing else.
```

- [ ] **Step 2: Verify the loader reads it**

Run: `python -c "from server.agent.translate_prompts import reset_translate_prompts; s=reset_translate_prompts(); print(bool(s.get('marketing_transcreate_user')), bool(s.get('marketing_system')))"`
Expected: `True True`

- [ ] **Step 3: Commit**

```bash
git add config/prompts/translate.yaml
git commit -m "feat(translate): add externalized translate.yaml prompts"
```

---

## Task 3: NLLB draft batch helper

**Files:**
- Modify: `server/agent/translate.py` (add helper after `_qwen_translate_lang`, ~line 100)
- Test: `tests/test_translate.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_translate.py` (inside a new class):

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_translate.py::TestNllbDraftBatch -v`
Expected: FAIL — `ImportError: cannot import name '_nllb_draft_batch'`

- [ ] **Step 3: Write minimal implementation**

In `server/agent/translate.py`, add after `_qwen_translate_lang` (after line 100, before `translate_batch`):

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_translate.py::TestNllbDraftBatch -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add server/agent/translate.py tests/test_translate.py
git commit -m "feat(translate): add _nllb_draft_batch helper"
```

---

## Task 4: Qwen transcreate-from-draft helper

**Files:**
- Modify: `server/agent/translate.py` (add default template constant near top + helpers)
- Test: `tests/test_translate.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_translate.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_translate.py::TestQwenTranscreate -v`
Expected: FAIL — `ImportError: cannot import name '_qwen_transcreate_from_draft'`

- [ ] **Step 3: Write minimal implementation**

In `server/agent/translate.py`, add the default template constant near the other
module constants (after line 30, below `LANG_NAMES`):

```python
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
    'rewritten text, e.g. {"0":"...","1":"..."}. No prose, no code fences.\n'
    "Items:\n{items_block}"
)
```

Then add the helpers after `_nllb_draft_batch`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_translate.py::TestQwenTranscreate -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add server/agent/translate.py tests/test_translate.py
git commit -m "feat(translate): add Qwen transcreate-from-draft helper + prompt builder"
```

---

## Task 5: Wire hybrid marketing into `translate_batch`

**Files:**
- Modify: `server/agent/translate.py` (add `_marketing_translate_lang`; split the `else` branch in `translate_batch`, lines 124-136)
- Test: `tests/test_translate.py` (append the 4-cell matrix + multi-lang)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_translate.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_translate.py::TestMarketingHybrid -v`
Expected: FAIL — current `marketing` goes straight to Qwen, so
`test_nllb_ok_qwen_parsefail_falls_back_to_draft` fails (gets a `parse_failed`
error instead of the NLLB draft).

- [ ] **Step 3: Write minimal implementation**

In `server/agent/translate.py`, add `_marketing_translate_lang` after
`_qwen_transcreate_from_draft`:

```python
async def _marketing_translate_lang(vllm_client, model, items, source_lang,
                                    target_lang, context, glossary) -> dict:
    """Hybrid NLLB->Qwen for one language. Returns {index_str: text}.

    NLLB ok  -> Qwen transcreates the draft; per item, Qwen wins, NLLB draft is
                the fallback floor (so a Qwen parse-fail never empties the item).
    NLLB down -> Qwen translates directly (old marketing behavior).
    """
    drafts, nllb_failed = await _nllb_draft_batch(items, source_lang, target_lang)
    if nllb_failed:
        return await _qwen_translate_lang(
            vllm_client, model, items, source_lang, target_lang, context, "marketing", glossary)

    try:
        polished = await _qwen_transcreate_from_draft(
            vllm_client, model, items, drafts, source_lang, target_lang, context, glossary)
    except Exception as e:
        logger.warning("qwen transcreate failed (lang=%s): %s", target_lang, e)
        polished = {}

    out: dict[str, str] = {}
    for idx in range(len(items)):
        val = polished.get(str(idx))
        out[str(idx)] = val if val else drafts.get(idx, "")
    return out
```

Then change the `else` branch of `translate_batch` (currently lines 124-136) to
split `marketing` from `llm`:

```python
        else:
            try:
                if style == "marketing":
                    per = await _marketing_translate_lang(
                        vllm_client, model, items, source_lang, lang, context, glossary)
                else:
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
```

(The merge loop is unchanged — it already does "value present → use it, else error".
The NLLB-draft fallback inside `_marketing_translate_lang` is what makes a Qwen
parse-fail resolve to the draft instead of an error.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_translate.py::TestMarketingHybrid -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add server/agent/translate.py tests/test_translate.py
git commit -m "feat(translate): hybrid marketing style (NLLB draft -> Qwen transcreate)"
```

---

## Task 6: Full regression + finish

**Files:** none (verification + branch finish)

- [ ] **Step 1: Run the whole translate test module**

Run: `python -m pytest tests/test_translate.py -v`
Expected: PASS — all new tests + the original `TestTranslateBatch` (including
`test_marketing_style_uses_qwen`) and `TestEndpoint`. Note: the original
`test_marketing_style_uses_qwen` passes `vllm` with no NLLB monkeypatch, so the
real `get_translation_client()` is hit — confirm it either (a) still passes because
that test's NLLB call fails fast and falls to Qwen-direct, or (b) update that test
to monkeypatch NLLB. If it breaks, add `self._nllb`-style monkeypatch to it.

- [ ] **Step 2: Run the full suite**

Run: `python -m pytest -q`
Expected: PASS (no regressions across the repo).

- [ ] **Step 3: Sanity-check the prompt fallback path**

Run: `python -c "import os; os.environ['PROMPT_CONFIG_DIR']='/nonexistent'; from server.agent.translate_prompts import reset_translate_prompts; print(reset_translate_prompts('/nonexistent').get('marketing_transcreate_user'))"`
Expected: `None` (loader returns None → code uses `_DEFAULT_TRANSCREATE_USER`).

- [ ] **Step 4: Finish the branch**

Use the `superpowers:finishing-a-development-branch` skill to decide merge/PR/cleanup.
```

---

## Self-Review

**Spec coverage:**
- §3 pipeline + §3.1 matrix → Task 5 (`_marketing_translate_lang` + the 5 matrix tests). ✓
- §4 Cách A (batch JSON + fallback) → Task 4 (`_qwen_transcreate_from_draft` batch call) + Task 5 merge. ✓
- §5 prompt YAML + §5.1 loader → Task 1 (loader) + Task 2 (yaml). ✓
- §6 code changes (helpers, split branch, llm/faithful/router untouched) → Tasks 3-5. ✓
- §7 tests (7 scenarios) → Task 1 (loader), Task 5 (matrix + multi-lang), Task 6 (llm regression). ✓
- §8 assumptions (short-code→FLORES, 60s budget, circuit breaker) → covered by reusing `get_translation_client` (which calls `to_flores`) and Task 6 regression. ✓

**Placeholder scan:** No TBD/TODO; every code step shows full code; commands have expected output. ✓

**Type/name consistency:** Helper names consistent across tasks — `_nllb_draft_batch` (Task 3, used in Task 5), `_qwen_transcreate_from_draft`/`_build_transcreate_prompt` (Task 4, used in Task 5), `_marketing_translate_lang` (Task 5), `_DEFAULT_TRANSCREATE_USER` (Task 4). Loader API `get()`/`reset_translate_prompts()`/`get_translate_prompts()` consistent (Task 1, used Task 4). `_nllb_draft_batch` returns `(dict[int,str], bool)` — consumed as `drafts.get(idx)` (int key) and `nllb_failed` in Task 5. Qwen helper returns `{index_str: text}` (string key) — consumed via `polished.get(str(idx))`. ✓
