# Translation API — Implementation Spec (NLLB-200-3.3B, multi-model)

> **Trạng thái:** SPEC để review. Code khi đã OK.
> **Mục tiêu:** thêm `/v1/translate` cho ai-agent, phục vụ bởi một service NLLB riêng (CTranslate2), không đụng vLLM. Nội bộ (license CC-BY-NC).

---

## 1. Kiến trúc

```
            ┌── /v1/chat/completions ─→ vLLM            (coding, decoder-only)
ai-agent ───┤   (multi-model router)
            └── /v1/translate ─────────→ translation-service  (NLLB, encoder-decoder)
                                          └─ CTranslate2 int8 + tokenizer
```

- **ai-agent** = client: map mã ngôn ngữ → FLORES, gọi HTTP, bọc circuit breaker. KHÔNG host model.
- **translation-service** = server riêng (như vLLM): FastAPI mỏng bọc CTranslate2. Chạy dưới docker-compose **profile `translate`**.
- Gate `ENABLE_TRANSLATE` (off mặc định) → app chính không kéo torch/CT2.

---

## 2. API contract

### Public (ai-agent)
```
POST /v1/translate            # auth X-Api-Key; 503 nếu ENABLE_TRANSLATE=false
  { "text": "Xin chào", "source_lang": "vi", "target_lang": "en" }
→ 200 { "translated": "Hello", "source_lang": "vi", "target_lang": "en", "model": "nllb-200-3.3B" }
  400  nếu ngôn ngữ không hỗ trợ
  422  nếu text rỗng / quá lớn (reuse validation)

GET /v1/translate/languages
→ { "languages": ["ar","de","en","fr","hi","id","ja","km","ko","lo","ms","pt","ru","th","vi","zh", ...] }
```

### Internal (ai-agent → translation-service)
```
POST /translate   { "text": "...", "source": "vie_Latn", "target": "eng_Latn" }
→ { "translation": "..." }
GET  /health → { "status": "ok", "model_loaded": true }
```

---

## 3. ai-agent side (3 file)

### 3.1 `server/translation.py`  (DYNAMIC — không hardcode danh sách)
- **Không** giữ bảng ~18 ngôn ngữ cứng. ai-agent chỉ là router mỏng:
  - Nhận **mã FLORES trực tiếp** (`vie_Latn`, `eng_Latn`, … toàn bộ ~200) → **pass-through**.
  - Một **alias map nhỏ tùy chọn** cho mã ngắn quen dùng (`vi`,`en`,`zh`→`zho_Hans`…) — chỉ là tiện ích, không giới hạn.
  - Mã không nhận diện được → **gửi nguyên cho service**, service mới là **source of truth** (validate theo tokenizer NLLB). Lang lạ → service trả 400.
- `to_flores(code)`: có `_` → giữ nguyên; có trong alias → map; còn lại → trả nguyên (để service validate).
- `enable_translate()`, `get_translation_client()`.
- `/v1/translate/languages` của ai-agent **proxy** `GET {TRANSLATION_URL}/languages` → danh sách **động từ model** (cache ngắn).

```python
ALIASES = {"vi":"vie_Latn","en":"eng_Latn","zh":"zho_Hans","zh-tw":"zho_Hant"}  # tiện ích, optional

def to_flores(code):
    if not code: raise ValueError("language code required")
    if "_" in code: return code                 # FLORES code → pass-through (dynamic, ~200)
    return ALIASES.get(code.lower(), code)       # alias hoặc để service tự validate

class TranslationClient:
    async def translate(self, text, source_lang, target_lang):
        src, tgt = to_flores(source_lang), to_flores(target_lang)
        cb = get_circuit_breaker("translation")
        async def _call():
            async with httpx.AsyncClient(timeout=TRANSLATION_TIMEOUT) as c:
                r = await c.post(f"{self._url}/translate",
                                 json={"text": text, "source": src, "target": tgt})
                r.raise_for_status()
                return r.json()["translation"]
        return await cb.call(_call)

    async def languages(self):                   # proxy danh sách động từ service
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(f"{self._url}/languages"); r.raise_for_status()
            return r.json()["languages"]
```

### 3.2 `server/routers/translate.py`
```python
@router.post("/v1/translate")
async def translate(body: TranslateRequest, req, x_api_key=Header(None), authorization=Header(None)):
    verify_api_key(req, x_api_key, authorization)
    if not enable_translate():
        raise HTTPException(503, "Translation disabled (ENABLE_TRANSLATE=false)")
    if not body.text or len(body.text) > MAX_TRANSLATE_CHARS:   # reuse validation idea
        raise HTTPException(422, "invalid text length")
    try:
        out = await get_translation_client().translate(body.text, body.source_lang, body.target_lang)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"translated": out, "source_lang": body.source_lang,
            "target_lang": body.target_lang, "model": "nllb-200-3.3B"}

@router.get("/v1/translate/languages")
async def languages(): return {"languages": supported_languages()}
```

### 3.3 `server/app.py`
- `app.include_router(translate_router)` (luôn include; endpoint tự trả 503 khi disabled).
- Env mới: `ENABLE_TRANSLATE`, `TRANSLATION_URL`, `TRANSLATION_TIMEOUT`, `MAX_TRANSLATE_CHARS`.

---

## 4. translation-service (service riêng)

```
translation-service/
├── server.py          # FastAPI + CTranslate2
├── requirements.txt   # ctranslate2, transformers, sentencepiece, fastapi, uvicorn
├── Dockerfile
└── README.md          # convert model + run
```

### 4.1 `server.py` (cốt lõi)
```python
import os, re
import ctranslate2, transformers
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_DIR = os.environ["MODEL_DIR"]            # /models/nllb-200-3.3B-ct2
DEVICE    = os.environ.get("DEVICE", "cuda")   # cpu | cuda
COMPUTE   = os.environ.get("COMPUTE_TYPE", "int8_float16")  # GPU: int8_float16; CPU: int8
MOCK      = os.environ.get("TRANSLATE_MOCK", "false").lower() == "true"

app = FastAPI()
_translator = None; _tokenizer = None; _langs: set[str] = set()

def _load():
    global _translator, _tokenizer, _langs
    if _translator is None and not MOCK:
        _translator = ctranslate2.Translator(MODEL_DIR, device=DEVICE,
                                             compute_type=COMPUTE)
        _tokenizer  = transformers.AutoTokenizer.from_pretrained("facebook/nllb-200-3.3B")
        # DYNAMIC: ~200 FLORES lang codes lấy thẳng từ tokenizer
        _langs = {t for t in _tokenizer.additional_special_tokens if "_" in t}

def _split_sentences(text):                    # NLLB ~ sentence-level (<512 tok)
    parts = re.split(r"(?<=[.!?。！？\n])\s+", text.strip())
    return [p for p in parts if p]

class Req(BaseModel): text: str; source: str; target: str

@app.get("/health")
def health(): return {"status": "ok", "model_loaded": _translator is not None or MOCK}

@app.get("/languages")                          # DYNAMIC: toàn bộ lang model hỗ trợ
def languages():
    _load()
    return {"languages": sorted(_langs) if _langs else []}

@app.post("/translate")
def translate(r: Req):
    if MOCK:
        return {"translation": f"[mock {r.source}->{r.target}] {r.text}"}
    _load()
    if _langs and (r.source not in _langs or r.target not in _langs):
        raise HTTPException(400, f"unsupported language: {r.source} / {r.target}")
    _tokenizer.src_lang = r.source
    outs = []
    for sent in _split_sentences(r.text):
        src_tokens = _tokenizer.convert_ids_to_tokens(_tokenizer.encode(sent))
        result = _translator.translate_batch([src_tokens],
                    target_prefix=[[r.target]], beam_size=2, max_decoding_length=512)
        hyp = result[0].hypotheses[0][1:]      # bỏ target lang token
        outs.append(_tokenizer.decode(_tokenizer.convert_tokens_to_ids(hyp)))
    return {"translation": " ".join(outs)}
```

> **MOCK mode** (`TRANSLATE_MOCK=true`): trả stub, không cần tải NLLB → dùng cho dev + test + CI.

### 4.2 `Dockerfile` (GPU / CUDA — đã chốt)
```dockerfile
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04
WORKDIR /app
RUN apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
COPY server.py .
ENV MODEL_DIR=/models/nllb-200-3.3B-ct2 DEVICE=cuda
EXPOSE 8100
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8100"]
```
> `ctranslate2` wheel có sẵn CUDA — chỉ cần base image CUDA runtime + cuDNN. `compute_type="int8_float16"` trên GPU (nhanh + nhẹ VRAM). Build **trên server** (`docker compose build`).

### 4.3 `requirements.txt`
```
ctranslate2>=4.3
transformers>=4.40
sentencepiece>=0.2
fastapi>=0.110
uvicorn[standard]>=0.29
```

---

## 5. docker-compose (profile `translate`)
```yaml
  translation-service:
    build: ./translation-service           # build TRÊN SERVER
    container_name: ai-agent-translation
    restart: unless-stopped
    profiles: ["translate"]
    environment:
      MODEL_DIR: /models/nllb-200-3.3B-ct2
      DEVICE: cuda
      COMPUTE_TYPE: int8_float16
      # TRANSLATE_MOCK: "true"   # bật để chạy không cần model
    volumes:
      - /models/nllb-200-3.3B-ct2:/models/nllb-200-3.3B-ct2:ro
    ports:
      - "127.0.0.1:8100:8100"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```
> Server cần **nvidia-container-toolkit** để Docker thấy GPU (vLLM chắc đã có sẵn).
ai-agent env thêm:
```yaml
      ENABLE_TRANSLATE: "true"
      TRANSLATION_URL: http://translation-service:8100
```

---

## 6. Chuẩn bị model (1 lần)
```bash
pip install ctranslate2 transformers
ct2-transformers-converter --model facebook/nllb-200-3.3B \
    --quantization int8 --output_dir /models/nllb-200-3.3B-ct2
# kết quả ~3.4GB → mount làm volume
```
- Server có internet → convert tại chỗ.
- Không có internet → convert ở máy có net rồi `rsync` thư mục ~4GB sang server.

---

## 7. Tests (không cần NLLB thật)
`tests/test_translation.py`:
- `to_flores` map đúng; mã FLORES giữ nguyên; ngôn ngữ lạ → `ValueError`.
- `/v1/translate` disabled (no `ENABLE_TRANSLATE`) → 503.
- `/v1/translate` enabled + **monkeypatch `get_translation_client`** (fake trả "Hello") → 200, body đúng.
- ngôn ngữ không hỗ trợ → 400.
- `/v1/translate/languages` → list.
- (service) test `_split_sentences` + MOCK translate trả stub.

→ CI chạy với `TRANSLATE_MOCK` / fake client, **không tải model**.

---

## 8. Resilience / vận hành (reuse hạ tầng có sẵn)
- **Circuit breaker `"translation"`** + timeout → translation-service down không kéo sập, hiện ở `/health/deep` (có thể thêm probe).
- **Auth + validation** dùng lại `auth.py` / giới hạn size.
- **Bật/tắt**: `ENABLE_TRANSLATE` + `docker compose --profile translate up`.

---

## 9. Phạm vi & thứ tự code (khi OK)
```
1. server/translation.py           (client + FLORES)        + test
2. server/routers/translate.py      (/v1/translate)          + test
3. server/app.py                    (include router + env)
4. translation-service/ (server.py, Dockerfile, requirements, README)
5. docker-compose.yml               (profile translate)
6. docs/DEVELOPMENT.md              (mục Translation)
7. commit xanh
```

## 10. Điểm chốt — ĐÃ XONG
- [x] **DEVICE: `cuda`** (GPU, `int8_float16`, ~4GB VRAM). Cần check VRAM còn trống (vLLM đang chiếm).
- [x] **Build image TRÊN SERVER** (`docker compose build`). Cần nvidia-container-toolkit.
- [x] **Ngôn ngữ: DYNAMIC** — toàn bộ ~200 FLORES từ model; ai-agent pass-through + alias nhỏ tùy chọn.
- [x] **API contract** (mục 2) — OK (đã thống nhất).

→ Spec đã chốt toàn bộ. Sẵn sàng code theo thứ tự mục 9.

---

*Spec generated for review. Implement sau khi sign-off.*
