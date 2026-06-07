# translation-service (NLLB-200-3.3B + CTranslate2)

Encoder-decoder translation model served separately from vLLM. The ai-agent
calls it over HTTP (`TRANSLATION_URL`). Languages are dynamic (~200 FLORES codes
from the NLLB tokenizer). License: **NLLB weights are CC-BY-NC — internal use only.**

## 1. Convert the model (one-time, ~4GB output)
```bash
pip install ctranslate2 transformers
ct2-transformers-converter --model facebook/nllb-200-3.3B \
    --quantization int8 --output_dir /models/nllb-200-3.3B-ct2
```
- Server has internet → run on the server.
- No internet → convert on a machine with internet, then `rsync` the folder to the server.

## 2. Run (GPU, via docker-compose)
```bash
# build on the server; needs nvidia-container-toolkit
docker compose --profile translate up -d --build
```
Enable on the ai-agent side: `ENABLE_TRANSLATE=true`, `TRANSLATION_URL=http://translation-service:8100`.

## 3. Dev / CI without the model
```bash
TRANSLATE_MOCK=true MODEL_DIR=/none uvicorn server:app --port 8100
```
Mock returns a stub; `/languages` returns a tiny list.

## API
```
GET  /health      → {status, model_loaded, mock}
GET  /languages   → {languages: [...FLORES codes]}
POST /translate   {text, source, target(FLORES)} → {translation}
```

## Env
| Var | Default | |
|-----|---------|--|
| MODEL_DIR | /models/nllb-200-3.3B-ct2 | CTranslate2 model dir |
| DEVICE | cuda | cpu \| cuda |
| COMPUTE_TYPE | int8_float16 | CPU: use `int8` |
| TRANSLATE_MOCK | false | stub mode (no model) |
| BEAM_SIZE | 2 | |
| MAX_DECODING_LENGTH | 512 | |
