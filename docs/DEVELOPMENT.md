# Development Guide

Hướng dẫn chạy & phát triển AI Coding Agent ở local. Xem `docs/architecture.md`
cho kiến trúc tổng thể.

## 1. Cài đặt

```bash
pip install -r requirements.txt
```

## 2. Local dev mode (không cần vLLM thật)

Đặt `DEV_MODE=true` → app dùng **mock vLLM client** (`server/dev_mode.py`), trả
lời stub. RAG đã opt-in (off) nên local run không cần Qdrant/embedder.

```bash
DEV_MODE=true python main.py
# hoặc test nhanh qua CLI:
DEV_MODE=true python main.py &        # chạy server
python cli.py config                  # kiểm tra cấu hình + kết nối
python cli.py chat "write hello world" -k <API_KEY>
```

> Production: bỏ `DEV_MODE`, đặt `VLLM_BASE_URL` trỏ tới vLLM thật.

## 3. CLI (`cli.py`)

| Lệnh | Mô tả |
|------|-------|
| `health [--deep]` | Health check (`--deep` probe vLLM/Qdrant/Redis/Postgres) |
| `config` | Validate cấu hình CLI + server reachable |
| `chat "<msg>" [--agents]` | Gửi chat; `--agents` bật multi-agent (xem `docs/multi-agent-design.md`) |
| `review --diff-file <f>` | Code review một diff |
| `index --repo <path>` | Index repo (chỉ dùng khi bật RAG) |

## 4. Biến môi trường chính

| Biến | Mặc định | Ý nghĩa |
|------|----------|---------|
| `DEV_MODE` | `false` | Mock vLLM, không cần model server |
| `VLLM_BASE_URL` | `http://localhost:8000/v1` | Endpoint vLLM |
| `VLLM_MODEL` | `qwen2.5-coder` | Model id |
| `ENABLE_RAG` | `false` | Bật RAG/Qdrant (agentic-first → off) |
| `API_KEY` | (empty) | Khóa API (bắt buộc set để auth pass) |
| `REDIS_URL` | (unset) | Bật Redis session+rate-limit (multi-instance) |
| `DATABASE_URL` | (unset) | Postgres cho metrics/audit (else SQLite) |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | (unset) | Bật OTel tracing → Jaeger |
| `ENABLE_AB_PROMPT` | `false` | Bật A/B prompt variant |
| `SPECULATIVE_PREWARM` | `false` | Pre-warm vLLM lúc khởi động |

## 5. Test

```bash
python -m pytest                       # toàn bộ
python -m pytest tests/integration     # e2e (dùng DEV_MODE mock vLLM)
python -m pytest tests/test_phase11_reliability.py -v
```

- Unit tests: `tests/test_*.py`.
- Integration (Phase 20.3): `tests/integration/` — chạy full graph qua FastAPI
  app với mock vLLM, không cần dịch vụ ngoài.

## 6. Docker / profiles

```bash
docker compose up                          # app + postgres + prometheus + grafana
docker compose --profile rag up            # + qdrant (khi bật RAG)
docker compose --profile scale up          # + redis (multi-instance)
docker compose --profile observability up  # + jaeger (OTel tracing)
```

## 6b. Translation API (NLLB-200-3.3B, opt-in)

Multi-model: coding → vLLM, translation → a separate `translation-service`
(NLLB + CTranslate2, GPU). See `docs/translation-api-implementation.md` and
`translation-service/README.md`.

```bash
# 1. one-time: convert NLLB → CTranslate2 int8 (~4GB) into /models/nllb-200-3.3B-ct2
# 2. start the service (build on server; needs nvidia-container-toolkit):
docker compose --profile translate up -d --build
# 3. enable on ai-agent: ENABLE_TRANSLATE=true, TRANSLATION_URL=http://translation-service:8100
```
API: `POST /v1/translate {text, source_lang, target_lang}` · `GET /v1/translate/languages`
(languages are dynamic — all ~200 NLLB FLORES codes). Dev/CI: `TRANSLATE_MOCK=true`.
License: NLLB is CC-BY-NC — internal use only.

## 7. Troubleshooting

| Triệu chứng | Nguyên nhân / cách xử lý |
|-------------|--------------------------|
| `403 Invalid API key` | Chưa set `API_KEY` env hoặc header `X-Api-Key` sai |
| Chat treo / lỗi LLM | vLLM không chạy → dùng `DEV_MODE=true`, hoặc kiểm tra `/health/deep` |
| `422` khi gọi chat | Payload vi phạm validation (rỗng / quá lớn / path traversal) |
| RAG không trả context | RAG off mặc định — đặt `ENABLE_RAG=true` + `--profile rag` |
| Khởi động chậm | `ENABLE_RAG=true` load embedder (torch). Để off nếu không cần. |
| Rate-limit lệch khi scale | Set `REDIS_URL` để chia sẻ giữa instance |

## 8. Cấu trúc thư mục
Xem `docs/architecture.md` mục "Cấu trúc thư mục".
