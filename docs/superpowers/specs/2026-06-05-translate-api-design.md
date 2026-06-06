# Translate API — Design Spec

**Date:** 2026-06-05
**Status:** Draft (pending review)
**Owner:** huynmb
**Stakeholders:** Nguyễn Khắc Tiệp (VSF-DL-KS-PMKD), Nguyễn Thạch Vũ (VSF-DL-KS-PMKD)

## 1. Problem & Scope

Backend (BE) lưu dữ liệu nghiệp vụ trong một **bảng translation riêng**. Khi có
nội dung mới (vd tiếng Việt) chưa có bản dịch, BE cần một dịch vụ AI để sinh bản
dịch sang ngôn ngữ đích (vd tiếng Anh) rồi BE tự ghi vào DB của họ.

Phạm vi của spec này **chỉ là endpoint AI translate** do service `ai-agent`
(FastAPI + vLLM) expose. Service **stateless** về phía dịch: không đụng DB, không
session, không quản lý bảng translation.

**Ngoài phạm vi (out of scope):** cơ chế kích hoạt dịch (DB trigger / CDC / BE
gọi trong luồng insert). Endpoint này dùng được cho mọi hướng kích hoạt; việc
chọn hướng do hai anh Tiệp/Vũ quyết định riêng. (Ghi chú: "nghe event ở bảng" là
địa hạt DB trigger/CDC, **không phải** vai của MCP.)

## 2. API Contract

### Endpoint
```
POST /v1/translate
Headers: X-Api-Key: <key>      # tái dùng verify_api_key hiện có
```

### Request
```jsonc
{
  "sourceLang": "vi",            // optional, mặc định "auto" (LLM tự nhận diện)
  "targetLangs": ["en", "ko"],   // 1..N ngôn ngữ đích
  "items": [
    { "ref": { "id": 123, "field": "name" }, "text": "Quản trị viên" },
    { "ref": { "id": 123, "field": "desc" }, "text": "Người quản lý hệ thống" }
  ],
  "context": "Tên & mô tả vai trò trong hệ thống IAM"  // optional, gợi ý domain
}
```

- `ref` là **object/giá trị tùy ý** do BE đặt. Server **không hiểu, không xử lý**,
  chỉ **echo nguyên si** trong response. BE nhét `id`+`field` (hoặc gì tùy ý) để
  map ngược vào DB row — không phải nối chuỗi key, không lo lệch thứ tự.
- `text` là plain text. (Giữ HTML/placeholder `{var}` → xem mục Future Work.)

### Response 200
```jsonc
{
  "results": [
    { "ref": { "id": 123, "field": "name" },
      "translations": { "en": "Administrator", "ko": "관리자" } },
    { "ref": { "id": 123, "field": "desc" },
      "translations": { "en": "System-wide manager", "ko": "..." } }
  ],
  "errors": [
    // các (item index, lang) không dịch được sau retry; phần còn lại vẫn trả
    // { "index": 0, "lang": "ko", "reason": "parse_failed" }
  ],
  "model": "qwen2.5-coder"
}
```

- `results` theo **đúng thứ tự** `items` của request; mỗi phần tử echo lại `ref`
  và map `lang -> translated_text`.
- `errors` rỗng khi mọi thứ thành công.

### Mã lỗi
| Tình huống | HTTP |
|---|---|
| Thiếu/sai API key | 401 |
| `items` rỗng / vượt giới hạn / tổng ký tự vượt ngưỡng | 422 |
| Quá thời gian xử lý (`asyncio.wait_for`) | 504 |
| vLLM lỗi | 502 |

## 3. Architecture

Theo khuôn `server/routers/review.py` (stateless, gọi thẳng vLLM, bỏ qua agent graph).

- **Router**: `server/routers/translate.py`
  - Pydantic models: `TranslateItem`, `TranslateRequest`, `TranslateResult`,
    `TranslateResponse`.
  - `verify_api_key` → validate giới hạn → `asyncio.wait_for(_run_translate, TIMEOUT)`.
  - Đăng ký trong `server/app.py` (`app.include_router(translate_router)`).
- **Module dịch**: `server/agent/translate.py`
  - `async def translate_batch(vllm_client, model, items, source_lang, target_langs, context) -> tuple[results, errors]`
  - Tách khỏi router để unit-test độc lập (giống `review_analyze`).

### Luồng xử lý
1. Router xác thực + validate giới hạn.
2. Với **mỗi `targetLang`**: build 1 prompt gom tất cả `items` (đánh số theo
   **index** nội bộ — `ref` không gửi cho model), nhiệt độ thấp, yêu cầu trả
   **JSON nghiêm ngặt** dạng `{ "0": "...", "1": "..." }` (index → bản dịch).
3. Parse JSON. Nếu lỗi format → **retry 1 lần** với prompt nhắc chặt hơn. Vẫn lỗi
   → đẩy các (index, lang) vào `errors`, **không làm hỏng cả batch**.
4. Gắn `ref` (theo index) vào kết quả; gộp các lang thành `translations` per item.

### Vì sao map nội bộ theo index, đối ngoại theo `ref`
- Model làm việc với danh sách đánh số (đơn giản, ít nhầm) → JSON gọn.
- Server giữ mảng `ref` song song theo index; sau khi parse thì zip lại. BE chỉ
  thấy `ref`, không thấy index.

## 4. Limits & Config (env)
| Env | Mặc định | Ý nghĩa |
|---|---|---|
| `TRANSLATE_MAX_ITEMS` | 50 | Số item tối đa/request |
| `TRANSLATE_MAX_CHARS` | 20000 | Tổng ký tự tối đa/request |
| `TRANSLATE_TIMEOUT_SECS` | 60 | Timeout toàn request |
| `TRANSLATE_TEMPERATURE` | 0.2 | Nhiệt độ gọi LLM |

Tái dùng rate limiter hiện có nếu áp được cho endpoint này.

## 5. Testing
- **Unit (`tests/test_translate.py`)** — mock vLLM:
  - happy path: 2 items × 2 langs → mapping đúng `ref` + `translations`.
  - JSON hỏng lần 1 → retry → lần 2 ok.
  - JSON hỏng cả 2 lần → item/lang đó vào `errors`, các phần khác vẫn trả.
  - `sourceLang` mặc định "auto".
- **Endpoint (FastAPI TestClient)**:
  - 200 happy path.
  - 422 khi `items` rỗng / vượt `TRANSLATE_MAX_ITEMS` / vượt `TRANSLATE_MAX_CHARS`.
  - 401 khi thiếu API key.

## 6. Future Work (YAGNI — không làm phase đầu)
- `mimeType` / tag-handling để giữ HTML & placeholder `{var}` (như Google/DeepL).
- `formality`, `glossary` (thuật ngữ cố định).
- Async/job cho batch lớn (hiện chỉ sync).
- Caching bản dịch (nếu cùng text lặp lại nhiều).

## 7. Decisions Log
- **Index → `ref` echo-back (Option B)** thay vì key chuỗi do BE tự đặt: tránh
  phiền/sai cho BE, không lo lệch thứ tự khi có nhiều target lang.
- **Sync** (không async/job) cho phase đầu: đơn giản, hợp record lẻ/nhỏ; giới hạn
  batch để tránh timeout.
- **Engine**: vLLM nội bộ (model hiện hành của service).
- **Multi-target** trong 1 request (giống Azure) thay vì 1 target/request.
