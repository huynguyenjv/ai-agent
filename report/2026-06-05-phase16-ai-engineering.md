# Phase 16 — AI Engineering Maturity

**Date:** 2026-06-05
**Branch:** `feature/new-architecture`
**Status:** ✅ COMPLETED (16.1–16.6) — 391 tests passing

---

## 1. Modules

| Mục | File | Nội dung | Wired? |
|-----|------|----------|--------|
| 16.4 Prompt Versioning | `config/prompts/intents.yaml`, `server/agent/prompt_store.py` | Prompt theo YAML: version, **variants**, **hot-reload** (mtime), **rollback**. Intent không có trong YAML → fallback `INTENT_PROMPTS`. | ✅ `generate._to_openai_messages` dùng store + variant, fallback an toàn |
| 16.2 A/B Testing | `server/experiment.py` | `ExperimentManager`: gán variant **deterministic** theo hash(unit), weights, track outcome per variant. | ✅ `chat.py` gán `experiment_variant` (theo conversation_id), `generate` chọn prompt variant. Bật bằng `ENABLE_AB_PROMPT` |
| 16.1 LLM-as-Judge | `eval/llm_judge.py` | Chấm response 4 chiều (correctness/completeness/quality/clarity/overall) qua judge model **inject được**; parse JSON chịu lỗi; `judge_batch`. | Offline (eval) |
| 16.5 Online Quality Metrics | `server/metrics/quality.py` | `QualityTracker`: satisfaction/task-completion/retry/code-acceptance rate + response length, theo intent. | Standalone (sẵn sàng nối feedback) |
| 16.3 Trace → Dataset | `eval/trace_collector.py` | Thu thập example +/- → JSONL versioned; **redact secret/PII** (dùng `secret_scanner`) trước khi ghi. | Offline (eval) |
| 16.6 Model Comparison | `eval/model_comparison.py` | Chạy cases trên nhiều model, so avg score/latency/tokens/cost, **recommend** best score-per-cost. invoke/score inject được. | Offline (eval) |

`AgentState` thêm `experiment_variant`.

## 2. Thiết kế đáng chú ý

- **An toàn/khả nghịch:** prompt versioning có **fallback** về prompt hardcode → YAML hỏng/thiếu intent vẫn chạy. A/B **mặc định off** (`ENABLE_AB_PROMPT=false`) → variant luôn "default", hành vi không đổi.
- **Testable:** judge & model-comparison nhận callable inject (không phụ thuộc vLLM thật) → test bằng fake.
- **PII-safe:** trace→dataset tái dùng `secret_scanner.redact` để không bao giờ ghi secret vào dataset.
- **Đo lường:** QualityTracker bổ sung tín hiệu *chất lượng output* bên cạnh perf metrics (latency/token) sẵn có.

## 3. Tests

`tests/test_phase16_ai_eng.py` (16): prompt store (3), experiment (4), judge (5), quality (1), trace collector (2), model comparison (1). 375 → **391 passed**. compileall exit 0.

## 4. Cách dùng nhanh

- **Đổi prompt không cần deploy:** sửa `config/prompts/intents.yaml` → hot-reload theo mtime.
- **Bật A/B prompt:** `ENABLE_AB_PROMPT=true` → 50/50 default vs concise cho `code_gen`; xem kết quả qua `ExperimentManager.results("code_gen_prompt")`.
- **Chấm chất lượng offline:** `judge_batch(cases, complete)` với `complete` gọi judge model.
- **Gom dataset fine-tune:** `TraceCollector.add_positive/negative(...)` → `export_jsonl()`.

## 5. Trace nhanh

```
config/prompts/intents.yaml          (mới) prompt versioned
server/agent/prompt_store.py         (mới) loader hot-reload/variant/rollback
server/experiment.py                 (mới) A/B manager
server/metrics/quality.py            (mới) online quality
eval/llm_judge.py                    (mới) LLM-as-judge
eval/trace_collector.py              (mới) trace→dataset (PII redact)
eval/model_comparison.py             (mới) model comparison
server/agent/generate.py             ~ dùng prompt_store + variant (fallback)
server/routers/chat.py               ~ gán experiment_variant
server/agent/state.py                + experiment_variant
tests/test_phase16_ai_eng.py         + 16 tests
```

---

*Report generated: 2026-06-05. Liên quan: Phase 9 (benchmark/feedback), improvement-plan Phase 16.*
