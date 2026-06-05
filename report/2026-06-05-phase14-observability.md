# Phase 14 — Observability

**Date:** 2026-06-05
**Branch:** `feature/new-architecture`
**Status:** ✅ COMPLETED (chosen scope) — 410 tests passing
**Scope:** 14.1 OTel tracing · 14.2 Jaeger · 14.3 Correlation middleware · 14.5 SLO dashboard · 14.6 logging polish

> 14.4 Sentry **bỏ** (SaaS ngoài). Prometheus + Grafana đã có từ Phase 3.

---

## 1. 14.1 OpenTelemetry Tracing — `server/tracing.py`

- `setup_tracing(app)` — khởi tạo OTel TracerProvider + OTLP exporter + instrument FastAPI/httpx. **Gated**: no-op nếu `OTEL_EXPORTER_OTLP_ENDPOINT` chưa set **hoặc** lib OTel chưa cài. Trả True/False.
- `span(name, **attrs)` — context manager **luôn an toàn** (yield None khi tracing off) → call site không cần branch.
- **Wire:** `app.py` gọi `setup_tracing(app)`; `chat.py` bọc `agent.ainvoke` trong `span("agent.invoke")`.
- Deps OTel để **optional/comment** trong `requirements.txt` (không ép cài).

## 2. 14.2 Jaeger — `docker-compose.yml`

- Service `jaeger` (all-in-one) dưới `profiles: ["observability"]`, OTLP gRPC `4317` + UI `16686`.
- Bật: `docker compose --profile observability up` + set `OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:4317`.

## 3. 14.3 Correlation Middleware — `server/middleware/correlation.py`

- `register_correlation_middleware(app)`: **honor incoming `X-Correlation-ID`** (id sống xuyên service) hoặc tự sinh; set contextvar (vào mọi log) + echo header response + log request timing.
- `app.py` thay middleware inline cũ bằng module này (chuẩn hoá, thêm propagate header vào).

## 4. 14.5 SLO Dashboard — `deploy/grafana/dashboards/slo.json`

- Panels: **Availability** (SLO 99.9%), **Error rate** (<1%), **Latency P99** (<10s) + P50/P95/P99 timeseries + error-budget burn. Dùng metric `ai_agent_*` (Prometheus) sẵn có.

## 5. 14.6 Structured Logging — `server/logging_config.py`

- JSON log thêm field `service` (từ `OTEL_SERVICE_NAME`). (Correlation-id + JSON đã có sẵn từ Phase 7.)

---

## 6. Tests

`tests/test_phase14_observability.py` (6): tracing disabled/no-op, correlation header (sinh + echo incoming), JSON formatter fields, SLO dashboard valid. 404 → **410 passed**. compileall exit 0.

## 7. Trace nhanh
```
server/tracing.py                       (mới) OTel gated + span()
server/middleware/correlation.py        (mới) correlation-id middleware
deploy/grafana/dashboards/slo.json      (mới) SLO dashboard
server/app.py                           ~ register correlation + setup_tracing
server/routers/chat.py                  ~ span("agent.invoke")
server/logging_config.py                ~ + service field
docker-compose.yml                      ~ jaeger (profile observability) + OTEL env
requirements.txt                        ~ OTel deps (optional, commented)
tests/test_phase14_observability.py     + 6 tests
```

---

*Report generated: 2026-06-05. Liên quan: Phase 3 (Prometheus/Grafana), improvement-plan Phase 14.*
