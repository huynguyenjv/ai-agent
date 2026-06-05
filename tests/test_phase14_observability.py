"""Phase 14 — OTel tracing (gated), correlation middleware, structured logging, SLO dashboard."""

from __future__ import annotations

import json
import logging


# --------------------------------------------------------------------------- #
# 14.1 OTel tracing (gated / no-op by default)
# --------------------------------------------------------------------------- #
class TestTracing:
    def test_disabled_without_endpoint(self, monkeypatch):
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
        from fastapi import FastAPI
        from server.tracing import setup_tracing, is_enabled

        assert setup_tracing(FastAPI()) is False
        assert is_enabled() is False

    def test_span_is_safe_noop(self):
        from server.tracing import span

        with span("unit-test-span", foo="bar") as s:
            assert s is None  # no-op when tracing is off


# --------------------------------------------------------------------------- #
# 14.3 Correlation middleware
# --------------------------------------------------------------------------- #
class TestCorrelation:
    def _client(self, monkeypatch):
        monkeypatch.delenv("ENABLE_RAG", raising=False)
        from fastapi.testclient import TestClient
        from server.app import create_app

        return TestClient(create_app())

    def test_response_has_correlation_header(self, monkeypatch):
        with self._client(monkeypatch) as c:
            r = c.get("/health/live")
            assert "X-Correlation-ID" in r.headers

    def test_incoming_correlation_is_echoed(self, monkeypatch):
        with self._client(monkeypatch) as c:
            r = c.get("/health/live", headers={"X-Correlation-ID": "trace-abc123"})
            assert r.headers["X-Correlation-ID"] == "trace-abc123"


# --------------------------------------------------------------------------- #
# 14.6 Structured logging
# --------------------------------------------------------------------------- #
class TestStructuredLogging:
    def test_json_formatter_fields(self):
        from server.logging_config import StructuredJsonFormatter

        rec = logging.LogRecord("mylogger", logging.INFO, "path", 1, "hello", None, None)
        out = json.loads(StructuredJsonFormatter().format(rec))
        assert out["message"] == "hello"
        assert out["service"]
        assert "correlation_id" in out
        assert out["level"] == "INFO"


# --------------------------------------------------------------------------- #
# 14.5 SLO dashboard
# --------------------------------------------------------------------------- #
class TestSloDashboard:
    def test_dashboard_is_valid(self):
        with open("deploy/grafana/dashboards/slo.json", encoding="utf-8") as f:
            d = json.load(f)
        assert d["uid"] == "ai-agent-slo"
        assert len(d["panels"]) >= 3
        # every panel target has a PromQL expr
        for p in d["panels"]:
            for t in p.get("targets", []):
                assert t.get("expr")
