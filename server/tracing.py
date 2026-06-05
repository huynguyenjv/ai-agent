"""OpenTelemetry Tracing — Phase 14.1.

Distributed tracing, fully optional: a no-op unless OTEL_EXPORTER_OTLP_ENDPOINT
is set AND the opentelemetry packages are installed. `span()` is always safe to
call so call sites never need to branch on whether tracing is enabled.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager

logger = logging.getLogger("server.tracing")

_enabled = False
_tracer = None


def setup_tracing(app) -> bool:
    """Initialize OTel + instrument FastAPI/httpx. Returns True if enabled."""
    global _enabled, _tracer

    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint:
        logger.info("OTel tracing disabled (OTEL_EXPORTER_OTLP_ENDPOINT not set).")
        return False

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        provider = TracerProvider(
            resource=Resource.create(
                {"service.name": os.environ.get("OTEL_SERVICE_NAME", "ai-agent")}
            )
        )
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
        trace.set_tracer_provider(provider)
        FastAPIInstrumentor.instrument_app(app)

        try:
            from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

            HTTPXClientInstrumentor().instrument()
        except Exception:
            pass  # httpx instrumentation is best-effort

        _tracer = trace.get_tracer("ai-agent")
        _enabled = True
        logger.info("OTel tracing enabled → %s", endpoint)
        return True
    except Exception as e:
        logger.warning("OTel setup failed (packages missing?): %s", e)
        _enabled = False
        return False


def is_enabled() -> bool:
    return _enabled


@contextmanager
def span(name: str, **attributes):
    """Custom span context manager. No-op (yields None) when tracing is off."""
    if not _enabled or _tracer is None:
        yield None
        return
    with _tracer.start_as_current_span(name) as s:
        for k, v in attributes.items():
            try:
                s.set_attribute(k, v)
            except Exception:
                pass
        yield s
