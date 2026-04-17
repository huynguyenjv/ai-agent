"""Structured JSON logging — Phase 7.

Outputs structured JSON log lines for production observability.
Includes correlation ID from request context when available.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from contextvars import ContextVar

# Context variable for correlation ID propagation across async boundaries
correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="-")


class StructuredJsonFormatter(logging.Formatter):
    """JSON log formatter for production use."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": correlation_id_var.get("-"),
        }
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry, ensure_ascii=False)


def configure_logging(log_level: str = "INFO", json_format: bool = False) -> None:
    """Configure application logging.

    Args:
        log_level: Logging level string (INFO, DEBUG, etc.)
        json_format: If True, output structured JSON. If False, human-readable.
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    if json_format:
        formatter = StructuredJsonFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s %(message)s"
        )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)
    # Remove existing handlers to avoid duplicate output
    root.handlers.clear()
    root.addHandler(handler)
