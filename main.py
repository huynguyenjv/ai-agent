<<<<<<< HEAD
"""
Main entry point for the AI Coding Agent.
"""

import os
import sys

# =============================================================================
# ENVIRONMENT CONFIGURATION
# =============================================================================

# Sentence transformers / ONNX model path
os.environ['SENTENCE_TRANSFORMERS_HOME'] = os.getenv('SENTENCE_TRANSFORMERS_HOME', './models')

# HuggingFace telemetry
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'

# Configure SSL settings (if needed)
from utils.ssl_utils import ssl_config

# =============================================================================

import structlog
import logging
import uvicorn
from dotenv import load_dotenv

# Set root logger level from env (defaults to INFO — silences DEBUG from httpx, urllib3, etc.)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(), format="%(message)s")
# Quiet noisy third-party loggers
for _name in ("httpx", "httpcore", "urllib3", "onnxruntime"):
    logging.getLogger(_name).setLevel(logging.WARNING)

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.dev.ConsoleRenderer() if sys.stderr.isatty() else structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


def main():
    """Run the AI Agent server."""
    load_dotenv()

    host = os.getenv("SERVER_HOST", "0.0.0.0")
    port = int(os.getenv("SERVER_PORT", "8080"))
    log_level = os.getenv("LOG_LEVEL", "INFO").lower()

    logger.info(
        "Starting AI Coding Agent",
        host=host,
        port=port,
        log_level=log_level,
    )

    uvicorn.run(
        "server.api:app",
        host=host,
        port=port,
        log_level=log_level,
        reload=os.getenv("DEBUG", "false").lower() == "true",
=======
"""Entry point for the new AI Coding Agent architecture.

Section 17: Configuration via environment variables.
"""

from __future__ import annotations

import logging
import os
import sys

import uvicorn
from dotenv import load_dotenv

from server.logging_config import configure_logging


def main() -> None:
    load_dotenv(override=True)  # Must run before any os.environ.get

    # Configure logging — use JSON in production (LOG_FORMAT=json)
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    json_format = os.environ.get("LOG_FORMAT", "").lower() == "json"
    configure_logging(log_level=log_level, json_format=json_format)

    host = os.environ.get("HOST", "0.0.0.0")
    try:
        port = int(os.environ.get("PORT", "8000"))
    except ValueError:
        logging.getLogger("main").error("Invalid PORT value, defaulting to 8000")
        port = 8000

    # Import app AFTER load_dotenv so lifespan reads correct env vars
    from server.app import app  # noqa: E402

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=log_level.lower(),
>>>>>>> feature/new-architecture
    )


if __name__ == "__main__":
    main()
<<<<<<< HEAD

=======
>>>>>>> feature/new-architecture
