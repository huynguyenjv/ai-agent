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
    )


if __name__ == "__main__":
    main()

