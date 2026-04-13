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
    load_dotenv()  # Must run before any os.environ.get

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
    )


if __name__ == "__main__":
    main()
