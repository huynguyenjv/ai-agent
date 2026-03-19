"""
Backward-compatible wrapper — re-exports from server.app.

This file ensures that ``server.api:app`` (used in main.py and uvicorn)
continues to work after the router split refactoring.
"""

from .app import app, create_app

__all__ = ["app", "create_app"]
