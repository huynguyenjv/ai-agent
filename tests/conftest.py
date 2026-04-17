"""Shared pytest config: asyncio mode + env defaults for tests."""
import os

os.environ.setdefault("API_KEY", "test-key")
os.environ.setdefault("GITLAB_URL", "https://gitlab.test")
os.environ.setdefault("GITLAB_TOKEN", "test-token")
os.environ.setdefault("VLLM_BASE_URL", "http://test/v1")
os.environ.setdefault("VLLM_MODEL", "test-model")
os.environ.setdefault("QDRANT_URL", "http://test:6333")

import pytest


@pytest.fixture
def anyio_backend():
    return "asyncio"
