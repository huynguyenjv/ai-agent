"""Local Development Mode — Phase 20.2.

When DEV_MODE=true the app uses a mock vLLM client so you can run / test the
full pipeline without a real model server. RAG is already opt-in (off), so a
local run needs nothing but this. The mock mimics the AsyncOpenAI streaming +
non-streaming chat interface used by the agent.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger("server.dev_mode")

MOCK_REPLY = "[dev-mode mock response] This is a stubbed answer from the mock vLLM client."


def is_dev_mode() -> bool:
    return os.environ.get("DEV_MODE", "false").lower() in ("1", "true", "yes")


# --- minimal shapes matching the OpenAI streaming/non-streaming responses ---

class _Delta:
    def __init__(self, content=None):
        self.content = content
        self.tool_calls = None


class _Choice:
    def __init__(self, *, delta=None, message=None):
        self.delta = delta
        self.message = message


class _Message:
    def __init__(self, content):
        self.content = content
        self.tool_calls = None


class _Chunk:
    def __init__(self, choices):
        self.choices = choices


class _Response:
    def __init__(self, content):
        self.choices = [_Choice(message=_Message(content))]


class _MockStream:
    """Async iterator yielding the mock reply token by token."""

    def __init__(self, text: str):
        self._tokens = [t + " " for t in text.split(" ")]

    def __aiter__(self):
        self._it = iter(self._tokens)
        return self

    async def __anext__(self):
        try:
            tok = next(self._it)
        except StopIteration:
            raise StopAsyncIteration
        return _Chunk([_Choice(delta=_Delta(content=tok))])


class _MockCompletions:
    async def create(self, *, stream: bool = False, **kwargs):
        if stream:
            return _MockStream(MOCK_REPLY)
        return _Response(MOCK_REPLY)


class _MockChat:
    def __init__(self):
        self.completions = _MockCompletions()


class _MockModels:
    async def list(self):
        return {"data": [{"id": "mock-model"}]}


class MockVLLMClient:
    """Drop-in stand-in for AsyncOpenAI used by the agent in DEV_MODE."""

    def __init__(self):
        self.chat = _MockChat()
        self.models = _MockModels()


def build_mock_vllm_client() -> MockVLLMClient:
    logger.warning("DEV_MODE: using mock vLLM client (no real model server).")
    return MockVLLMClient()
