import pytest
from unittest.mock import AsyncMock, MagicMock
from langchain_core.messages import HumanMessage

from server.agent.generate import generate


class FakeChoice:
    def __init__(self, delta, finish_reason=None):
        self.delta = delta
        self.finish_reason = finish_reason


class FakeChunk:
    def __init__(self, delta, finish_reason=None):
        self.choices = [FakeChoice(delta, finish_reason)]


class FakeStream:
    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        self._iter = iter(self._chunks)
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration


@pytest.mark.asyncio
async def test_generate_streams_content():
    calls = []

    async def sse_cb(event_type, content):
        calls.append((event_type, content))

    d1 = MagicMock(content="Hello ", tool_calls=None)
    d2 = MagicMock(content="world", tool_calls=None)
    stream = FakeStream([FakeChunk(d1), FakeChunk(d2, finish_reason="stop")])

    vllm = MagicMock()
    vllm.chat.completions.create = AsyncMock(return_value=stream)

    state = {"messages": [HumanMessage(content="Hi")], "client_tools": []}
    result = await generate(state, vllm_client=vllm, model="m", sse_callback=sse_cb)

    assert result["draft"] == "Hello world"
    assert ("content", "Hello ") in calls
    assert ("content", "world") in calls


@pytest.mark.asyncio
async def test_generate_captures_tool_calls():
    tc_delta = [{"index": 0, "id": "call_1", "type": "function",
                 "function": {"name": "read_file", "arguments": '{"path":"/a"}'}}]
    d1 = MagicMock(content=None, tool_calls=tc_delta)
    stream = FakeStream([FakeChunk(d1, finish_reason="tool_calls")])

    vllm = MagicMock()
    vllm.chat.completions.create = AsyncMock(return_value=stream)

    state = {"messages": [HumanMessage(content="read /a")], "client_tools": []}
    result = await generate(state, vllm_client=vllm, model="m", sse_callback=None)

    assert result["pending_tool_calls"]
    assert result["pending_tool_calls"][0]["function"]["name"] == "read_file"


@pytest.mark.asyncio
async def test_generate_forwards_merged_tools_to_vllm():
    d1 = MagicMock(content="ok", tool_calls=None)
    stream = FakeStream([FakeChunk(d1, finish_reason="stop")])
    vllm = MagicMock()
    vllm.chat.completions.create = AsyncMock(return_value=stream)

    state = {
        "messages": [HumanMessage(content="hi")],
        "client_tools": [{"type": "function", "function": {"name": "foo", "parameters": {"type": "object", "properties": {}}}}],
    }
    await generate(state, vllm_client=vllm, model="m", sse_callback=None)

    call_kwargs = vllm.chat.completions.create.call_args.kwargs
    tool_names = {t["function"]["name"] for t in call_kwargs["tools"]}
    assert tool_names == {"read_file", "grep_code", "foo"}
    assert call_kwargs["stream"] is True
