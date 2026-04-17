import pytest
from unittest.mock import MagicMock
from langchain_core.messages import HumanMessage

from server.agent import graph as graph_mod
from server.agent.graph import build_agent_graph


@pytest.fixture
def fake_services():
    return MagicMock(), MagicMock(), MagicMock()


@pytest.mark.asyncio
async def test_graph_rag_off_code_gen_goes_direct_to_generate(fake_services, monkeypatch):
    vllm, qdrant, embedder = fake_services
    calls = []

    async def fake_generate(state, **kw):
        calls.append("generate")
        return {"draft": "ok", "pending_tool_calls": []}

    async def fake_classify(state):
        return {"intent": "code_gen", "is_tool_result_turn": False}

    async def fake_route_ctx(state):
        return {"volatile_rejected": False, "mentioned_files": []}

    async def fake_post_process(state):
        return {}

    monkeypatch.setattr(graph_mod, "generate", fake_generate)
    monkeypatch.setattr(graph_mod, "classify_intent", fake_classify)
    monkeypatch.setattr(graph_mod, "route_context", fake_route_ctx)
    monkeypatch.setattr(graph_mod, "post_process", fake_post_process)

    graph = build_agent_graph(vllm, "m", qdrant, embedder, enable_rag=False)
    result = await graph.ainvoke({"messages": [HumanMessage(content="gen a function")]})

    assert "generate" in calls
    assert result["draft"] == "ok"


@pytest.mark.asyncio
async def test_graph_tool_result_turn_bypasses_route_context(fake_services, monkeypatch):
    vllm, qdrant, embedder = fake_services
    seen = []

    async def fake_classify(state):
        seen.append("classify")
        return {"intent": "explain", "is_tool_result_turn": True}

    async def fake_route_ctx(state):
        seen.append("route_context")
        return {}

    async def fake_generate(state, **kw):
        seen.append("generate")
        return {"draft": "x", "pending_tool_calls": []}

    async def fake_post_process(state):
        return {}

    monkeypatch.setattr(graph_mod, "classify_intent", fake_classify)
    monkeypatch.setattr(graph_mod, "route_context", fake_route_ctx)
    monkeypatch.setattr(graph_mod, "generate", fake_generate)
    monkeypatch.setattr(graph_mod, "post_process", fake_post_process)

    graph = build_agent_graph(vllm, "m", qdrant, embedder, enable_rag=False)
    await graph.ainvoke({"messages": [HumanMessage(content="tool result")]})

    assert "route_context" not in seen
    assert "generate" in seen
