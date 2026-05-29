"""Tests for classify_intent module."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from server.agent.classify_intent import (
    classify_intent,
    _python_classify,
    _extract_file_target,
    _detect_freshness_signal,
)
from server.agent.rules_loader import RulesLoader, reset_rules_loader


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def rules_loader():
    """Get the default rules loader."""
    return reset_rules_loader()


@pytest.fixture
def temp_rules_file():
    """Create a temporary rules file for testing."""
    content = """intents:
  - name: unit_test
    description: "Write tests"
    keywords_vi: ["viet test", "tao test"]
    keywords_en: ["write test", "test case"]
    priority: 1
  - name: code_gen
    description: "Default"
    keywords_vi: []
    keywords_en: []
    priority: 99

freshness_keywords:
  - "vua sua"
  - "just changed"

file_mention_suffixes:
  - Service
  - Controller

file_extensions:
  - .java
  - .py

confidence_threshold: 0.7
rules_reload_interval_seconds: 1
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as f:
        f.write(content)
        f.flush()
        yield Path(f.name)


# =============================================================================
# Test Group 1: Turn 2 Detection
# =============================================================================

@pytest.mark.asyncio
async def test_turn2_preserves_intent():
    """Turn 2 should preserve intent from previous turn."""
    state = {
        "messages": [
            HumanMessage(content="viết test cho X"),
            ToolMessage(content="{}", tool_call_id="c1"),
        ],
        "intent": "unit_test",
    }

    result = await classify_intent(state)

    assert result["is_tool_result_turn"] is True
    assert result["intent"] == "unit_test"


@pytest.mark.asyncio
async def test_turn2_detected_from_dict_message():
    """Turn 2 should be detected from dict messages."""
    state = {
        "messages": [
            {"role": "user", "content": "test"},
            {"role": "tool", "content": "{}"},
        ],
    }

    result = await classify_intent(state)

    assert result["is_tool_result_turn"] is True


# =============================================================================
# Test Group 2: LLM Path (Mock vllm_client)
# =============================================================================

@pytest.mark.asyncio
async def test_llm_result_used_when_high_confidence():
    """LLM result should be used when confidence is high."""
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = '{"intent": "unit_test", "confidence": 0.95, "file_target": null, "freshness_signal": false, "reasoning": "test"}'
    mock_client.chat.completions.create.return_value = mock_response

    state = {
        "messages": [HumanMessage(content="viết test cho UserService")],
    }

    result = await classify_intent(state, vllm_client=mock_client, model="test-model")

    assert result["intent"] == "unit_test"
    assert result["confidence"] == 0.95
    assert result["is_tool_result_turn"] is False


@pytest.mark.asyncio
async def test_fallback_when_llm_confidence_low():
    """Should use Python fallback when LLM confidence is low."""
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = '{"intent": "code_gen", "confidence": 0.5, "file_target": null, "freshness_signal": false, "reasoning": "unsure"}'
    mock_client.chat.completions.create.return_value = mock_response

    state = {
        "messages": [HumanMessage(content="viết test cho UserService")],
    }

    result = await classify_intent(state, vllm_client=mock_client, model="test-model")

    # Python fallback should match "viết test" keyword
    assert result["intent"] == "unit_test"
    assert result["confidence"] == 0.6  # Python fallback confidence


@pytest.mark.asyncio
async def test_fallback_when_llm_raises():
    """Should use Python fallback when LLM raises exception."""
    mock_client = AsyncMock()
    mock_client.chat.completions.create.side_effect = Exception("timeout")

    state = {
        "messages": [HumanMessage(content="phân tích cấu trúc project")],
    }

    result = await classify_intent(state, vllm_client=mock_client, model="test-model")

    assert result["intent"] == "structural_analysis"
    assert "Python fallback" in result["reasoning"]


@pytest.mark.asyncio
async def test_fallback_when_vllm_client_none():
    """Should use Python fallback when vllm_client is None."""
    state = {
        "messages": [HumanMessage(content="viết test cho UserService.java")],
    }

    result = await classify_intent(state, vllm_client=None)

    assert result["intent"] == "unit_test"
    assert result["file_target"] == "UserService.java"


# =============================================================================
# Test Group 3: Python Fallback — Intent Detection
# =============================================================================

def test_structural_intent(rules_loader):
    """Should detect structural analysis intent."""
    result = _python_classify("phân tích cấu trúc project này", rules_loader)

    assert result["intent"] == "structural_analysis"
    assert result["confidence"] == 0.6


def test_unit_test_with_file(rules_loader):
    """Should detect unit test intent with file target."""
    result = _python_classify("viết unit test cho UserService.java", rules_loader)

    assert result["intent"] == "unit_test"
    assert result["file_target"] == "UserService.java"


def test_refine_with_freshness(rules_loader):
    """Should detect refine intent with freshness signal."""
    result = _python_classify(
        "vừa sửa xong OrderService, refactor lại giúp tôi",
        rules_loader,
    )

    assert result["intent"] == "refine"
    assert result["file_target"] == "OrderService"
    assert result["freshness_signal"] is True


def test_explain_no_file(rules_loader):
    """Should detect explain intent without file target."""
    result = _python_classify("giải thích luồng xử lý payment", rules_loader)

    assert result["intent"] == "explain"
    assert result["file_target"] is None


def test_search_intent(rules_loader):
    """Should detect search intent with file target."""
    result = _python_classify("tìm tất cả nơi dùng PaymentService", rules_loader)

    assert result["intent"] == "search"
    assert result["file_target"] == "PaymentService"


def test_fallback_to_code_gen(rules_loader):
    """Should fallback to code_gen for unknown queries."""
    result = _python_classify("hello", rules_loader)

    assert result["intent"] == "code_gen"
    assert result["confidence"] <= 0.5


# =============================================================================
# Test Group 4: File Target Detection
# =============================================================================

def test_extract_file_with_extension(rules_loader):
    """Should extract filename with extension."""
    result = _extract_file_target("đọc file OrderService.java", rules_loader)
    assert result == "OrderService.java"


def test_extract_file_at_mention(rules_loader):
    """Should extract @mention."""
    result = _extract_file_target("review @UserController", rules_loader)
    assert result == "UserController"


def test_extract_class_with_suffix(rules_loader):
    """Should extract class name with known suffix."""
    result = _extract_file_target("tìm PaymentService", rules_loader)
    assert result == "PaymentService"


def test_extract_deictic_with_active_file(rules_loader):
    """Should use active_file for deictic reference."""
    result = _extract_file_target("giải thích file này", rules_loader, active_file="Main.java")
    assert result == "Main.java"


def test_extract_deictic_without_active_file(rules_loader):
    """Should return None for deictic without active_file."""
    result = _extract_file_target("giải thích file này", rules_loader, active_file=None)
    assert result is None


# =============================================================================
# Test Group 5: Freshness Detection
# =============================================================================

def test_freshness_detected_vi(rules_loader):
    """Should detect Vietnamese freshness keywords."""
    assert _detect_freshness_signal("vừa sửa xong code", rules_loader) is True


def test_freshness_detected_en(rules_loader):
    """Should detect English freshness keywords."""
    assert _detect_freshness_signal("just changed the config", rules_loader) is True


def test_freshness_not_detected(rules_loader):
    """Should not detect freshness when no keywords."""
    assert _detect_freshness_signal("review this code", rules_loader) is False


# =============================================================================
# Test Group 6: Hot-Reload Rules
# =============================================================================

def test_rules_loader_returns_valid_structure(rules_loader):
    """RulesLoader should return valid structure."""
    intents = rules_loader.get_intents()

    assert isinstance(intents, list)
    assert len(intents) > 0

    for intent in intents:
        assert "name" in intent
        assert "priority" in intent


def test_rules_loader_custom_path(temp_rules_file):
    """RulesLoader should load from custom path."""
    loader = RulesLoader(temp_rules_file)
    intents = loader.get_intents()

    assert len(intents) == 2
    assert intents[0]["name"] == "unit_test"  # Priority 1


def test_rules_loader_keeps_old_on_invalid_yaml(temp_rules_file):
    """RulesLoader should keep old rules on invalid YAML."""
    loader = RulesLoader(temp_rules_file)
    old_intents = loader.get_intents()

    # Corrupt the file
    with open(temp_rules_file, "w") as f:
        f.write("invalid: yaml: content: [")

    # Force reload check
    loader._last_check = 0
    new_intents = loader.get_intents()

    # Should keep old rules
    assert new_intents == old_intents


# =============================================================================
# Test Group 7: Return Shape Contract
# =============================================================================

@pytest.mark.asyncio
async def test_all_fields_always_present():
    """All required fields should always be present."""
    queries = [
        "viết test cho UserService",
        "phân tích cấu trúc project",
        "tìm PaymentGateway",
        "giải thích code này",
        "hello world",
    ]

    required_fields = [
        "intent",
        "sub_intent",
        "file_target",
        "freshness_signal",
        "is_tool_result_turn",
        "confidence",
        "reasoning",
    ]

    for query in queries:
        state = {"messages": [HumanMessage(content=query)]}
        result = await classify_intent(state, vllm_client=None)

        for field in required_fields:
            assert field in result, f"Missing field '{field}' for query: {query}"
