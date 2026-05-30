"""Tests for source verification and hallucination mitigation."""

import pytest

from server.agent.verify_sources import (
    extract_code_blocks,
    compute_similarity,
    find_best_match,
    verify_rag_sources,
    add_citations,
    check_hallucination_risk,
)


class TestExtractCodeBlocks:
    """Tests for code block extraction."""

    def test_single_block(self):
        text = "Here is code:\n```python\ndef foo():\n    return 42\n```"
        blocks = extract_code_blocks(text)
        assert len(blocks) == 1
        assert "def foo" in blocks[0]

    def test_multiple_blocks(self):
        text = "```\ndef function_one():\n    return 'first block code'\n```\ntext\n```\ndef function_two():\n    return 'second block code'\n```"
        blocks = extract_code_blocks(text)
        assert len(blocks) == 2

    def test_no_blocks(self):
        text = "Just regular text without code"
        blocks = extract_code_blocks(text)
        assert blocks == []

    def test_filters_short_blocks(self):
        text = "```\nshort\n```"
        blocks = extract_code_blocks(text)
        assert blocks == []


class TestComputeSimilarity:
    """Tests for similarity computation."""

    def test_identical_text(self):
        assert compute_similarity("hello world", "hello world") == 1.0

    def test_similar_text(self):
        score = compute_similarity("hello world", "hello there world")
        assert 0.5 < score < 1.0

    def test_different_text(self):
        score = compute_similarity("hello", "goodbye")
        assert score < 0.5

    def test_empty_text(self):
        assert compute_similarity("", "hello") == 0.0
        assert compute_similarity("hello", "") == 0.0


class TestFindBestMatch:
    """Tests for finding best matching chunk."""

    def test_finds_match(self):
        chunks = [
            {"body": "def foo(): return 42", "file_path": "a.py"},
            {"body": "class Bar: pass", "file_path": "b.py"},
        ]
        match = find_best_match("def foo(): return 42", chunks)
        assert match is not None
        assert match["file_path"] == "a.py"
        assert match["score"] > 0.8

    def test_low_similarity_match(self):
        chunks = [
            {"body": "def calculate_sum(a, b): return a + b", "file_path": "a.py"},
        ]
        # Different code should have low similarity but may still return a match
        match = find_best_match("class UserRepository: pass", chunks)
        # Should return match with low score (threshold is 0.3)
        if match:
            assert match["score"] < 0.5

    def test_empty_chunks(self):
        match = find_best_match("query", [])
        assert match is None


class TestVerifyRagSources:
    """Tests for RAG source verification."""

    def test_grounded_response(self):
        response = "Here is the code:\n```python\ndef calculate(x):\n    return x * 2\n```"
        chunks = [
            {"body": "def calculate(x):\n    return x * 2", "file_path": "math.py", "start_line": 1, "end_line": 2},
        ]

        result = verify_rag_sources(response, chunks)

        assert result["total_blocks"] == 1
        assert result["grounded_blocks"] == 1
        assert result["grounding_rate"] == 1.0
        assert len(result["potentially_hallucinated"]) == 0

    def test_ungrounded_response(self):
        response = "```python\ndef totally_new_function():\n    return 'made up'\n```"
        chunks = [
            {"body": "def existing_function(): pass", "file_path": "a.py"},
        ]

        result = verify_rag_sources(response, chunks)

        assert result["total_blocks"] == 1
        assert result["grounded_blocks"] == 0
        assert result["grounding_rate"] == 0.0
        assert len(result["potentially_hallucinated"]) == 1

    def test_empty_response(self):
        result = verify_rag_sources("", [])
        assert result["grounding_rate"] == 1.0

    def test_no_code_blocks(self):
        result = verify_rag_sources("Just text explanation", [{"body": "code"}])
        assert result["total_blocks"] == 0
        assert result["grounding_rate"] == 1.0


class TestAddCitations:
    """Tests for citation generation."""

    def test_adds_citations(self):
        response = "```python\ndef calculate_average(numbers):\n    total = sum(numbers)\n    return total / len(numbers)\n```"
        chunks = [
            {"body": "def calculate_average(numbers):\n    total = sum(numbers)\n    return total / len(numbers)", "file_path": "utils.py", "start_line": 10, "end_line": 13},
        ]

        result = add_citations(response, chunks)

        assert "**Sources:**" in result
        assert "utils.py" in result
        assert "lines 10-13" in result

    def test_no_chunks(self):
        response = "Some text"
        result = add_citations(response, [])
        assert result == response

    def test_empty_response(self):
        result = add_citations("", [{"body": "code"}])
        assert result == ""


class TestCheckHallucinationRisk:
    """Tests for quick risk assessment."""

    def test_high_risk_no_context(self):
        result = check_hallucination_risk("```code```", [])
        assert result["risk_level"] == "high"

    def test_low_risk_no_code(self):
        result = check_hallucination_risk("Just text", [{"body": "code"}])
        assert result["risk_level"] == "low"

    def test_grounded_code(self):
        response = "```python\ndef foo(): return 1\n```"
        chunks = [{"body": "def foo(): return 1", "file_path": "a.py"}]
        result = check_hallucination_risk(response, chunks)
        assert result["risk_level"] == "low"
