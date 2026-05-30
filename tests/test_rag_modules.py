"""Tests for RAG modules: query expansion, context retrieval, chunking."""

import pytest

from server.rag.query_expand import _parse_variants, _merge_results, _get_doc_id
from server.rag.context_retrieval import _find_container, retrieve_with_context
from server.rag.chunking import (
    chunk_by_lines,
    chunk_by_tokens,
    dedup_overlapping_results,
    Chunk,
)


class TestQueryExpand:
    """Tests for query expansion module."""

    def test_parse_variants_valid_json(self):
        content = '["find user", "search for user", "locate user record"]'
        variants = _parse_variants(content)
        assert len(variants) == 3
        assert "find user" in variants

    def test_parse_variants_with_wrapper(self):
        content = 'Here are variants: ["var1", "var2"]'
        variants = _parse_variants(content)
        assert len(variants) == 2

    def test_parse_variants_invalid(self):
        content = "not valid json"
        variants = _parse_variants(content)
        assert variants == []

    def test_merge_results_rrf(self):
        list1 = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
        list2 = [{"id": "b"}, {"id": "c"}, {"id": "d"}]

        merged = _merge_results([list1, list2])

        # b and c should rank higher (appear in both)
        ids = [r["id"] for r in merged]
        assert ids.index("b") < ids.index("a")  # b appears in both, ranks higher
        assert ids.index("c") < ids.index("d")

    def test_get_doc_id(self):
        assert _get_doc_id({"id": "123"}) == "123"
        assert _get_doc_id({"file_path": "a.py", "start_line": 10}) == "a.py10"


class TestContextRetrieval:
    """Tests for context retrieval module."""

    def test_find_container_python_function(self):
        lines = [
            "import os\n",
            "\n",
            "def my_function():\n",
            "    x = 1\n",
            "    return x\n",
        ]
        container = _find_container(lines, 4)
        assert container is not None
        assert container["type"] == "function"
        assert container["name"] == "my_function"

    def test_find_container_python_class(self):
        lines = [
            "class MyClass:\n",
            "    x = 1\n",
            "    y = 2\n",
        ]
        container = _find_container(lines, 3)
        assert container is not None
        assert container["type"] == "class"
        assert container["name"] == "MyClass"

    def test_find_container_java_class(self):
        lines = [
            "package com.example;\n",
            "\n",
            "public class UserService {\n",
            "    private String name;\n",
            "}\n",
        ]
        container = _find_container(lines, 4)
        assert container is not None
        assert container["name"] == "UserService"

    def test_find_container_none(self):
        lines = ["x = 1\n", "y = 2\n"]
        container = _find_container(lines, 2)
        assert container is None


class TestChunking:
    """Tests for chunking module."""

    def test_chunk_by_lines_small_file(self):
        content = "line1\nline2\nline3"
        chunks = chunk_by_lines(content, chunk_size=10, overlap=2)
        assert len(chunks) == 1
        assert chunks[0].has_overlap is False

    def test_chunk_by_lines_with_overlap(self):
        content = "\n".join([f"line{i}" for i in range(50)])
        chunks = chunk_by_lines(content, chunk_size=20, overlap=5)

        assert len(chunks) >= 2
        # Second chunk should have overlap
        assert chunks[1].has_overlap is True

        # Check overlap - end of chunk 1 should overlap with start of chunk 2
        chunk1_end = chunks[0].end_line
        chunk2_start = chunks[1].start_line
        assert chunk2_start < chunk1_end  # Overlap exists

    def test_chunk_by_tokens_small_content(self):
        content = "hello world"
        chunks = chunk_by_tokens(content, max_tokens=100, overlap_tokens=10)
        assert len(chunks) == 1

    def test_chunk_by_tokens_large_content(self):
        content = " ".join([f"word{i}" for i in range(100)])
        chunks = chunk_by_tokens(content, max_tokens=30, overlap_tokens=5)
        assert len(chunks) >= 3

    def test_dedup_overlapping_same_file(self):
        results = [
            {"file_path": "a.py", "start_line": 1, "end_line": 50},
            {"file_path": "a.py", "start_line": 10, "end_line": 60},  # Heavy overlap (40/50 = 80%)
            {"file_path": "b.py", "start_line": 1, "end_line": 50},  # Different file
        ]
        deduped = dedup_overlapping_results(results, overlap_threshold=0.5)

        # Should keep first from a.py and b.py
        assert len(deduped) == 2
        files = [r["file_path"] for r in deduped]
        assert "a.py" in files
        assert "b.py" in files

    def test_dedup_no_overlap(self):
        results = [
            {"file_path": "a.py", "start_line": 1, "end_line": 50},
            {"file_path": "a.py", "start_line": 100, "end_line": 150},
        ]
        deduped = dedup_overlapping_results(results)
        assert len(deduped) == 2

    def test_dedup_empty(self):
        assert dedup_overlapping_results([]) == []

    def test_dedup_single(self):
        results = [{"file_path": "a.py", "start_line": 1, "end_line": 10}]
        assert dedup_overlapping_results(results) == results
