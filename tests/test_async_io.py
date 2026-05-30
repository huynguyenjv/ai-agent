"""Tests for async file I/O utilities."""

import pytest

from server.utils.async_io import (
    read_file_async,
    read_file_lines_async,
    write_file_async,
    file_exists_async,
    list_files_async,
    read_files_parallel,
    get_file_stats_async,
)


class TestReadFileAsync:
    """Tests for async file reading."""

    @pytest.mark.asyncio
    async def test_read_file(self, tmp_path):
        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello World!")

        content = await read_file_async(str(test_file))
        assert content == "Hello World!"

    @pytest.mark.asyncio
    async def test_read_nonexistent_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            await read_file_async(str(tmp_path / "nonexistent.txt"))

    @pytest.mark.asyncio
    async def test_read_with_encoding(self, tmp_path):
        test_file = tmp_path / "unicode.txt"
        test_file.write_text("こんにちは", encoding="utf-8")

        content = await read_file_async(str(test_file), encoding="utf-8")
        assert content == "こんにちは"


class TestReadFileLinesAsync:
    """Tests for async line reading."""

    @pytest.mark.asyncio
    async def test_read_all_lines(self, tmp_path):
        test_file = tmp_path / "lines.txt"
        test_file.write_text("line1\nline2\nline3\n")

        lines, total = await read_file_lines_async(str(test_file))
        assert len(lines) == 3
        assert total == 3

    @pytest.mark.asyncio
    async def test_read_specific_lines(self, tmp_path):
        test_file = tmp_path / "lines.txt"
        test_file.write_text("line1\nline2\nline3\nline4\nline5\n")

        lines, total = await read_file_lines_async(str(test_file), start_line=2, end_line=4)
        assert len(lines) == 3
        assert "line2" in lines[0]
        assert total == 5


class TestWriteFileAsync:
    """Tests for async file writing."""

    @pytest.mark.asyncio
    async def test_write_file(self, tmp_path):
        test_file = tmp_path / "output.txt"

        await write_file_async(str(test_file), "Test content")

        assert test_file.exists()
        assert test_file.read_text() == "Test content"

    @pytest.mark.asyncio
    async def test_write_creates_dirs(self, tmp_path):
        test_file = tmp_path / "subdir" / "nested" / "file.txt"

        await write_file_async(str(test_file), "Nested content")

        assert test_file.exists()


class TestFileExistsAsync:
    """Tests for async file existence check."""

    @pytest.mark.asyncio
    async def test_exists(self, tmp_path):
        test_file = tmp_path / "exists.txt"
        test_file.write_text("x")

        assert await file_exists_async(str(test_file)) is True

    @pytest.mark.asyncio
    async def test_not_exists(self, tmp_path):
        assert await file_exists_async(str(tmp_path / "nope.txt")) is False


class TestListFilesAsync:
    """Tests for async file listing."""

    @pytest.mark.asyncio
    async def test_list_all_files(self, tmp_path):
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.txt").write_text("b")
        (tmp_path / "c.py").write_text("c")

        files = await list_files_async(str(tmp_path))
        assert len(files) == 3

    @pytest.mark.asyncio
    async def test_list_with_pattern(self, tmp_path):
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.txt").write_text("b")
        (tmp_path / "c.py").write_text("c")

        files = await list_files_async(str(tmp_path), pattern="*.txt")
        assert len(files) == 2


class TestReadFilesParallel:
    """Tests for parallel file reading."""

    @pytest.mark.asyncio
    async def test_read_multiple(self, tmp_path):
        (tmp_path / "a.txt").write_text("content a")
        (tmp_path / "b.txt").write_text("content b")

        paths = [str(tmp_path / "a.txt"), str(tmp_path / "b.txt")]
        results = await read_files_parallel(paths)

        assert results[paths[0]] == "content a"
        assert results[paths[1]] == "content b"

    @pytest.mark.asyncio
    async def test_read_with_error(self, tmp_path):
        (tmp_path / "exists.txt").write_text("ok")

        paths = [str(tmp_path / "exists.txt"), str(tmp_path / "missing.txt")]
        results = await read_files_parallel(paths)

        assert results[paths[0]] == "ok"
        assert isinstance(results[paths[1]], Exception)


class TestGetFileStatsAsync:
    """Tests for async file stats."""

    @pytest.mark.asyncio
    async def test_get_stats(self, tmp_path):
        test_file = tmp_path / "stats.txt"
        test_file.write_text("x" * 100)

        stats = await get_file_stats_async(str(test_file))

        assert stats is not None
        assert stats["size"] == 100
        assert stats["is_file"] is True

    @pytest.mark.asyncio
    async def test_stats_nonexistent(self, tmp_path):
        stats = await get_file_stats_async(str(tmp_path / "nope.txt"))
        assert stats is None
