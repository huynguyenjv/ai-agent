"""Async file I/O utilities.

Provides non-blocking file operations for better concurrency.
"""

from __future__ import annotations

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any

logger = logging.getLogger("server.utils.async_io")

# Thread pool for file I/O
_executor: ThreadPoolExecutor | None = None
_MAX_WORKERS = 4


def _get_executor() -> ThreadPoolExecutor:
    """Get or create thread pool executor."""
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=_MAX_WORKERS, thread_name_prefix="async_io")
    return _executor


async def read_file_async(
    file_path: str,
    encoding: str = "utf-8",
    errors: str = "replace",
) -> str:
    """Read file asynchronously.

    Args:
        file_path: Path to file
        encoding: File encoding
        errors: Error handling mode

    Returns:
        File content

    Raises:
        FileNotFoundError: If file doesn't exist
        OSError: On read error
    """
    loop = asyncio.get_event_loop()

    def _read():
        with open(file_path, "r", encoding=encoding, errors=errors) as f:
            return f.read()

    return await loop.run_in_executor(_get_executor(), _read)


async def read_file_lines_async(
    file_path: str,
    start_line: int = 1,
    end_line: int | None = None,
    encoding: str = "utf-8",
) -> tuple[list[str], int]:
    """Read specific lines from file asynchronously.

    Args:
        file_path: Path to file
        start_line: Start line (1-based)
        end_line: End line (inclusive), None for all
        encoding: File encoding

    Returns:
        Tuple of (lines, total_line_count)
    """
    loop = asyncio.get_event_loop()

    def _read_lines():
        with open(file_path, "r", encoding=encoding, errors="replace") as f:
            all_lines = f.readlines()

        total = len(all_lines)
        start = max(0, start_line - 1)
        end = end_line if end_line else total

        return all_lines[start:end], total

    return await loop.run_in_executor(_get_executor(), _read_lines)


async def write_file_async(
    file_path: str,
    content: str,
    encoding: str = "utf-8",
    create_dirs: bool = True,
) -> None:
    """Write file asynchronously.

    Args:
        file_path: Path to file
        content: Content to write
        encoding: File encoding
        create_dirs: Create parent directories if needed
    """
    loop = asyncio.get_event_loop()

    def _write():
        if create_dirs:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding=encoding) as f:
            f.write(content)

    await loop.run_in_executor(_get_executor(), _write)


async def file_exists_async(file_path: str) -> bool:
    """Check if file exists asynchronously.

    Args:
        file_path: Path to check

    Returns:
        True if file exists
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_get_executor(), os.path.isfile, file_path)


async def list_files_async(
    directory: str,
    pattern: str | None = None,
    recursive: bool = False,
) -> list[str]:
    """List files in directory asynchronously.

    Args:
        directory: Directory path
        pattern: Glob pattern (e.g., "*.py")
        recursive: Search recursively

    Returns:
        List of file paths
    """
    loop = asyncio.get_event_loop()

    def _list():
        import glob as glob_module

        if pattern:
            if recursive:
                search_pattern = os.path.join(directory, "**", pattern)
                return glob_module.glob(search_pattern, recursive=True)
            else:
                search_pattern = os.path.join(directory, pattern)
                return glob_module.glob(search_pattern)
        else:
            files = []
            for entry in os.scandir(directory):
                if entry.is_file():
                    files.append(entry.path)
            return files

    return await loop.run_in_executor(_get_executor(), _list)


async def read_files_parallel(
    file_paths: list[str],
    encoding: str = "utf-8",
) -> dict[str, str | Exception]:
    """Read multiple files in parallel.

    Args:
        file_paths: List of file paths
        encoding: File encoding

    Returns:
        Dict mapping file path to content or exception
    """
    tasks = [
        read_file_async(path, encoding)
        for path in file_paths
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    return {
        path: result
        for path, result in zip(file_paths, results)
    }


async def get_file_stats_async(file_path: str) -> dict[str, Any] | None:
    """Get file statistics asynchronously.

    Args:
        file_path: Path to file

    Returns:
        Stats dict or None if file doesn't exist
    """
    loop = asyncio.get_event_loop()

    def _stats():
        try:
            stat = os.stat(file_path)
            return {
                "size": stat.st_size,
                "mtime": stat.st_mtime,
                "ctime": stat.st_ctime,
                "is_file": os.path.isfile(file_path),
                "is_dir": os.path.isdir(file_path),
            }
        except OSError:
            return None

    return await loop.run_in_executor(_get_executor(), _stats)


def shutdown_executor() -> None:
    """Shutdown the thread pool executor."""
    global _executor
    if _executor:
        _executor.shutdown(wait=True)
        _executor = None
        logger.info("Async I/O executor shutdown")
