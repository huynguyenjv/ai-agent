"""Parallel tool execution with dependency awareness.

Executes independent tools (read-only) in parallel while maintaining
sequential execution for dependent tools (write operations).
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, Awaitable

logger = logging.getLogger("server.agent.parallel_tools")

# Read-only tools that can be executed in parallel
READ_ONLY_TOOLS = {
    "vtrip_read_file",
    "vtrip_search_symbol",
    "vtrip_get_project_skeleton",
    "vtrip_index_with_deps",
    "vtrip_git_status",
    "vtrip_git_diff",
    "vtrip_git_log",
    "vtrip_git_branch",
}

# Write tools that must be executed sequentially
WRITE_TOOLS = {
    "vtrip_run_command",
    "vtrip_diff_preview",
    "vtrip_apply_edits",
    "vtrip_git_commit",
}

# Maximum parallel executions
MAX_PARALLEL = 5


async def execute_tools_parallel(
    tool_calls: list[dict],
    executor: Callable[[dict], Awaitable[dict]],
    max_parallel: int = MAX_PARALLEL,
) -> list[dict]:
    """Execute tools with parallel optimization for read-only operations.

    Strategy:
    1. Classify tools as read-only or write
    2. Execute all read-only tools in parallel (with concurrency limit)
    3. Execute write tools sequentially after reads complete

    Args:
        tool_calls: List of tool call dicts with name, arguments
        executor: Async function to execute a single tool
        max_parallel: Maximum concurrent executions

    Returns:
        List of results in same order as tool_calls
    """
    if not tool_calls:
        return []

    start_time = time.time()

    # Classify tools
    read_indices = []
    write_indices = []

    for i, tc in enumerate(tool_calls):
        name = tc.get("function", {}).get("name", tc.get("name", ""))
        if name in READ_ONLY_TOOLS:
            read_indices.append(i)
        else:
            write_indices.append(i)

    logger.info("parallel_tools: %d read-only, %d write tools",
                len(read_indices), len(write_indices))

    # Pre-allocate results list
    results = [None] * len(tool_calls)

    # Execute read-only tools in parallel with semaphore
    if read_indices:
        semaphore = asyncio.Semaphore(max_parallel)

        async def execute_with_semaphore(idx: int) -> tuple[int, dict]:
            async with semaphore:
                result = await executor(tool_calls[idx])
                return idx, result

        read_tasks = [execute_with_semaphore(i) for i in read_indices]
        read_results = await asyncio.gather(*read_tasks, return_exceptions=True)

        for item in read_results:
            if isinstance(item, Exception):
                logger.error("parallel_tools: read tool failed: %s", item)
                # Find which index failed (we don't know exactly, mark first None)
                for i in read_indices:
                    if results[i] is None:
                        results[i] = {"error": str(item)}
                        break
            else:
                idx, result = item
                results[idx] = result

    # Execute write tools sequentially
    for idx in write_indices:
        try:
            result = await executor(tool_calls[idx])
            results[idx] = result
        except Exception as e:
            logger.error("parallel_tools: write tool failed: %s", e)
            results[idx] = {"error": str(e)}

    duration_ms = (time.time() - start_time) * 1000
    logger.info("parallel_tools: completed %d tools in %.0fms",
                len(tool_calls), duration_ms)

    return results


async def execute_tool_batch(
    tool_calls: list[dict],
    executor: Callable[[dict], Awaitable[dict]],
) -> list[dict]:
    """Execute a batch of tools, all in parallel (no dependency check).

    Use this when you know all tools are independent.

    Args:
        tool_calls: List of tool calls
        executor: Async function to execute a single tool

    Returns:
        List of results in same order
    """
    if not tool_calls:
        return []

    tasks = [executor(tc) for tc in tool_calls]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    return [
        r if not isinstance(r, Exception) else {"error": str(r)}
        for r in results
    ]


def classify_tool(tool_name: str) -> str:
    """Classify a tool as read-only or write.

    Args:
        tool_name: Tool function name

    Returns:
        "read" or "write"
    """
    if tool_name in READ_ONLY_TOOLS:
        return "read"
    return "write"


def can_parallelize(tool_calls: list[dict]) -> bool:
    """Check if tool calls can benefit from parallelization.

    Args:
        tool_calls: List of tool calls

    Returns:
        True if there are multiple read-only tools
    """
    read_count = sum(
        1 for tc in tool_calls
        if tc.get("function", {}).get("name", tc.get("name", "")) in READ_ONLY_TOOLS
    )
    return read_count > 1
