"""Tests for parallel tool execution."""

import asyncio
import pytest

from server.agent.parallel_tools import (
    execute_tools_parallel,
    execute_tool_batch,
    classify_tool,
    can_parallelize,
    READ_ONLY_TOOLS,
    WRITE_TOOLS,
)


class TestClassifyTool:
    """Tests for tool classification."""

    def test_read_only_tools(self):
        for tool in READ_ONLY_TOOLS:
            assert classify_tool(tool) == "read"

    def test_write_tools(self):
        for tool in WRITE_TOOLS:
            assert classify_tool(tool) == "write"

    def test_unknown_tool_is_write(self):
        assert classify_tool("unknown_tool") == "write"


class TestCanParallelize:
    """Tests for parallelization detection."""

    def test_multiple_reads_can_parallelize(self):
        tool_calls = [
            {"function": {"name": "vtrip_read_file"}},
            {"function": {"name": "vtrip_search_symbol"}},
        ]
        assert can_parallelize(tool_calls) is True

    def test_single_read_cannot_parallelize(self):
        tool_calls = [
            {"function": {"name": "vtrip_read_file"}},
        ]
        assert can_parallelize(tool_calls) is False

    def test_only_writes_cannot_parallelize(self):
        tool_calls = [
            {"function": {"name": "vtrip_apply_edits"}},
            {"function": {"name": "vtrip_run_command"}},
        ]
        assert can_parallelize(tool_calls) is False

    def test_empty_list(self):
        assert can_parallelize([]) is False


class TestExecuteToolsParallel:
    """Tests for parallel tool execution."""

    @pytest.mark.asyncio
    async def test_empty_list(self):
        async def executor(tc):
            return {"result": "ok"}

        results = await execute_tools_parallel([], executor)
        assert results == []

    @pytest.mark.asyncio
    async def test_read_tools_run_parallel(self):
        execution_order = []

        async def executor(tc):
            name = tc["function"]["name"]
            execution_order.append(f"start_{name}")
            await asyncio.sleep(0.05)  # Simulate work
            execution_order.append(f"end_{name}")
            return {"name": name, "result": "ok"}

        tool_calls = [
            {"function": {"name": "vtrip_read_file", "arguments": {}}},
            {"function": {"name": "vtrip_search_symbol", "arguments": {}}},
            {"function": {"name": "vtrip_git_status", "arguments": {}}},
        ]

        results = await execute_tools_parallel(tool_calls, executor)

        # All should complete
        assert len(results) == 3
        assert all(r["result"] == "ok" for r in results)

        # Parallel execution: starts should happen before ends
        starts = [e for e in execution_order if e.startswith("start")]
        assert len(starts) == 3

    @pytest.mark.asyncio
    async def test_write_tools_run_sequential(self):
        execution_order = []

        async def executor(tc):
            name = tc["function"]["name"]
            execution_order.append(f"start_{name}")
            await asyncio.sleep(0.01)
            execution_order.append(f"end_{name}")
            return {"name": name}

        tool_calls = [
            {"function": {"name": "vtrip_apply_edits", "arguments": {}}},
            {"function": {"name": "vtrip_run_command", "arguments": {}}},
        ]

        results = await execute_tools_parallel(tool_calls, executor)

        assert len(results) == 2
        # Sequential: first must end before second starts
        assert execution_order.index("end_vtrip_apply_edits") < \
               execution_order.index("start_vtrip_run_command")

    @pytest.mark.asyncio
    async def test_mixed_tools_order_preserved(self):
        async def executor(tc):
            name = tc["function"]["name"]
            return {"name": name}

        tool_calls = [
            {"function": {"name": "vtrip_read_file", "arguments": {}}},
            {"function": {"name": "vtrip_apply_edits", "arguments": {}}},
            {"function": {"name": "vtrip_search_symbol", "arguments": {}}},
        ]

        results = await execute_tools_parallel(tool_calls, executor)

        # Results should be in same order as input
        assert results[0]["name"] == "vtrip_read_file"
        assert results[1]["name"] == "vtrip_apply_edits"
        assert results[2]["name"] == "vtrip_search_symbol"

    @pytest.mark.asyncio
    async def test_executor_error_handled(self):
        async def executor(tc):
            name = tc["function"]["name"]
            if "error" in name:
                raise ValueError("Test error")
            return {"name": name}

        tool_calls = [
            {"function": {"name": "vtrip_read_file", "arguments": {}}},
            {"name": "error_tool"},  # Will fail
        ]

        results = await execute_tools_parallel(tool_calls, executor)

        assert results[0]["name"] == "vtrip_read_file"
        assert "error" in results[1]

    @pytest.mark.asyncio
    async def test_respects_max_parallel(self):
        concurrent_count = 0
        max_concurrent = 0

        async def executor(tc):
            nonlocal concurrent_count, max_concurrent
            concurrent_count += 1
            max_concurrent = max(max_concurrent, concurrent_count)
            await asyncio.sleep(0.02)
            concurrent_count -= 1
            return {"ok": True}

        # 10 read tools with max_parallel=3
        tool_calls = [
            {"function": {"name": "vtrip_read_file", "arguments": {}}}
            for _ in range(10)
        ]

        await execute_tools_parallel(tool_calls, executor, max_parallel=3)

        assert max_concurrent <= 3


class TestExecuteToolBatch:
    """Tests for batch execution (all parallel)."""

    @pytest.mark.asyncio
    async def test_all_run_parallel(self):
        execution_times = []

        async def executor(tc):
            start = asyncio.get_event_loop().time()
            await asyncio.sleep(0.05)
            execution_times.append(start)
            return {"ok": True}

        tool_calls = [{"name": f"tool_{i}"} for i in range(5)]

        results = await execute_tool_batch(tool_calls, executor)

        assert len(results) == 5
        # All should start at roughly the same time (parallel)
        time_spread = max(execution_times) - min(execution_times)
        assert time_spread < 0.02  # Should all start within 20ms

    @pytest.mark.asyncio
    async def test_error_in_batch(self):
        async def executor(tc):
            if tc["name"] == "fail":
                raise ValueError("fail")
            return {"ok": True}

        tool_calls = [
            {"name": "ok1"},
            {"name": "fail"},
            {"name": "ok2"},
        ]

        results = await execute_tool_batch(tool_calls, executor)

        assert results[0] == {"ok": True}
        assert "error" in results[1]
        assert results[2] == {"ok": True}
