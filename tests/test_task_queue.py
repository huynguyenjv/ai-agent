"""Tests for task queue with dependency resolution."""

import pytest

from server.agent.task_queue import Task, TaskQueue, TaskStatus


class TestTask:
    """Tests for Task dataclass."""

    def test_task_creation(self):
        task = Task(id="t1", action="read_file", params={"path": "test.py"})
        assert task.id == "t1"
        assert task.action == "read_file"
        assert task.status == TaskStatus.PENDING
        assert task.depends_on == []

    def test_task_duration(self):
        task = Task(id="t1", action="test", params={})
        task.start_time = 1.0
        task.end_time = 1.5
        assert task.duration_ms == 500.0


class TestTaskQueue:
    """Tests for TaskQueue."""

    def test_add_task(self):
        queue = TaskQueue()
        task = Task(id="t1", action="read", params={})
        queue.add_task(task)
        assert queue.get_task("t1") == task

    def test_add_duplicate_task_raises(self):
        queue = TaskQueue()
        task = Task(id="t1", action="read", params={})
        queue.add_task(task)
        with pytest.raises(ValueError, match="already exists"):
            queue.add_task(task)

    def test_get_ready_tasks_no_deps(self):
        queue = TaskQueue()
        queue.add_task(Task(id="t1", action="a", params={}))
        queue.add_task(Task(id="t2", action="b", params={}))
        ready = queue.get_ready_tasks()
        assert len(ready) == 2

    def test_get_ready_tasks_with_deps(self):
        queue = TaskQueue()
        queue.add_task(Task(id="t1", action="a", params={}))
        queue.add_task(Task(id="t2", action="b", params={}, depends_on=["t1"]))

        ready = queue.get_ready_tasks()
        assert len(ready) == 1
        assert ready[0].id == "t1"

    def test_get_ready_tasks_after_completion(self):
        queue = TaskQueue()
        t1 = Task(id="t1", action="a", params={})
        t2 = Task(id="t2", action="b", params={}, depends_on=["t1"])
        queue.add_task(t1)
        queue.add_task(t2)

        # Complete t1
        t1.status = TaskStatus.COMPLETED

        ready = queue.get_ready_tasks()
        assert len(ready) == 1
        assert ready[0].id == "t2"

    def test_skip_on_failed_dependency(self):
        queue = TaskQueue()
        t1 = Task(id="t1", action="a", params={})
        t2 = Task(id="t2", action="b", params={}, depends_on=["t1"])
        queue.add_task(t1)
        queue.add_task(t2)

        # Fail t1
        t1.status = TaskStatus.FAILED

        ready = queue.get_ready_tasks()
        assert len(ready) == 0
        assert t2.status == TaskStatus.SKIPPED

    def test_has_cycle_simple(self):
        queue = TaskQueue()
        queue.add_task(Task(id="t1", action="a", params={}, depends_on=["t2"]))
        queue.add_task(Task(id="t2", action="b", params={}, depends_on=["t1"]))
        assert queue.has_cycle() is True

    def test_no_cycle(self):
        queue = TaskQueue()
        queue.add_task(Task(id="t1", action="a", params={}))
        queue.add_task(Task(id="t2", action="b", params={}, depends_on=["t1"]))
        queue.add_task(Task(id="t3", action="c", params={}, depends_on=["t2"]))
        assert queue.has_cycle() is False

    def test_max_parallel(self):
        queue = TaskQueue(max_parallel=2)
        for i in range(5):
            queue.add_task(Task(id=f"t{i}", action="a", params={}))

        ready = queue.get_ready_tasks()
        assert len(ready) == 2

    @pytest.mark.asyncio
    async def test_execute_all_simple(self):
        queue = TaskQueue()
        queue.add_task(Task(id="t1", action="add", params={"x": 1}))
        queue.add_task(Task(id="t2", action="add", params={"x": 2}))

        async def executor(task):
            return task.params["x"] * 2

        results = await queue.execute_all(executor)
        assert results["t1"] == 2
        assert results["t2"] == 4

    @pytest.mark.asyncio
    async def test_execute_all_with_deps(self):
        queue = TaskQueue()
        queue.add_task(Task(id="t1", action="add", params={"x": 5}))
        queue.add_task(Task(id="t2", action="mult", params={"y": 3}, depends_on=["t1"]))

        async def executor(task):
            if task.action == "add":
                return task.params["x"] + 10
            else:
                dep_result = task.params["_dep_results"]["t1"]
                return dep_result * task.params["y"]

        results = await queue.execute_all(executor)
        assert results["t1"] == 15  # 5 + 10
        assert results["t2"] == 45  # 15 * 3

    @pytest.mark.asyncio
    async def test_execute_all_cycle_raises(self):
        queue = TaskQueue()
        queue.add_task(Task(id="t1", action="a", params={}, depends_on=["t2"]))
        queue.add_task(Task(id="t2", action="b", params={}, depends_on=["t1"]))

        async def executor(task):
            return "done"

        with pytest.raises(ValueError, match="Circular dependency"):
            await queue.execute_all(executor)

    def test_get_summary(self):
        queue = TaskQueue()
        t1 = Task(id="t1", action="a", params={})
        t2 = Task(id="t2", action="b", params={})
        queue.add_task(t1)
        queue.add_task(t2)

        t1.status = TaskStatus.COMPLETED
        t2.status = TaskStatus.FAILED

        summary = queue.get_summary()
        assert summary["total_tasks"] == 2
        assert summary["completed"] == 1
        assert summary["failed"] == 1

    def test_clear(self):
        queue = TaskQueue()
        queue.add_task(Task(id="t1", action="a", params={}))
        queue.clear()
        assert queue.get_task("t1") is None
