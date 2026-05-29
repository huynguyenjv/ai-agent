"""Task Queue with dependency resolution and parallel execution.

Manages task execution order based on dependencies, enabling parallel
execution of independent tasks.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Awaitable

logger = logging.getLogger("server.agent.task_queue")


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Task:
    """A single task in the queue."""
    id: str
    action: str
    params: dict[str, Any]
    depends_on: list[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: str | None = None
    start_time: float | None = None
    end_time: float | None = None

    @property
    def duration_ms(self) -> float | None:
        """Task duration in milliseconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time) * 1000
        return None


class TaskQueue:
    """Manages task execution with dependency resolution."""

    def __init__(self, max_parallel: int = 5):
        """Initialize task queue.

        Args:
            max_parallel: Maximum number of tasks to run in parallel
        """
        self._tasks: dict[str, Task] = {}
        self._results: dict[str, Any] = {}
        self._max_parallel = max_parallel
        self._execution_order: list[str] = []

    def add_task(self, task: Task) -> None:
        """Add a task to the queue.

        Args:
            task: Task to add

        Raises:
            ValueError: If task ID already exists or has invalid dependencies
        """
        if task.id in self._tasks:
            raise ValueError(f"Task {task.id} already exists")

        # Validate dependencies exist (except for first batch)
        for dep in task.depends_on:
            if dep not in self._tasks:
                logger.warning("Task %s depends on unknown task %s", task.id, dep)

        self._tasks[task.id] = task
        logger.debug("Added task %s (action=%s, depends_on=%s)",
                     task.id, task.action, task.depends_on)

    def add_tasks(self, tasks: list[Task]) -> None:
        """Add multiple tasks to the queue.

        Args:
            tasks: List of tasks to add
        """
        for task in tasks:
            self.add_task(task)

    def get_ready_tasks(self) -> list[Task]:
        """Get tasks whose dependencies are satisfied.

        Returns:
            List of tasks ready to execute
        """
        ready = []
        for task in self._tasks.values():
            if task.status != TaskStatus.PENDING:
                continue

            # Check all dependencies are completed
            deps_satisfied = all(
                self._tasks.get(dep, Task(dep, "", {})).status == TaskStatus.COMPLETED
                for dep in task.depends_on
            )

            # Check no dependency failed
            deps_failed = any(
                self._tasks.get(dep, Task(dep, "", {})).status == TaskStatus.FAILED
                for dep in task.depends_on
            )

            if deps_failed:
                task.status = TaskStatus.SKIPPED
                task.error = "Dependency failed"
                logger.info("Skipping task %s due to failed dependency", task.id)
            elif deps_satisfied:
                ready.append(task)

        return ready[:self._max_parallel]

    def has_pending_tasks(self) -> bool:
        """Check if there are pending tasks."""
        return any(t.status == TaskStatus.PENDING for t in self._tasks.values())

    def has_cycle(self) -> bool:
        """Check for circular dependencies using DFS.

        Returns:
            True if cycle detected
        """
        visited = set()
        rec_stack = set()

        def dfs(task_id: str) -> bool:
            visited.add(task_id)
            rec_stack.add(task_id)

            task = self._tasks.get(task_id)
            if task:
                for dep in task.depends_on:
                    if dep not in visited:
                        if dfs(dep):
                            return True
                    elif dep in rec_stack:
                        return True

            rec_stack.remove(task_id)
            return False

        for task_id in self._tasks:
            if task_id not in visited:
                if dfs(task_id):
                    return True

        return False

    async def execute_all(
        self,
        executor: Callable[[Task], Awaitable[Any]],
        timeout: float = 300.0,
    ) -> dict[str, Any]:
        """Execute all tasks respecting dependencies.

        Args:
            executor: Async function to execute each task
            timeout: Maximum total execution time in seconds

        Returns:
            Dict mapping task IDs to results
        """
        if self.has_cycle():
            raise ValueError("Circular dependency detected in task graph")

        start_time = time.time()

        while self.has_pending_tasks():
            # Check timeout
            if time.time() - start_time > timeout:
                logger.error("Task queue execution timed out after %.1fs", timeout)
                for task in self._tasks.values():
                    if task.status == TaskStatus.PENDING:
                        task.status = TaskStatus.FAILED
                        task.error = "Timeout"
                break

            ready = self.get_ready_tasks()
            if not ready:
                # No ready tasks but still pending = deadlock or waiting
                if self.has_pending_tasks():
                    logger.warning("No ready tasks but %d pending - possible deadlock",
                                   sum(1 for t in self._tasks.values()
                                       if t.status == TaskStatus.PENDING))
                break

            logger.info("Executing %d parallel tasks: %s",
                        len(ready), [t.id for t in ready])

            # Mark tasks as running
            for task in ready:
                task.status = TaskStatus.RUNNING
                task.start_time = time.time()

            # Execute ready tasks in parallel
            results = await asyncio.gather(
                *[self._execute_task(executor, task) for task in ready],
                return_exceptions=True
            )

            # Process results
            for task, result in zip(ready, results):
                task.end_time = time.time()
                self._execution_order.append(task.id)

                if isinstance(result, Exception):
                    task.status = TaskStatus.FAILED
                    task.error = str(result)
                    logger.error("Task %s failed: %s", task.id, result)
                else:
                    task.status = TaskStatus.COMPLETED
                    task.result = result
                    self._results[task.id] = result
                    logger.info("Task %s completed in %.0fms",
                                task.id, task.duration_ms)

        return self._results

    async def _execute_task(
        self,
        executor: Callable[[Task], Awaitable[Any]],
        task: Task,
    ) -> Any:
        """Execute a single task with error handling.

        Args:
            executor: Async function to execute the task
            task: Task to execute

        Returns:
            Task result
        """
        try:
            # Pass dependency results to task params
            dep_results = {
                dep: self._results.get(dep)
                for dep in task.depends_on
            }
            task.params["_dep_results"] = dep_results

            return await executor(task)
        except Exception as e:
            logger.exception("Error executing task %s", task.id)
            raise

    def get_summary(self) -> dict[str, Any]:
        """Get execution summary.

        Returns:
            Summary dict with counts and timing
        """
        completed = sum(1 for t in self._tasks.values() if t.status == TaskStatus.COMPLETED)
        failed = sum(1 for t in self._tasks.values() if t.status == TaskStatus.FAILED)
        skipped = sum(1 for t in self._tasks.values() if t.status == TaskStatus.SKIPPED)

        total_duration = sum(
            t.duration_ms or 0 for t in self._tasks.values()
            if t.duration_ms
        )

        return {
            "total_tasks": len(self._tasks),
            "completed": completed,
            "failed": failed,
            "skipped": skipped,
            "execution_order": self._execution_order,
            "total_duration_ms": total_duration,
            "results": self._results,
        }

    def get_task(self, task_id: str) -> Task | None:
        """Get a task by ID."""
        return self._tasks.get(task_id)

    def clear(self) -> None:
        """Clear all tasks."""
        self._tasks.clear()
        self._results.clear()
        self._execution_order.clear()
