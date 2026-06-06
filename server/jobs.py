"""Async Job Manager — Phase R4.

In-process background jobs so long-running work (full-repo index, batch ops)
doesn't block the request / hold an SSE slot. Submit a coroutine, get a job_id,
poll status via /jobs/{id}. Best-effort, single-instance (for distributed,
back with Redis/Celery later).
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable

logger = logging.getLogger("server.jobs")

JOB_TTL_SECONDS = 3600
MAX_JOBS = 1000


@dataclass
class Job:
    id: str
    name: str
    status: str = "pending"          # pending | running | completed | failed
    result: Any = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
        }


class JobManager:
    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._tasks: dict[str, asyncio.Task] = {}

    def submit(self, name: str, coro: Awaitable) -> str:
        """Schedule a coroutine to run in the background; returns a job id."""
        self._cleanup()
        job_id = uuid.uuid4().hex[:16]
        job = Job(id=job_id, name=name)
        self._jobs[job_id] = job
        self._tasks[job_id] = asyncio.create_task(self._run(job, coro))
        logger.info("job %s submitted (%s)", job_id, name)
        return job_id

    async def _run(self, job: Job, coro: Awaitable) -> None:
        job.status = "running"
        try:
            job.result = await coro
            job.status = "completed"
        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            logger.warning("job %s failed: %s", job.id, e)
        finally:
            job.finished_at = time.time()
            self._tasks.pop(job.id, None)

    def get(self, job_id: str) -> dict | None:
        job = self._jobs.get(job_id)
        return job.to_dict() if job else None

    def list(self, limit: int = 50) -> list[dict]:
        jobs = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
        return [j.to_dict() for j in jobs[:limit]]

    def _cleanup(self) -> None:
        now = time.time()
        stale = [
            jid for jid, j in self._jobs.items()
            if j.finished_at and now - j.finished_at > JOB_TTL_SECONDS
        ]
        for jid in stale:
            self._jobs.pop(jid, None)
        # bound memory
        if len(self._jobs) > MAX_JOBS:
            oldest = sorted(self._jobs.values(), key=lambda j: j.created_at)
            for j in oldest[: len(self._jobs) - MAX_JOBS]:
                self._jobs.pop(j.id, None)


_manager: JobManager | None = None


def get_job_manager() -> JobManager:
    global _manager
    if _manager is None:
        _manager = JobManager()
    return _manager


def reset_job_manager() -> None:
    global _manager
    _manager = JobManager()
