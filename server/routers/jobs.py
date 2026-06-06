"""Job status endpoints — Phase R4.

Poll background jobs submitted via server.jobs.JobManager so long tasks don't
hold the request open.
"""

from __future__ import annotations

from fastapi import APIRouter, Header, HTTPException, Request

from server.auth import verify_api_key
from server.jobs import get_job_manager

router = APIRouter()


@router.get("/jobs")
async def list_jobs(
    req: Request,
    x_api_key: str = Header(None),
    authorization: str = Header(None),
) -> dict:
    verify_api_key(req, x_api_key, authorization)
    return {"jobs": get_job_manager().list()}


@router.get("/jobs/{job_id}")
async def get_job(
    job_id: str,
    req: Request,
    x_api_key: str = Header(None),
    authorization: str = Header(None),
) -> dict:
    verify_api_key(req, x_api_key, authorization)
    job = get_job_manager().get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job
