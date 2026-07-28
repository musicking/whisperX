from typing import Annotated

from fastapi import Depends, Request
from fastapi.concurrency import run_in_threadpool

from whisperx_api.exceptions import ResourceNotFound
from whisperx_api.jobs.repository import JobRecord, JobRepository


async def get_job_repository(request: Request) -> JobRepository:
    return request.app.state.job_repository


JobRepositoryDependency = Annotated[JobRepository, Depends(get_job_repository)]


async def valid_job_id(
    job_id: str,
    repository: JobRepositoryDependency,
) -> JobRecord:
    record = await run_in_threadpool(repository.get, job_id)
    if record is None:
        raise ResourceNotFound(f"Job '{job_id}' was not found.")
    return record


JobDependency = Annotated[JobRecord, Depends(valid_job_id)]
