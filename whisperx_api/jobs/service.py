from whisperx_api.jobs.repository import JobRecord
from whisperx_api.jobs.schemas import JobError, JobLinks, JobResponse


def serialize_job(record: JobRecord) -> dict:
    error = None
    if record.error_code and record.error_message:
        error = JobError(code=record.error_code, message=record.error_message)
    response = JobResponse(
        id=record.id,
        status=record.status,
        stage=record.stage,
        progress=record.progress,
        created_at=record.created_at,
        updated_at=record.updated_at,
        started_at=record.started_at,
        finished_at=record.finished_at,
        error=error,
        links=JobLinks(
            self=f"/v1/jobs/{record.id}",
            events=f"/v1/jobs/{record.id}/events",
            result=f"/v1/jobs/{record.id}/result",
        ),
    )
    return response.model_dump(mode="json", exclude_none=True)
