import json
from collections.abc import AsyncIterator
from typing import Annotated

import anyio
from fastapi import APIRouter, File, Form, Query, Request, Response, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse

from whisperx_api.audio.renderers import render_result
from whisperx_api.audio.schemas import (
    AudioTask,
    PipelineOptions,
    ResponseFormat,
    TranscriptionResult,
)
from whisperx_api.dependencies import APIKeyDependency, SettingsDependency
from whisperx_api.exceptions import Conflict, ErrorResponse
from whisperx_api.jobs import service
from whisperx_api.jobs.dependencies import (
    JobDependency,
    JobRepositoryDependency,
)
from whisperx_api.jobs.schemas import JobListResponse, JobResponse, JobStatus, JobTask
from whisperx_api.storage import store_upload

router = APIRouter(prefix="/v1/jobs", tags=["Jobs"])


@router.post(
    "",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Create a durable transcription job",
    responses={
        status.HTTP_202_ACCEPTED: {"model": JobResponse},
        413: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
    },
)
async def create_job(
    _: APIKeyDependency,
    repository: JobRepositoryDependency,
    settings: SettingsDependency,
    file: Annotated[UploadFile, File(description="Audio or video file")],
    task: Annotated[JobTask, Form()] = JobTask.TRANSCRIBE,
    model: Annotated[str | None, Form()] = None,
    language: Annotated[str | None, Form()] = None,
    align: Annotated[bool, Form()] = True,
    diarize: Annotated[bool, Form()] = False,
    min_speakers: Annotated[int | None, Form(ge=1, le=100)] = None,
    max_speakers: Annotated[int | None, Form(ge=1, le=100)] = None,
    return_char_alignments: Annotated[bool, Form()] = False,
    return_speaker_embeddings: Annotated[bool, Form()] = False,
    prompt: Annotated[str | None, Form()] = None,
    hotwords: Annotated[str | None, Form()] = None,
    temperature: Annotated[float, Form(ge=0, le=1)] = 0,
    batch_size: Annotated[int | None, Form(ge=1, le=128)] = None,
    chunk_size: Annotated[int, Form(ge=1, le=120)] = 30,
    response_format: Annotated[ResponseFormat, Form()] = ResponseFormat.VERBOSE_JSON,
) -> dict:
    options = PipelineOptions(
        model=model,
        language=language,
        task=AudioTask(task.value),
        align=align,
        diarize=diarize,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        return_char_alignments=return_char_alignments,
        return_speaker_embeddings=return_speaker_embeddings,
        initial_prompt=prompt,
        hotwords=hotwords,
        temperature=temperature,
        batch_size=batch_size,
        chunk_size=chunk_size,
    )
    upload = await store_upload(file, settings)
    request_payload = {
        "options": options.model_dump(mode="json"),
        "response_format": response_format.value,
        "upload": {
            "size": upload.size,
            "sha256": upload.sha256,
            "content_type": upload.content_type,
            "original_filename": upload.original_filename,
        },
    }
    try:
        record = await run_in_threadpool(
            repository.create,
            input_path=upload.path,
            request=request_payload,
        )
    except BaseException:
        await anyio.Path(upload.path).unlink(missing_ok=True)
        raise
    return service.serialize_job(record)


@router.get(
    "",
    response_model=JobListResponse,
    summary="List transcription jobs",
)
async def list_jobs(
    _: APIKeyDependency,
    repository: JobRepositoryDependency,
    job_status: Annotated[JobStatus | None, Query(alias="status")] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> dict:
    records = await run_in_threadpool(
        repository.list,
        status=job_status,
        limit=limit,
        offset=offset,
    )
    return {
        "data": [service.serialize_job(record) for record in records],
        "limit": limit,
        "offset": offset,
    }


@router.get(
    "/{job_id}",
    response_model=JobResponse,
    summary="Get a transcription job",
)
async def get_job(_: APIKeyDependency, job: JobDependency) -> dict:
    return service.serialize_job(job)


@router.delete(
    "/{job_id}",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Request job cancellation",
)
async def cancel_job(
    _: APIKeyDependency,
    job: JobDependency,
    repository: JobRepositoryDependency,
) -> dict:
    record = await run_in_threadpool(repository.request_cancel, job.id)
    assert record is not None
    if record.status == JobStatus.CANCELLED:
        await anyio.Path(job.input_path).unlink(missing_ok=True)
    return service.serialize_job(record)


@router.get(
    "/{job_id}/result",
    summary="Get a completed job result",
    responses={
        status.HTTP_200_OK: {"description": "Transcription or subtitle result"},
        status.HTTP_409_CONFLICT: {"model": ErrorResponse},
    },
)
async def get_job_result(
    _: APIKeyDependency,
    job: JobDependency,
    response_format: Annotated[ResponseFormat | None, Query(alias="format")] = None,
) -> Response:
    if job.status != JobStatus.SUCCEEDED or job.result is None:
        raise Conflict(
            f"Job '{job.id}' has no result in status '{job.status.value}'.",
            details={"status": job.status.value},
        )
    result = TranscriptionResult.model_validate(job.result)
    selected_format = response_format or ResponseFormat(job.request["response_format"])
    body, media_type = render_result(result, selected_format)
    return Response(content=body, media_type=media_type)


@router.get(
    "/{job_id}/events",
    summary="Stream job status events",
    response_class=StreamingResponse,
)
async def stream_job_events(
    _: APIKeyDependency,
    request: Request,
    job: JobDependency,
    repository: JobRepositoryDependency,
) -> StreamingResponse:
    async def event_stream() -> AsyncIterator[str]:
        last_version: tuple[str, str, float] | None = None
        event_id = 0
        while True:
            if await request.is_disconnected():
                return
            record = await run_in_threadpool(repository.get, job.id)
            if record is None:
                return
            version = (record.status.value, record.stage.value, record.progress)
            if version != last_version:
                event_id += 1
                payload = service.serialize_job(record)
                yield (
                    f"id: {event_id}\n"
                    f"event: {'completed' if record.status.terminal else 'progress'}\n"
                    f"data: {json.dumps(payload)}\n\n"
                )
                last_version = version
            if record.status.terminal:
                return
            await anyio.sleep(0.75)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
