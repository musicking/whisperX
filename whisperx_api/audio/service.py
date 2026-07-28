import json
from pathlib import Path
from time import monotonic

import anyio
from fastapi.concurrency import run_in_threadpool

from whisperx_api.audio.engine import SpeechEngine
from whisperx_api.audio.schemas import (
    AlignmentRequest,
    DiarizationResult,
    PipelineOptions,
    TranscriptionResult,
)
from whisperx_api.exceptions import Conflict, InferenceTimeout, ModelUnavailable, UnsupportedOption
from whisperx_api.jobs.repository import JobRepository
from whisperx_api.jobs.schemas import JobStatus


async def transcribe(
    engine: SpeechEngine,
    audio_path: Path,
    options: PipelineOptions,
) -> TranscriptionResult:
    return await run_in_threadpool(engine.transcribe, audio_path, options)


async def transcribe_via_job(
    repository: JobRepository,
    audio_path: Path,
    options: PipelineOptions,
    *,
    response_format: str,
    timeout_seconds: float,
    poll_seconds: float,
) -> TranscriptionResult:
    try:
        record = await run_in_threadpool(
            repository.create,
            input_path=audio_path,
            request={
                "options": options.model_dump(mode="json"),
                "response_format": response_format,
                "source": "synchronous_api",
            },
        )
    except BaseException:
        await anyio.Path(audio_path).unlink(missing_ok=True)
        raise
    deadline = monotonic() + timeout_seconds
    while monotonic() < deadline:
        current = await run_in_threadpool(repository.get, record.id)
        assert current is not None
        if current.status == JobStatus.SUCCEEDED and current.result is not None:
            return TranscriptionResult.model_validate(current.result)
        if current.status == JobStatus.FAILED:
            raise ModelUnavailable(
                current.error_message or "Inference failed.",
                details={"job_id": current.id, "error_code": current.error_code},
            )
        if current.status in {JobStatus.CANCELLED, JobStatus.EXPIRED}:
            raise Conflict(
                f"Inference job entered terminal status '{current.status.value}'.",
                details={"job_id": current.id},
            )
        await anyio.sleep(poll_seconds)
    raise InferenceTimeout(details={"job_id": record.id})


async def align(
    engine: SpeechEngine,
    audio_path: Path,
    request_json: str,
) -> TranscriptionResult:
    try:
        request = AlignmentRequest.model_validate(json.loads(request_json))
    except (json.JSONDecodeError, ValueError) as exc:
        raise UnsupportedOption(f"Invalid alignment request: {exc}") from exc
    return await run_in_threadpool(engine.align, audio_path, request)


async def diarize(
    engine: SpeechEngine,
    audio_path: Path,
    *,
    min_speakers: int | None,
    max_speakers: int | None,
    return_embeddings: bool,
) -> DiarizationResult:
    if min_speakers is not None and max_speakers is not None and min_speakers > max_speakers:
        raise UnsupportedOption("min_speakers cannot exceed max_speakers.")
    return await run_in_threadpool(
        engine.diarize,
        audio_path,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        return_embeddings=return_embeddings,
    )
