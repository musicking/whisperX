from typing import Annotated

import anyio
from fastapi import APIRouter, File, Form, Response, UploadFile, status

from whisperx_api.audio import service
from whisperx_api.audio.dependencies import EngineDependency
from whisperx_api.audio.renderers import render_result
from whisperx_api.audio.schemas import (
    AudioTask,
    DiarizationResult,
    PipelineOptions,
    ResponseFormat,
    TranscriptionResult,
)
from whisperx_api.dependencies import APIKeyDependency, SettingsDependency
from whisperx_api.exceptions import ErrorResponse, UnsupportedOption
from whisperx_api.jobs.dependencies import JobRepositoryDependency
from whisperx_api.storage import store_upload

router = APIRouter(
    prefix="/v1/audio",
    tags=["Audio"],
    dependencies=[],
)


@router.post(
    "/transcriptions",
    summary="Transcribe audio",
    description=(
        "OpenAI-compatible audio transcription with optional WhisperX alignment and diarization."
    ),
    responses={
        status.HTTP_200_OK: {
            "content": {
                "application/json": {},
                "text/plain": {},
                "application/x-subrip": {},
                "text/vtt": {},
            }
        },
        413: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ErrorResponse},
    },
)
async def create_transcription(
    _: APIKeyDependency,
    engine: EngineDependency,
    settings: SettingsDependency,
    repository: JobRepositoryDependency,
    file: Annotated[UploadFile, File(description="Audio or video file")],
    model: Annotated[str | None, Form()] = None,
    language: Annotated[str | None, Form()] = None,
    prompt: Annotated[str | None, Form()] = None,
    response_format: Annotated[ResponseFormat, Form()] = ResponseFormat.JSON,
    temperature: Annotated[float, Form(ge=0, le=1)] = 0,
    timestamp_granularities: Annotated[list[str] | None, Form()] = None,
    timestamp_granularities_bracketed: Annotated[
        list[str] | None,
        Form(alias="timestamp_granularities[]"),
    ] = None,
    align: Annotated[bool | None, Form()] = None,
    diarize: Annotated[bool, Form()] = False,
    min_speakers: Annotated[int | None, Form(ge=1, le=100)] = None,
    max_speakers: Annotated[int | None, Form(ge=1, le=100)] = None,
    return_char_alignments: Annotated[bool, Form()] = False,
    return_speaker_embeddings: Annotated[bool, Form()] = False,
    hotwords: Annotated[str | None, Form()] = None,
) -> Response:
    granularities = set(timestamp_granularities or [])
    granularities.update(timestamp_granularities_bracketed or [])
    unknown = granularities - {"segment", "word"}
    if unknown:
        raise UnsupportedOption(f"Unsupported timestamp granularities: {sorted(unknown)}")
    should_align = align if align is not None else ("word" in granularities)
    options = PipelineOptions(
        model=model,
        language=language,
        task=AudioTask.TRANSCRIBE,
        align=should_align,
        diarize=diarize,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        return_char_alignments=return_char_alignments,
        return_speaker_embeddings=return_speaker_embeddings,
        initial_prompt=prompt,
        hotwords=hotwords,
        temperature=temperature,
    )
    upload = await store_upload(file, settings)
    if settings.sync_via_worker:
        result = await service.transcribe_via_job(
            repository,
            upload.path,
            options,
            response_format=response_format.value,
            timeout_seconds=settings.sync_timeout_seconds,
            poll_seconds=settings.sync_poll_seconds,
        )
        body, media_type = render_result(result, response_format)
        return Response(content=body, media_type=media_type)
    try:
        result = await service.transcribe(engine, upload.path, options)
    finally:
        await anyio.Path(upload.path).unlink(missing_ok=True)
    body, media_type = render_result(result, response_format)
    return Response(content=body, media_type=media_type)


@router.post(
    "/translations",
    summary="Translate audio to English",
    description=(
        "OpenAI-compatible speech translation. Forced alignment is unavailable for translation."
    ),
    responses={
        status.HTTP_200_OK: {"description": "Translation result"},
        422: {"model": ErrorResponse},
    },
)
async def create_translation(
    _: APIKeyDependency,
    engine: EngineDependency,
    settings: SettingsDependency,
    repository: JobRepositoryDependency,
    file: Annotated[UploadFile, File(description="Audio or video file")],
    model: Annotated[str | None, Form()] = None,
    language: Annotated[str | None, Form()] = None,
    prompt: Annotated[str | None, Form()] = None,
    temperature: Annotated[float, Form(ge=0, le=1)] = 0,
    response_format: Annotated[ResponseFormat, Form()] = ResponseFormat.JSON,
) -> Response:
    options = PipelineOptions(
        model=model,
        language=language,
        task=AudioTask.TRANSLATE,
        align=False,
        initial_prompt=prompt,
        temperature=temperature,
    )
    upload = await store_upload(file, settings)
    if settings.sync_via_worker:
        result = await service.transcribe_via_job(
            repository,
            upload.path,
            options,
            response_format=response_format.value,
            timeout_seconds=settings.sync_timeout_seconds,
            poll_seconds=settings.sync_poll_seconds,
        )
        body, media_type = render_result(result, response_format)
        return Response(content=body, media_type=media_type)
    try:
        result = await service.transcribe(engine, upload.path, options)
    finally:
        await anyio.Path(upload.path).unlink(missing_ok=True)
    body, media_type = render_result(result, response_format)
    return Response(content=body, media_type=media_type)


@router.post(
    "/alignments",
    response_model=TranscriptionResult,
    summary="Align an existing transcript to audio",
    status_code=status.HTTP_200_OK,
)
async def create_alignment(
    _: APIKeyDependency,
    engine: EngineDependency,
    settings: SettingsDependency,
    file: Annotated[UploadFile, File(description="Audio or video file")],
    request: Annotated[
        str,
        Form(
            description=(
                'JSON object: {"language":"en","segments":[{"start":0,"end":10,"text":"..."}]}'
            )
        ),
    ],
) -> dict:
    upload = await store_upload(file, settings)
    try:
        result = await service.align(engine, upload.path, request)
        return result.model_dump(mode="json", exclude_none=True)
    finally:
        await anyio.Path(upload.path).unlink(missing_ok=True)


@router.post(
    "/diarizations",
    response_model=DiarizationResult,
    summary="Identify anonymous speaker turns",
    status_code=status.HTTP_200_OK,
)
async def create_diarization(
    _: APIKeyDependency,
    engine: EngineDependency,
    settings: SettingsDependency,
    file: Annotated[UploadFile, File(description="Audio or video file")],
    min_speakers: Annotated[int | None, Form(ge=1, le=100)] = None,
    max_speakers: Annotated[int | None, Form(ge=1, le=100)] = None,
    return_speaker_embeddings: Annotated[bool, Form()] = False,
) -> dict:
    upload = await store_upload(file, settings)
    try:
        result = await service.diarize(
            engine,
            upload.path,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            return_embeddings=return_speaker_embeddings,
        )
        return result.model_dump(mode="json", exclude_none=True)
    finally:
        await anyio.Path(upload.path).unlink(missing_ok=True)
