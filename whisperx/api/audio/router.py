from typing import Annotated

from fastapi import APIRouter, File, Form, Request, Response, UploadFile

from whisperx.api.audio import service
from whisperx.api.audio.renderers import render_result
from whisperx.api.audio.schemas import (
    AlignmentRequest,
    AudioTask,
    DiarizationOptions,
    DiarizationResult,
    LanguageResult,
    PipelineOptions,
    ResponseFormat,
    TranscriptionResult,
    parse_request,
)
from whisperx.api.exceptions import UnsupportedOption

router = APIRouter(prefix="/v1/audio", tags=["Audio"])
AudioFile = Annotated[UploadFile, File(description="Audio or video file")]


async def _render(result, response_format: ResponseFormat, **layout):
    from functools import partial

    import anyio

    body, media_type = await anyio.to_thread.run_sync(
        partial(render_result, result, response_format, **layout)
    )
    return Response(body, media_type=media_type)


@router.post("/transcriptions", summary="Transcribe audio")
async def transcriptions(
    request: Request,
    file: AudioFile,
    model: Annotated[str | None, Form()] = None,
    language: Annotated[str | None, Form()] = None,
    prompt: Annotated[str | None, Form(max_length=4000)] = None,
    hotwords: Annotated[str | None, Form(max_length=4000)] = None,
    temperature: Annotated[float, Form(ge=0, le=1)] = 0,
    response_format: Annotated[ResponseFormat, Form()] = ResponseFormat.JSON,
    timestamp_granularities: Annotated[list[str] | None, Form()] = None,
    timestamp_granularities_bracketed: Annotated[
        list[str] | None, Form(alias="timestamp_granularities[]")
    ] = None,
    align: Annotated[bool | None, Form()] = None,
    diarize: Annotated[bool, Form()] = False,
    min_speakers: Annotated[int | None, Form(ge=1, le=100)] = None,
    max_speakers: Annotated[int | None, Form(ge=1, le=100)] = None,
    return_char_alignments: Annotated[bool, Form()] = False,
    return_speaker_embeddings: Annotated[bool, Form()] = False,
    batch_size: Annotated[int | None, Form(ge=1, le=128)] = None,
    chunk_size: Annotated[int, Form(ge=1, le=120)] = 30,
) -> Response:
    granularities = set(timestamp_granularities or []) | set(
        timestamp_granularities_bracketed or []
    )
    if granularities - {"word", "segment"}:
        raise UnsupportedOption("timestamp_granularities must be word or segment")
    if granularities and response_format != ResponseFormat.VERBOSE_JSON:
        raise UnsupportedOption("timestamp_granularities requires response_format=verbose_json")
    if "word" in granularities and align is False:
        raise UnsupportedOption("Word timestamps require align=true")
    options = parse_request(
        PipelineOptions,
        dict(
            model=model,
            language=language,
            initial_prompt=prompt,
            hotwords=hotwords,
            temperature=temperature,
            align=align if align is not None else "word" in granularities,
            diarize=diarize,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            return_char_alignments=return_char_alignments,
            return_speaker_embeddings=return_speaker_embeddings,
            batch_size=batch_size,
            chunk_size=chunk_size,
        ),
    )
    result = await service.execute(
        request, file, lambda engine, path: engine.transcribe(path, options)
    )
    return await _render(result, response_format)


@router.post("/translations", summary="Translate speech into English")
async def translations(
    request: Request,
    file: AudioFile,
    model: Annotated[str | None, Form()] = None,
    language: Annotated[str | None, Form()] = None,
    prompt: Annotated[str | None, Form(max_length=4000)] = None,
    temperature: Annotated[float, Form(ge=0, le=1)] = 0,
    response_format: Annotated[ResponseFormat, Form()] = ResponseFormat.JSON,
    batch_size: Annotated[int | None, Form(ge=1, le=128)] = None,
    chunk_size: Annotated[int, Form(ge=1, le=120)] = 30,
) -> Response:
    options = parse_request(
        PipelineOptions,
        dict(
            model=model,
            language=language,
            initial_prompt=prompt,
            temperature=temperature,
            task=AudioTask.TRANSLATE,
            align=False,
            batch_size=batch_size,
            chunk_size=chunk_size,
        ),
    )
    result = await service.execute(
        request, file, lambda engine, path: engine.transcribe(path, options)
    )
    return await _render(result, response_format)


@router.post("/alignments", response_model=TranscriptionResult, summary="Align existing text")
async def alignments(
    http_request: Request,
    file: AudioFile,
    request: Annotated[
        str,
        Form(
            description=('JSON: {"language":"en","segments":[{"start":0,"end":10,"text":"Hello"}]}')
        ),
    ],
) -> TranscriptionResult:
    alignment = parse_request(AlignmentRequest, request)
    return await service.execute(
        http_request, file, lambda engine, path: engine.align(path, alignment)
    )


@router.post("/diarizations", response_model=DiarizationResult, summary="Detect speaker turns")
async def diarizations(
    request: Request,
    file: AudioFile,
    num_speakers: Annotated[int | None, Form(ge=1, le=100)] = None,
    min_speakers: Annotated[int | None, Form(ge=1, le=100)] = None,
    max_speakers: Annotated[int | None, Form(ge=1, le=100)] = None,
    return_speaker_embeddings: Annotated[bool, Form()] = False,
) -> DiarizationResult:
    options = parse_request(
        DiarizationOptions,
        dict(
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            return_embeddings=return_speaker_embeddings,
        ),
    )
    return await service.execute(
        request,
        file,
        lambda engine, path: engine.diarize(
            path,
            num_speakers=options.num_speakers,
            min_speakers=options.min_speakers,
            max_speakers=options.max_speakers,
            return_embeddings=options.return_embeddings,
        ),
    )


@router.post("/subtitles", summary="Generate word-aligned subtitles")
async def subtitles(
    request: Request,
    file: AudioFile,
    model: Annotated[str | None, Form()] = None,
    language: Annotated[str | None, Form()] = None,
    prompt: Annotated[str | None, Form(max_length=4000)] = None,
    hotwords: Annotated[str | None, Form(max_length=4000)] = None,
    temperature: Annotated[float, Form(ge=0, le=1)] = 0,
    response_format: Annotated[ResponseFormat, Form()] = ResponseFormat.SRT,
    batch_size: Annotated[int | None, Form(ge=1, le=128)] = None,
    chunk_size: Annotated[int, Form(ge=1, le=120)] = 30,
    max_line_width: Annotated[int | None, Form(ge=1, le=200)] = None,
    max_line_count: Annotated[int | None, Form(ge=1, le=10)] = None,
    highlight_words: Annotated[bool, Form()] = False,
) -> Response:
    form = await request.form()
    if "document" in form or "document_text" in form:
        raise UnsupportedOption("Document-assisted subtitles are not supported in this version")
    if max_line_count is not None and max_line_width is None:
        raise UnsupportedOption("max_line_count requires max_line_width")
    options = parse_request(
        PipelineOptions,
        dict(
            model=model,
            language=language,
            initial_prompt=prompt,
            hotwords=hotwords,
            temperature=temperature,
            align=True,
            batch_size=batch_size,
            chunk_size=chunk_size,
        ),
    )
    result = await service.execute(
        request, file, lambda engine, path: engine.transcribe(path, options)
    )
    return await _render(
        result,
        response_format,
        max_line_width=max_line_width,
        max_line_count=max_line_count,
        highlight_words=highlight_words,
    )


@router.post("/language", response_model=LanguageResult, summary="Detect audio language")
async def language(
    request: Request,
    file: AudioFile,
    model: Annotated[str | None, Form()] = None,
) -> LanguageResult:
    return await service.execute(
        request, file, lambda engine, path: engine.detect_language(path, model=model)
    )
