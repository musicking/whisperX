import json
from collections.abc import Iterable

from whisperx_api.audio.schemas import ResponseFormat, Segment, TranscriptionResult


def _timestamp(seconds: float, *, decimal_marker: str = ",") -> str:
    milliseconds = round(seconds * 1000)
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    whole_seconds, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}{decimal_marker}{milliseconds:03d}"


def _speaker_text(segment: Segment) -> str:
    prefix = f"[{segment.speaker}] " if segment.speaker else ""
    return f"{prefix}{segment.text.strip()}"


def render_result(
    result: TranscriptionResult,
    response_format: ResponseFormat,
) -> tuple[bytes, str]:
    if response_format in {ResponseFormat.JSON, ResponseFormat.VERBOSE_JSON}:
        if response_format == ResponseFormat.JSON:
            payload = {"text": result.text}
        else:
            payload = result.model_dump(mode="json", exclude_none=True)
        return json.dumps(payload, ensure_ascii=False).encode(), "application/json"
    if response_format == ResponseFormat.TEXT:
        return f"{result.text}\n".encode(), "text/plain; charset=utf-8"
    if response_format == ResponseFormat.SRT:
        return _render_srt(result.segments).encode(), "application/x-subrip; charset=utf-8"
    if response_format == ResponseFormat.VTT:
        return _render_vtt(result.segments).encode(), "text/vtt; charset=utf-8"
    if response_format == ResponseFormat.TSV:
        return _render_tsv(result.segments).encode(), "text/tab-separated-values; charset=utf-8"
    raise ValueError(f"Unsupported response format: {response_format}")


def _render_srt(segments: Iterable[Segment]) -> str:
    blocks = []
    for index, segment in enumerate(segments, start=1):
        blocks.append(
            f"{index}\n{_timestamp(segment.start)} --> {_timestamp(segment.end)}\n"
            f"{_speaker_text(segment)}"
        )
    return "\n\n".join(blocks) + "\n"


def _render_vtt(segments: Iterable[Segment]) -> str:
    blocks = ["WEBVTT"]
    for segment in segments:
        blocks.append(
            f"{_timestamp(segment.start, decimal_marker='.')} --> "
            f"{_timestamp(segment.end, decimal_marker='.')}\n{_speaker_text(segment)}"
        )
    return "\n\n".join(blocks) + "\n"


def _render_tsv(segments: Iterable[Segment]) -> str:
    rows = ["start\tend\ttext"]
    rows.extend(
        f"{round(segment.start * 1000)}\t{round(segment.end * 1000)}\t"
        f"{_speaker_text(segment).replace(chr(9), ' ')}"
        for segment in segments
    )
    return "\n".join(rows) + "\n"
