import io
import json

from whisperx.api.audio.schemas import ResponseFormat, TranscriptionResult


def render_result(
    result: TranscriptionResult,
    response_format: ResponseFormat,
    *,
    max_line_width: int | None = None,
    max_line_count: int | None = None,
    highlight_words: bool = False,
) -> tuple[bytes, str]:
    payload = result.model_dump(mode="json", exclude_none=True)
    if response_format in {ResponseFormat.JSON, ResponseFormat.VERBOSE_JSON}:
        if response_format == ResponseFormat.JSON:
            payload = {"text": result.text}
        return json.dumps(payload, ensure_ascii=False).encode(), "application/json"
    # Reuse the upstream writers, including their subtitle layout and speaker labels.
    from whisperx.utils import get_writer

    extension = "txt" if response_format == ResponseFormat.TEXT else response_format.value
    writer = get_writer(extension, ".")
    output = io.StringIO()
    writer.write_result(
        payload,
        output,
        {
            "max_line_width": max_line_width,
            "max_line_count": max_line_count,
            "highlight_words": highlight_words,
        },
    )
    media_types = {
        "txt": "text/plain",
        "srt": "application/x-subrip",
        "vtt": "text/vtt",
        "tsv": "text/tab-separated-values",
    }
    return output.getvalue().encode(), media_types[extension]
