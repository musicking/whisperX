import io
import json
import re

from whisperx.api.audio.schemas import AudioTask, ResponseFormat, TranscriptionResult
from whisperx.api.audio.subtitles import split_subtitles


def render_result(
    result: TranscriptionResult,
    response_format: ResponseFormat,
    *,
    max_line_width: int | None = None,
    max_line_count: int | None = None,
    highlight_words: bool = False,
) -> tuple[bytes, str]:
    if response_format in {ResponseFormat.SRT, ResponseFormat.VTT}:
        result = split_subtitles(result)
        # Preserve natural clause boundaries. Chinese character limits are soft:
        # without a word tokenizer, wrapping a long clause could split a word.
        from whisperx.utils import LANGUAGES_WITHOUT_SPACES

        if result.language in LANGUAGES_WITHOUT_SPACES and result.task != AudioTask.TRANSLATE:
            max_line_width = None
        max_line_count = None
    payload = result.model_dump(mode="json", exclude_none=True)
    if response_format in {ResponseFormat.JSON, ResponseFormat.VERBOSE_JSON}:
        if response_format == ResponseFormat.JSON:
            payload = {"text": result.text}
        return json.dumps(payload, ensure_ascii=False).encode(), "application/json"
    if result.task == AudioTask.TRANSLATE:
        # The API language describes the source audio; writers need the output language.
        payload["language"] = "en"
    # Reuse upstream timestamp formatting, word highlighting and speaker labels.
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
    text = output.getvalue()
    if response_format in {ResponseFormat.SRT, ResponseFormat.VTT}:
        # Remove cue-ending punctuation after layout; retain closing quotes,
        # highlight tags and all timestamps, including punctuation word timings.
        text = re.sub(
            r"[，。！？；：、,.!?;:…—]+(?=(?:[’”\"'）)\]】》]|</u>)*\s*(?:\n\n|$))",
            "",
            text,
        )
    return text.encode(), media_types[extension]
