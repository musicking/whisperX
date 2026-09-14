from whisperx.api.audio.schemas import Segment, TranscriptionResult, Word

_BREAKS = frozenset("，。！？；：,.!?;:")
_CLOSERS = frozenset("，。！？；：,.!?;:’”\"'）)]】》")
_QUOTES = "’”\"'）)]】》"


def split_subtitles(result: TranscriptionResult) -> TranscriptionResult:
    """Split aligned subtitles at punctuation, keeping timestamps and speaker labels."""
    segments = []
    for segment in result.segments:
        if not segment.words:
            segments.append(segment)
            continue
        words: list[Word] = []
        for index, word in enumerate(segment.words):
            words.append(word)
            text = "".join(item.word for item in words).rstrip().rstrip(_QUOTES)
            following = (
                segment.words[index + 1].word.lstrip() if index + 1 < len(segment.words) else ""
            )
            if not text or text[-1] not in _BREAKS:
                continue
            # Keep decimal numbers/times intact and attach closing quotes to the clause.
            numeric_separator = (
                text[-1] in ".,:"
                and len(text) > 1
                and text[-2].isdigit()
                and following[:1].isdigit()
            )
            if numeric_separator or following[:1] in _CLOSERS:
                continue
            segments.append(_subtitle_segment(segment, words, len(segments)))
            words = []
        if words:
            segments.append(_subtitle_segment(segment, words, len(segments)))
    return result.model_copy(update={"segments": segments})


def _subtitle_segment(segment: Segment, words: list[Word], index: int) -> Segment:
    starts = [word.start for word in words if word.start is not None]
    ends = [word.end for word in words if word.end is not None]
    return segment.model_copy(
        update={
            "id": index,
            "start": min(starts) if starts else segment.start,
            "end": max(ends) if ends else segment.end,
            "words": words,
            "text": "".join(word.word for word in words),
        }
    )
