"""Match an ordered narration script to the ASR segments used for alignment."""

from bisect import bisect_left
from difflib import SequenceMatcher
from unicodedata import normalize

from whisperx.api.audio.schemas import AlignmentSegment, TranscriptionResult
from whisperx.api.exceptions import UnsupportedOption


def _matching_text(text: str) -> tuple[str, list[int]]:
    # Normalize only the comparison text; offsets always point into the original.
    characters = []
    offsets = []
    for offset, character in enumerate(text):
        for normalized in normalize("NFKC", character).casefold():
            if normalized.isalnum():
                characters.append(normalized)
                offsets.append(offset)
    return "".join(characters), offsets


def match_document(document: str, segments: list[AlignmentSegment]) -> list[AlignmentSegment]:
    reference, offsets = _matching_text(document)
    if not reference:
        raise UnsupportedOption("document_text must contain spoken text")
    recognized_parts = [_matching_text(segment.text)[0] for segment in segments]
    recognized = "".join(recognized_parts)
    matcher = SequenceMatcher(None, recognized, reference, autojunk=False)
    blocks = matcher.get_matching_blocks()
    matched = sum(block.size for block in blocks)
    if not recognized or matcher.ratio() < 0.65 or matched / len(reference) < 0.8:
        raise UnsupportedOption("document_text does not sufficiently match the audio transcript")

    # Equal characters anchor script boundaries. Timing comes from ASR, never from
    # distributing the script evenly across the audio duration.
    anchors = {block.a + index: block.b + index for block in blocks for index in range(block.size)}
    positions = list(anchors)
    result = []
    recognized_start = 0
    document_start = 0
    for index, (segment, part) in enumerate(zip(segments, recognized_parts, strict=True)):
        recognized_end = recognized_start + len(part)
        left = bisect_left(positions, recognized_start)
        right = bisect_left(positions, recognized_end)
        if not part or (right - left) / len(part) < 0.5:
            raise UnsupportedOption(
                "Cannot reliably locate a transcript segment in document_text",
                details={"segment": index, "text": segment.text},
            )
        if index == len(segments) - 1:
            document_end = len(document)
        elif right < len(positions):
            document_end = offsets[anchors[positions[right]]]
        else:
            raise UnsupportedOption("Cannot locate the remaining document_text in the audio")
        original_text = document[document_start:document_end]
        if not _matching_text(original_text)[0]:
            raise UnsupportedOption("Cannot determine a nonempty document segment")
        result.append(segment.model_copy(update={"text": original_text}))
        recognized_start = recognized_end
        document_start = document_end
    return result


def validate_document_alignment(document: str, result: TranscriptionResult) -> None:
    reference, _ = _matching_text(document)
    aligned_text, _ = _matching_text("".join(segment.text for segment in result.segments))
    if aligned_text != reference or not result.segments:
        raise UnsupportedOption("Alignment did not preserve all document_text")
    for index, segment in enumerate(result.segments):
        text, _ = _matching_text(segment.text)
        timed_text, _ = _matching_text(
            "".join(
                word.word
                for word in segment.words or []
                if word.start is not None and word.end is not None and word.end > word.start
            )
        )
        # Unsupported characters (such as digits) may have no word timestamp.
        # Allow a small gap, but reject native fallback segments or mostly untimed text.
        if text and len(timed_text) / len(text) < 0.8:
            raise UnsupportedOption(
                "Unable to align enough document_text to the audio", details={"segment": index}
            )
