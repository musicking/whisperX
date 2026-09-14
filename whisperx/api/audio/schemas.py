from typing import Any, TypeVar

from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, ValidationError, model_validator

from whisperx.api.enums import StrEnum
from whisperx.api.models import APIModel

RequestT = TypeVar("RequestT", bound=BaseModel)


def parse_request(model: type[RequestT], value: dict[str, Any] | str) -> RequestT:
    """Translate validation errors only at the HTTP input boundary."""
    try:
        return (
            model.model_validate_json(value)
            if isinstance(value, str)
            else model.model_validate(value)
        )
    except ValidationError as exc:
        raise RequestValidationError(exc.errors()) from exc


class AudioTask(StrEnum):
    TRANSCRIBE = "transcribe"
    TRANSLATE = "translate"
    ALIGN = "align"
    DIARIZE = "diarize"


class ResponseFormat(StrEnum):
    JSON = "json"
    VERBOSE_JSON = "verbose_json"
    TEXT = "text"
    SRT = "srt"
    VTT = "vtt"
    TSV = "tsv"


class Word(APIModel):
    word: str
    start: float | None = None
    end: float | None = None
    score: float | None = None
    speaker: str | None = None


class Segment(APIModel):
    id: int
    start: float
    end: float
    text: str
    speaker: str | None = None
    avg_logprob: float | None = None
    words: list[Word] | None = None
    chars: list[dict[str, Any]] | None = None


class TranscriptionResult(APIModel):
    task: AudioTask = AudioTask.TRANSCRIBE
    language: str
    duration: float | None = None
    text: str
    segments: list[Segment]
    word_segments: list[Word] | None = None
    speaker_embeddings: dict[str, list[float]] | None = None
    warnings: list[str] = Field(default_factory=list)


class PipelineOptions(APIModel):
    model: str | None = None
    language: str | None = Field(default=None, min_length=2, max_length=32)
    task: AudioTask = AudioTask.TRANSCRIBE
    align: bool = True
    diarize: bool = False
    return_char_alignments: bool = False
    return_speaker_embeddings: bool = False
    min_speakers: int | None = Field(default=None, ge=1, le=100)
    max_speakers: int | None = Field(default=None, ge=1, le=100)
    initial_prompt: str | None = Field(default=None, max_length=4000)
    hotwords: str | None = Field(default=None, max_length=4000)
    temperature: float = Field(default=0, ge=0, le=1)
    batch_size: int | None = Field(default=None, ge=1, le=128)
    chunk_size: int = Field(default=30, ge=1, le=120)

    @model_validator(mode="after")
    def validate_pipeline(self) -> "PipelineOptions":
        if self.task == AudioTask.TRANSLATE and self.align:
            raise ValueError("Translation cannot be combined with forced alignment.")
        if self.return_char_alignments and not self.align:
            raise ValueError("Character alignments require align=true.")
        if self.return_speaker_embeddings and not self.diarize:
            raise ValueError("Speaker embeddings require diarize=true.")
        if (
            self.min_speakers is not None
            and self.max_speakers is not None
            and self.min_speakers > self.max_speakers
        ):
            raise ValueError("min_speakers cannot exceed max_speakers.")
        return self


class AlignmentSegment(APIModel):
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    text: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_times(self) -> "AlignmentSegment":
        if self.end <= self.start:
            raise ValueError("end must be greater than start")
        return self


class AlignmentRequest(APIModel):
    language: str = Field(min_length=2, max_length=32)
    segments: list[AlignmentSegment] = Field(min_length=1)
    return_char_alignments: bool = False


class LanguageResult(APIModel):
    language: str


class DiarizationOptions(APIModel):
    num_speakers: int | None = Field(default=None, ge=1, le=100)
    min_speakers: int | None = Field(default=None, ge=1, le=100)
    max_speakers: int | None = Field(default=None, ge=1, le=100)
    return_embeddings: bool = False

    @model_validator(mode="after")
    def validate_speakers(self) -> "DiarizationOptions":
        if (
            self.min_speakers is not None
            and self.max_speakers is not None
            and self.min_speakers > self.max_speakers
        ):
            raise ValueError("min_speakers cannot exceed max_speakers")
        if self.num_speakers is not None:
            if self.min_speakers is not None and self.num_speakers < self.min_speakers:
                raise ValueError("num_speakers cannot be less than min_speakers")
            if self.max_speakers is not None and self.num_speakers > self.max_speakers:
                raise ValueError("num_speakers cannot exceed max_speakers")
        return self


class DiarizationTurn(APIModel):
    start: float
    end: float
    speaker: str


class DiarizationResult(APIModel):
    turns: list[DiarizationTurn]
    speaker_embeddings: dict[str, list[float]] | None = None
