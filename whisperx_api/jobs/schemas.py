from datetime import datetime

from pydantic import Field

from whisperx_api.audio.schemas import PipelineOptions, ResponseFormat, TranscriptionResult
from whisperx_api.enums import StrEnum
from whisperx_api.models import APIModel


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    EXPIRED = "expired"

    @property
    def terminal(self) -> bool:
        return self in {
            JobStatus.SUCCEEDED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
            JobStatus.EXPIRED,
        }


class JobTask(StrEnum):
    TRANSCRIBE = "transcribe"
    TRANSLATE = "translate"


class JobStage(StrEnum):
    QUEUED = "queued"
    DECODE = "decode"
    VAD = "vad"
    TRANSCRIBE = "transcribe"
    ALIGN = "align"
    DIARIZE = "diarize"
    EXPORT = "export"
    COMPLETE = "complete"


class JobError(APIModel):
    code: str
    message: str


class JobLinks(APIModel):
    self: str
    events: str
    result: str


class JobResponse(APIModel):
    id: str
    status: JobStatus
    stage: JobStage
    progress: float = Field(ge=0, le=100)
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: JobError | None = None
    links: JobLinks


class JobListResponse(APIModel):
    data: list[JobResponse]
    limit: int
    offset: int


class JobCreate(APIModel):
    options: PipelineOptions
    response_format: ResponseFormat = ResponseFormat.VERBOSE_JSON


class JobResultResponse(APIModel):
    id: str
    result: TranscriptionResult
