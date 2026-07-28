from datetime import datetime

from whisperx_api.models import APIModel


class HealthResponse(APIModel):
    status: str
    checks: dict[str, bool] | None = None


class WorkerStatus(APIModel):
    connected: bool
    worker_id: str | None = None
    last_heartbeat_at: datetime | None = None


class CapabilitiesResponse(APIModel):
    tasks: list[str]
    response_formats: list[str]
    alignment: bool
    diarization: bool
    speaker_embeddings: bool
    realtime_streaming: bool
    max_upload_bytes: int
    models: list[str]
    worker: WorkerStatus


class ModelInfo(APIModel):
    id: str
    object: str = "model"
    owned_by: str = "whisperx"
    default: bool = False


class ModelListResponse(APIModel):
    object: str = "list"
    data: list[ModelInfo]


class VersionResponse(APIModel):
    service_version: str
    whisperx_version: str
    environment: str
