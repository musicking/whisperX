from whisperx.api.models import APIModel


class HealthResponse(APIModel):
    status: str


class ModelInfo(APIModel):
    id: str
    object: str = "model"
    owned_by: str = "whisperx"
    default: bool


class ModelListResponse(APIModel):
    object: str = "list"
    data: list[ModelInfo]


class CapabilitiesResponse(APIModel):
    tasks: list[str]
    response_formats: list[str]
    models: list[str]
    alignment: bool = True
    diarization: bool
    speaker_embeddings: bool
    realtime_streaming: bool = False
    document_subtitles: bool = False
