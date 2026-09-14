from fastapi import APIRouter

from whisperx.api.audio.schemas import AudioTask, ResponseFormat
from whisperx.api.dependencies import SettingsDependency
from whisperx.api.system.schemas import CapabilitiesResponse, HealthResponse, ModelListResponse

router = APIRouter(tags=["System"])


@router.get("/health/live", response_model=HealthResponse)
async def live():
    return {"status": "ok"}


@router.get("/health", response_model=HealthResponse)
async def ready(settings: SettingsDependency):
    # Lifespan has initialized the engine. Models load lazily unless preload_model is enabled.
    return {"status": "ready"}


@router.get("/v1/models", response_model=ModelListResponse)
async def models(settings: SettingsDependency):
    return {
        "object": "list",
        "data": [
            {"id": model, "default": model == settings.model_name}
            for model in settings.allowed_models
        ],
    }


@router.get("/v1/capabilities", response_model=CapabilitiesResponse)
async def capabilities(settings: SettingsDependency):
    return {
        "tasks": [task.value for task in AudioTask] + ["language"],
        "response_formats": [format.value for format in ResponseFormat],
        "models": list(settings.allowed_models),
        "diarization": bool(settings.hf_token),
        "speaker_embeddings": bool(settings.hf_token) and settings.allow_speaker_embeddings,
        "document_subtitles": True,
    }
