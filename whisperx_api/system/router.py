from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version

from fastapi import APIRouter, Response, status
from fastapi.concurrency import run_in_threadpool
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from whisperx_api.audio.schemas import AudioTask, ResponseFormat
from whisperx_api.dependencies import APIKeyDependency, SettingsDependency
from whisperx_api.jobs.dependencies import JobRepositoryDependency
from whisperx_api.system.schemas import (
    CapabilitiesResponse,
    HealthResponse,
    ModelInfo,
    ModelListResponse,
    VersionResponse,
    WorkerStatus,
)

router = APIRouter(tags=["System"])


async def _worker_status(repository: JobRepositoryDependency) -> WorkerStatus:
    state = await run_in_threadpool(repository.get_state, "worker_heartbeat")
    if state is None:
        return WorkerStatus(connected=False)
    worker_id, updated_at = state
    age_seconds = (datetime.now(timezone.utc) - updated_at).total_seconds()
    return WorkerStatus(
        connected=age_seconds < 30,
        worker_id=worker_id,
        last_heartbeat_at=updated_at,
    )


@router.get("/health/live", response_model=HealthResponse, summary="Liveness probe")
async def liveness() -> dict:
    return {"status": "ok"}


@router.get(
    "/health/ready",
    response_model=HealthResponse,
    summary="Readiness probe",
    responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"model": HealthResponse}},
)
async def readiness(
    settings: SettingsDependency,
    repository: JobRepositoryDependency,
    response: Response,
) -> dict:
    database_ok = await run_in_threadpool(settings.database_path.exists)
    worker = await _worker_status(repository)
    checks = {
        "database": database_ok,
        "worker": worker.connected,
    }
    ready = database_ok and (worker.connected or not settings.require_worker_for_readiness)
    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {"status": "ready" if ready else "not_ready", "checks": checks}


@router.get(
    "/v1/capabilities",
    response_model=CapabilitiesResponse,
    summary="Discover enabled API capabilities",
)
async def capabilities(
    _: APIKeyDependency,
    settings: SettingsDependency,
    repository: JobRepositoryDependency,
) -> dict:
    worker = await _worker_status(repository)
    diarization_enabled = bool(settings.hf_token)
    return {
        "tasks": [task.value for task in AudioTask],
        "response_formats": [item.value for item in ResponseFormat],
        "alignment": True,
        "diarization": diarization_enabled,
        "speaker_embeddings": (diarization_enabled and settings.allow_speaker_embeddings),
        "realtime_streaming": False,
        "max_upload_bytes": settings.max_upload_bytes,
        "models": list(settings.allowed_models),
        "worker": worker.model_dump(mode="json"),
    }


@router.get(
    "/v1/models",
    response_model=ModelListResponse,
    summary="List enabled ASR models",
)
async def models(_: APIKeyDependency, settings: SettingsDependency) -> dict:
    return {
        "object": "list",
        "data": [
            ModelInfo(id=model, default=model == settings.model_name).model_dump()
            for model in settings.allowed_models
        ],
    }


@router.get("/version", response_model=VersionResponse, summary="Get build versions")
async def versions(_: APIKeyDependency, settings: SettingsDependency) -> dict:
    try:
        whisperx_version = version("whisperx")
    except PackageNotFoundError:
        whisperx_version = "source"
    return {
        "service_version": settings.app_version,
        "whisperx_version": whisperx_version,
        "environment": settings.environment.value,
    }


@router.get("/metrics", include_in_schema=False)
async def metrics(_: APIKeyDependency) -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
