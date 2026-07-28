from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import anyio
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError

from whisperx_api.audio.engine import WhisperXEngine
from whisperx_api.audio.router import router as audio_router
from whisperx_api.config import Settings, get_settings
from whisperx_api.exceptions import APIError, api_error_handler, validation_error_handler
from whisperx_api.jobs.repository import JobRepository
from whisperx_api.jobs.router import router as jobs_router
from whisperx_api.middleware import RequestContextMiddleware
from whisperx_api.system.router import router as system_router


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        resolved_settings.data_dir.mkdir(parents=True, exist_ok=True)
        repository = JobRepository(resolved_settings.database_path)
        await anyio.to_thread.run_sync(repository.initialize)
        engine = WhisperXEngine(resolved_settings)
        app.state.settings = resolved_settings
        app.state.job_repository = repository
        app.state.engine = engine
        if resolved_settings.preload_model:
            await anyio.to_thread.run_sync(engine.preload)
        yield
        await anyio.to_thread.run_sync(engine.unload)

    docs_url = "/docs" if resolved_settings.docs_enabled else None
    openapi_url = "/openapi.json" if resolved_settings.docs_enabled else None
    app = FastAPI(
        title=resolved_settings.app_name,
        version=resolved_settings.app_version,
        description=("High-performance, OpenAI-compatible transcription API powered by WhisperX."),
        docs_url=docs_url,
        redoc_url=None,
        openapi_url=openapi_url,
        lifespan=lifespan,
    )
    app.add_middleware(RequestContextMiddleware)
    app.add_exception_handler(APIError, api_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.include_router(audio_router)
    app.include_router(jobs_router)
    app.include_router(system_router)
    return app


app = create_app()
