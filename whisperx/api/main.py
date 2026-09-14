import logging
from asyncio import Queue
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from time import monotonic
from uuid import uuid4

import anyio
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError

from whisperx.api.audio.engine import WhisperXEngine
from whisperx.api.audio.router import router as audio_router
from whisperx.api.config import Settings, get_settings
from whisperx.api.exceptions import APIError, api_error_handler, validation_error_handler
from whisperx.api.logs import file_logging
from whisperx.api.system.router import router as system_router


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        await anyio.Path(resolved.data_dir).mkdir(parents=True, exist_ok=True)
        app.state.settings = resolved
        with file_logging(resolved):
            # Each active request exclusively owns an Engine and its mutable model state.
            engines = [WhisperXEngine(resolved) for _ in range(resolved.inference_concurrency)]
            app.state.available_engines = Queue(maxsize=len(engines))
            for engine in engines:
                app.state.available_engines.put_nowait(engine)
            logger = logging.getLogger(__name__)
            try:
                if resolved.preload_model:
                    for engine in engines:
                        await anyio.to_thread.run_sync(engine.preload)
                logger.info("API started inference_concurrency=%s", len(engines))
                yield
            finally:
                for engine in engines:
                    await anyio.to_thread.run_sync(engine.unload)
                logger.info("API stopped")

    app = FastAPI(
        title="WhisperX API",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs" if resolved.docs_enabled else None,
        openapi_url="/openapi.json" if resolved.docs_enabled else None,
        redoc_url=None,
    )
    app.add_exception_handler(APIError, api_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request.state.request_id = request.headers.get("X-Request-ID") or str(uuid4())
        started = monotonic()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        logging.getLogger(__name__).info(
            "request_id=%s method=%s path=%s status=%s elapsed_ms=%.1f",
            request.state.request_id,
            request.method,
            request.url.path,
            response.status_code,
            (monotonic() - started) * 1000,
        )
        return response

    app.add_exception_handler(Exception, api_error_handler)

    app.include_router(audio_router)
    app.include_router(system_router)
    return app


app = create_app()
