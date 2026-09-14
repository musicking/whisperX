import logging
from asyncio import Queue
from collections.abc import Callable
from functools import partial
from pathlib import Path
from time import monotonic
from typing import TypeVar

import anyio
from fastapi import Request, UploadFile

from whisperx.api.audio.engine import WhisperXEngine
from whisperx.api.storage import temporary_audio

ResultT = TypeVar("ResultT")
logger = logging.getLogger(__name__)


async def execute(
    context: Request,
    file: UploadFile,
    invoke: Callable[[WhisperXEngine, Path], ResultT],
) -> ResultT:
    """Run an explicit model call with exclusive use of an Engine."""
    engines: Queue[WhisperXEngine] = context.app.state.available_engines
    async with temporary_audio(file, context.app.state.settings) as audio_path:
        waiting_since = monotonic()
        engine = await engines.get()
        try:
            logger.info(
                "request_id=%s wait_ms=%.1f inference started",
                context.state.request_id,
                (monotonic() - waiting_since) * 1000,
            )
            # Wait for synchronous inference to finish before returning the Engine.
            return await anyio.to_thread.run_sync(partial(invoke, engine, audio_path))
        finally:
            # Both successful and failed requests must return their exclusive instance.
            engines.put_nowait(engine)
