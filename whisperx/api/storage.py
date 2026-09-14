from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

import anyio
from fastapi import UploadFile

from whisperx.api.config import Settings


@asynccontextmanager
async def temporary_audio(upload: UploadFile, settings: Settings) -> AsyncIterator[Path]:
    upload_dir = settings.data_dir / "uploads"
    await anyio.Path(upload_dir).mkdir(parents=True, exist_ok=True)
    suffix = Path(upload.filename or "").suffix.lower()[:16]
    path = upload_dir / f"{uuid4().hex}{suffix}"
    try:
        target = await anyio.open_file(path, "wb")
        try:
            while chunk := await upload.read(settings.upload_chunk_bytes):
                await target.write(chunk)
        finally:
            with anyio.CancelScope(shield=True):
                await target.aclose()
        await upload.close()
        yield path
    finally:
        with anyio.CancelScope(shield=True):
            try:
                await upload.close()
            finally:
                await anyio.Path(path).unlink(missing_ok=True)
