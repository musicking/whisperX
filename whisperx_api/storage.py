import hashlib
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import anyio
from fastapi import UploadFile

from whisperx_api.config import Settings
from whisperx_api.exceptions import UploadTooLarge


@dataclass(frozen=True, slots=True)
class StoredUpload:
    path: Path
    size: int
    sha256: str
    content_type: str | None
    original_filename: str | None


async def store_upload(upload: UploadFile, settings: Settings) -> StoredUpload:
    upload_dir = settings.data_dir / "uploads"
    await anyio.Path(upload_dir).mkdir(parents=True, exist_ok=True)
    suffix = Path(upload.filename or "").suffix.lower()[:16]
    path = upload_dir / f"{uuid4().hex}{suffix}"
    size = 0
    digest = hashlib.sha256()
    try:
        async with await anyio.open_file(path, "wb") as target:
            while chunk := await upload.read(settings.upload_chunk_bytes):
                size += len(chunk)
                if size > settings.max_upload_bytes:
                    raise UploadTooLarge(details={"max_upload_bytes": settings.max_upload_bytes})
                digest.update(chunk)
                await target.write(chunk)
    except BaseException:
        await anyio.Path(path).unlink(missing_ok=True)
        raise
    finally:
        await upload.close()
    return StoredUpload(
        path=path,
        size=size,
        sha256=digest.hexdigest(),
        content_type=upload.content_type,
        original_filename=upload.filename,
    )
