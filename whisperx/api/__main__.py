import uvicorn

from whisperx.api.config import get_settings


def run() -> None:
    settings = get_settings()
    uvicorn.run(
        "whisperx.api.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
        workers=1,
    )


if __name__ == "__main__":
    run()
