import logging
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler

from whisperx.api.config import Settings


@contextmanager
def file_logging(settings: Settings):
    """Own the file handler for the lifetime of this single-process API."""
    settings.log_file.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        settings.log_file,
        maxBytes=settings.log_max_bytes,
        backupCount=settings.log_backup_count,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    level = settings.log_level.upper()
    handler.setLevel(level)
    loggers = [logging.getLogger("whisperx"), logging.getLogger("uvicorn")]
    previous_levels = [logger.level for logger in loggers]
    for logger in loggers:
        logger.setLevel(level)
        logger.addHandler(handler)
    try:
        yield
    finally:
        for logger, previous_level in zip(loggers, previous_levels, strict=True):
            logger.removeHandler(handler)
            logger.setLevel(previous_level)
        handler.close()
