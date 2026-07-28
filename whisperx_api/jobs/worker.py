import logging
import os
import socket
import time
from pathlib import Path

from whisperx_api.audio.engine import WhisperXEngine
from whisperx_api.audio.schemas import PipelineOptions
from whisperx_api.config import get_settings
from whisperx_api.exceptions import APIError
from whisperx_api.jobs.repository import JobRecord, JobRepository

logger = logging.getLogger(__name__)


class JobCancelled(Exception):
    pass


class Worker:
    def __init__(
        self,
        repository: JobRepository,
        engine: WhisperXEngine,
        *,
        worker_id: str,
        poll_seconds: float,
    ) -> None:
        self.repository = repository
        self.engine = engine
        self.worker_id = worker_id
        self.poll_seconds = poll_seconds

    def run_forever(self) -> None:
        logger.info("Worker %s started", self.worker_id)
        while True:
            self.repository.heartbeat(self.worker_id)
            record = self.repository.claim_next(self.worker_id)
            if record is None:
                time.sleep(self.poll_seconds)
                continue
            self.process(record)

    def process(self, record: JobRecord) -> None:
        last_progress = -1.0

        def report(stage: str, progress: float) -> None:
            nonlocal last_progress
            current = self.repository.get(record.id)
            if current is None or current.cancel_requested:
                raise JobCancelled()
            if progress >= last_progress + 0.5 or progress >= 100:
                self.repository.update_progress(record.id, stage, progress)
                last_progress = progress

        try:
            options = PipelineOptions.model_validate(record.request["options"])
            result = self.engine.transcribe(record.input_path, options, report)
            current = self.repository.get(record.id)
            if current and current.cancel_requested:
                raise JobCancelled()
            self.repository.succeed(
                record.id,
                result.model_dump(mode="json", exclude_none=True),
            )
        except JobCancelled:
            self.repository.cancel(record.id)
        except APIError as exc:
            logger.exception("Job %s failed", record.id)
            self.repository.fail(record.id, exc.code, exc.message)
        except Exception as exc:
            logger.exception("Job %s failed", record.id)
            self.repository.fail(record.id, "inference_failed", str(exc))
        finally:
            self._remove_input(record.input_path)

    @staticmethod
    def _remove_input(path: Path) -> None:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Unable to remove job input %s", path, exc_info=True)


def run() -> None:
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    repository = JobRepository(settings.database_path)
    repository.initialize()
    worker_id = (
        settings.worker_id or f"{socket.gethostname()}-{os.getpid()}-gpu{settings.device_index}"
    )
    worker = Worker(
        repository,
        WhisperXEngine(settings),
        worker_id=worker_id,
        poll_seconds=settings.worker_poll_seconds,
    )
    try:
        worker.run_forever()
    except KeyboardInterrupt:
        logger.info("Worker stopped")


if __name__ == "__main__":
    run()
