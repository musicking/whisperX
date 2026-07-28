import asyncio
import time
from pathlib import Path

from whisperx_api.audio.schemas import (
    AudioTask,
    PipelineOptions,
    Segment,
    TranscriptionResult,
)
from whisperx_api.audio.service import transcribe_via_job
from whisperx_api.jobs.repository import JobRepository
from whisperx_api.jobs.schemas import JobStatus
from whisperx_api.jobs.worker import Worker


class WorkerEngine:
    def transcribe(self, audio_path, options, progress):
        progress("transcribe", 50)
        return TranscriptionResult(
            task=AudioTask.TRANSCRIBE,
            language="en",
            text="Finished.",
            segments=[Segment(id=0, start=0, end=1, text="Finished.")],
        )


def wait_for_job(repository: JobRepository) -> None:
    deadline = time.monotonic() + 2
    while not repository.list(status=None, limit=1, offset=0):
        if time.monotonic() >= deadline:
            raise TimeoutError("job was not created")
        time.sleep(0.01)


def test_worker_completes_and_cleans_input(tmp_path: Path) -> None:
    repository = JobRepository(tmp_path / "jobs.sqlite3")
    repository.initialize()
    input_path = tmp_path / "audio.wav"
    input_path.write_bytes(b"audio")
    options = PipelineOptions(align=False)
    queued = repository.create(
        input_path=input_path,
        request={
            "options": options.model_dump(mode="json"),
            "response_format": "verbose_json",
        },
    )
    claimed = repository.claim_next("test-worker")
    assert claimed is not None
    assert claimed.id == queued.id

    worker = Worker(
        repository,
        WorkerEngine(),  # type: ignore[arg-type]
        worker_id="test-worker",
        poll_seconds=0.01,
    )
    worker.process(claimed)

    finished = repository.get(queued.id)
    assert finished is not None
    assert finished.status == JobStatus.SUCCEEDED
    assert finished.result["text"] == "Finished."
    assert not input_path.exists()


async def test_synchronous_adapter_waits_for_durable_worker(tmp_path: Path) -> None:
    repository = JobRepository(tmp_path / "jobs.sqlite3")
    repository.initialize()
    input_path = tmp_path / "audio.wav"
    input_path.write_bytes(b"audio")
    pending = asyncio.create_task(
        transcribe_via_job(
            repository,
            input_path,
            PipelineOptions(align=False),
            response_format="json",
            timeout_seconds=2,
            poll_seconds=0.01,
        )
    )
    await asyncio.to_thread(wait_for_job, repository)
    claimed = repository.claim_next("test-worker")
    assert claimed is not None
    worker = Worker(
        repository,
        WorkerEngine(),  # type: ignore[arg-type]
        worker_id="test-worker",
        poll_seconds=0.01,
    )
    await asyncio.to_thread(worker.process, claimed)

    result = await pending

    assert result.text == "Finished."
    assert not input_path.exists()
