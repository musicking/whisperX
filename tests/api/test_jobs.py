from httpx import AsyncClient

from whisperx_api.audio.schemas import (
    AudioTask,
    Segment,
    TranscriptionResult,
)
from whisperx_api.jobs.repository import JobRepository


async def test_create_and_cancel_job(client: AsyncClient, app) -> None:
    created = await client.post(
        "/v1/jobs",
        files={"file": ("sample.wav", b"fake audio", "audio/wav")},
        data={"task": "transcribe", "align": "true"},
    )

    assert created.status_code == 202
    job_id = created.json()["id"]
    assert created.json()["status"] == "queued"
    repository: JobRepository = app.state.job_repository
    queued = repository.get(job_id)
    assert queued is not None
    assert queued.input_path.exists()

    cancelled = await client.delete(f"/v1/jobs/{job_id}")
    assert cancelled.status_code == 202
    assert cancelled.json()["status"] == "cancelled"
    assert not queued.input_path.exists()


async def test_completed_job_result(
    client: AsyncClient,
    app,
) -> None:
    created = await client.post(
        "/v1/jobs",
        files={"file": ("sample.wav", b"fake audio", "audio/wav")},
        data={"response_format": "verbose_json"},
    )
    job_id = created.json()["id"]
    repository: JobRepository = app.state.job_repository
    result = TranscriptionResult(
        task=AudioTask.TRANSCRIBE,
        language="en",
        text="Done.",
        segments=[Segment(id=0, start=0, end=1, text="Done.")],
    )
    repository.succeed(job_id, result.model_dump(mode="json", exclude_none=True))

    response = await client.get(f"/v1/jobs/{job_id}/result")

    assert response.status_code == 200
    assert response.json()["text"] == "Done."


async def test_unknown_job_is_404(client: AsyncClient) -> None:
    response = await client.get("/v1/jobs/job_missing")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


async def test_job_rejects_non_transcription_pipeline(client: AsyncClient) -> None:
    response = await client.post(
        "/v1/jobs",
        files={"file": ("sample.wav", b"fake audio", "audio/wav")},
        data={"task": "align"},
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
