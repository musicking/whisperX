from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from whisperx_api.audio.dependencies import get_engine
from whisperx_api.audio.schemas import (
    AudioTask,
    DiarizationResult,
    DiarizationTurn,
    Segment,
    TranscriptionResult,
    Word,
)
from whisperx_api.config import Environment, Settings
from whisperx_api.main import create_app


class FakeEngine:
    def transcribe(self, audio_path: Path, options, progress=None) -> TranscriptionResult:
        assert audio_path.exists()
        if progress:
            progress("transcribe", 100)
        return TranscriptionResult(
            task=options.task,
            language=options.language or "en",
            duration=1.25,
            text="Hello world.",
            segments=[
                Segment(
                    id=0,
                    start=0,
                    end=1.25,
                    text="Hello world.",
                    words=[
                        Word(word="Hello", start=0, end=0.5, score=0.95),
                        Word(word="world.", start=0.55, end=1.25, score=0.93),
                    ],
                )
            ],
        )

    def align(self, audio_path: Path, request, progress=None) -> TranscriptionResult:
        return TranscriptionResult(
            task=AudioTask.ALIGN,
            language=request.language,
            duration=1,
            text=request.segments[0].text,
            segments=[
                Segment(
                    id=0,
                    start=request.segments[0].start,
                    end=request.segments[0].end,
                    text=request.segments[0].text,
                )
            ],
        )

    def diarize(
        self,
        audio_path: Path,
        *,
        min_speakers,
        max_speakers,
        return_embeddings,
        progress=None,
    ) -> DiarizationResult:
        return DiarizationResult(
            turns=[DiarizationTurn(start=0, end=1, speaker="SPEAKER_00")],
            speaker_embeddings={"SPEAKER_00": [0.1]} if return_embeddings else None,
        )

    def unload(self) -> None:
        pass


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        environment=Environment.TEST,
        data_dir=tmp_path,
        database_path=tmp_path / "test.sqlite3",
        device="cpu",
        compute_type="int8",
        require_worker_for_readiness=False,
        sync_via_worker=False,
    )


@pytest.fixture
async def app(settings: Settings):
    application = create_app(settings)
    fake_engine = FakeEngine()
    application.dependency_overrides[get_engine] = lambda: fake_engine
    async with application.router.lifespan_context(application):
        yield application
    application.dependency_overrides.clear()


@pytest.fixture
async def client(app) -> AsyncIterator[AsyncClient]:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as http_client:
        yield http_client
