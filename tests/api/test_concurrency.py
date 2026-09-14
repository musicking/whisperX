import asyncio
import logging
import threading
from dataclasses import dataclass

import anyio
import anyio.lowlevel
import httpx
import numpy as np

import whisperx
from whisperx.api.config import Settings
from whisperx.api.logs import file_logging
from whisperx.api.main import create_app
from whisperx.api.storage import temporary_audio


@dataclass
class Options:
    temperatures: object = None
    initial_prompt: str | None = None
    hotwords: str | None = None


async def test_two_independent_models_run_concurrently(tmp_path, monkeypatch):
    barrier = threading.Barrier(2)
    pipelines = []

    class Pipeline:
        def __init__(self):
            self.options = Options()
            self._vad_params = {}
            self.tokenizer = None

        def transcribe(self, audio, **kwargs):
            prompt = self.options.initial_prompt
            barrier.wait(timeout=5)
            assert self.options.initial_prompt == prompt
            return {"language": "en", "segments": [{"start": 0, "end": 1, "text": prompt}]}

    def load_model(*args, **kwargs):
        pipeline = Pipeline()
        pipelines.append(pipeline)
        return pipeline

    monkeypatch.setattr(whisperx, "load_model", load_model)
    monkeypatch.setattr(whisperx, "load_audio", lambda path: np.zeros(16000, dtype=np.float32))
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path,
        log_file=tmp_path / "logs/api.log",
        device="cpu",
        inference_concurrency=2,
    )
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
            base_url="http://test",
        ) as client:
            responses = await asyncio.gather(
                *[
                    client.post(
                        "/v1/audio/transcriptions",
                        files={"file": ("audio.mp3", b"audio")},
                        data={"prompt": str(index)},
                        headers={"X-Request-ID": f"request-{index}"},
                    )
                    for index in range(4)
                ]
            )
            assert [response.json()["text"] for response in responses] == ["0", "1", "2", "3"]
            assert len(pipelines) == 2
            assert app.state.available_engines.qsize() == 2
            assert all(pipeline.tokenizer is None for pipeline in pipelines)
            assert not list(tmp_path.glob("uploads/*"))

            def fail(*args, **kwargs):
                raise RuntimeError("inference failed")

            for pipeline in pipelines:
                pipeline.transcribe = fail
            response = await client.post(
                "/v1/audio/transcriptions",
                files={"file": ("audio.mp3", b"audio")},
            )
            assert response.status_code == 500
            assert app.state.available_engines.qsize() == 2
    logs = settings.log_file.read_text(encoding="utf-8")
    assert "inference_concurrency=2" in logs
    assert "request_id=request-0" in logs
    assert "status=200" in logs
    assert "status=500" in logs
    assert "inference failed" in logs
    assert "API stopped" in logs
    assert "private-key" not in logs


async def test_cancelled_upload_is_closed_and_deleted(tmp_path):
    closed = False

    class Upload:
        filename = "audio.wav"
        reads = 0

        async def read(self, size):
            self.reads += 1
            if self.reads == 1:
                return b"audio"
            scope.cancel()
            await anyio.lowlevel.checkpoint()

        async def close(self):
            nonlocal closed
            await anyio.lowlevel.checkpoint()
            closed = True

    settings = Settings(_env_file=None, data_dir=tmp_path)
    with anyio.CancelScope() as scope:
        async with temporary_audio(Upload(), settings):
            raise AssertionError("Upload should be cancelled before yielding")
    assert closed
    assert not list(tmp_path.glob("uploads/*"))


def test_file_logs_rotate_and_release_handler(tmp_path):
    settings = Settings(
        _env_file=None,
        log_file=tmp_path / "api.log",
        log_max_bytes=200,
        log_backup_count=2,
    )
    logger = logging.getLogger("whisperx.api")
    previous_handlers = list(logging.getLogger("whisperx").handlers)
    with file_logging(settings):
        for index in range(10):
            logger.info("rotation test %s %s", index, "x" * 100)
    assert settings.log_file.exists()
    assert (tmp_path / "api.log.1").exists()
    assert (tmp_path / "api.log.2").exists()
    assert not (tmp_path / "api.log.3").exists()
    assert logging.getLogger("whisperx").handlers == previous_handlers
