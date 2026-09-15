import asyncio
import threading
import time
from dataclasses import dataclass
from types import SimpleNamespace

import httpx
import numpy as np
import pytest

import whisperx
from whisperx.api.config import Settings
from whisperx.api.main import create_app


@dataclass
class Options:
    temperatures: object = None
    initial_prompt: str | None = None
    hotwords: str | None = None
    beam_size: int = 5
    best_of: int = 5
    patience: float = 1
    length_penalty: float = 1
    repetition_penalty: float = 1
    no_repeat_ngram_size: int = 0
    compression_ratio_threshold: float = 2.4
    log_prob_threshold: float = -1
    no_speech_threshold: float = 0.6
    suppress_blank: bool = True
    suppress_tokens: object = None


class NativeModel:
    def __init__(self):
        self.calls = []

    def transcribe(self, audio, **kwargs):
        self.calls.append(kwargs)
        segment = SimpleNamespace(
            start=0.2,
            end=0.8,
            text="原生分段。",
            avg_logprob=-0.1,
            words=None,
        )
        return iter([segment]), SimpleNamespace(language=kwargs.get("language") or "zh")


class Pipeline:
    def __init__(self):
        self.options = Options()
        self.tokenizer = None
        self._vad_params = {}
        self.calls = []
        self.active = 0
        self.maximum_active = 0
        self.model = NativeModel()

    def transcribe(self, audio, **kwargs):
        self.active += 1
        self.maximum_active = max(self.maximum_active, self.active)
        self.calls.append((kwargs, self.options, threading.get_ident()))
        time.sleep(0.03)
        self.active -= 1
        return {
            "language": kwargs.get("language") or "zh",
            "segments": [{"start": 0.0, "end": 1.0, "text": "你好"}],
        }

    def detect_language(self, audio):
        return "zh"


@pytest.fixture
async def api(tmp_path, monkeypatch):
    pipeline = Pipeline()
    loads = []

    def load_model(name, **kwargs):
        loads.append(name)
        return pipeline

    def align(segments, model, metadata, audio, device, **kwargs):
        words = [{"word": "你好", "start": 0.1, "end": 0.9, "score": 0.99}]
        return {"segments": [{**segments[0], "words": words}], "word_segments": words}

    monkeypatch.setattr(whisperx, "load_model", load_model)
    monkeypatch.setattr(whisperx, "load_audio", lambda path: np.zeros(16000, dtype=np.float32))
    monkeypatch.setattr(
        whisperx, "load_align_model", lambda **kw: (object(), {"language": kw["language_code"]})
    )
    monkeypatch.setattr(whisperx, "align", align)
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path,
        log_file=tmp_path / "logs/api.log",
        device="cpu",
    )
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
            base_url="http://test",
        ) as client:
            yield client, pipeline, loads, tmp_path, app


def audio():
    return {"file": ("audio.mp3", b"example audio", "audio/mpeg")}


def test_empty_hf_token_is_disabled():
    settings = Settings(_env_file=None, hf_token=" ")
    assert settings.hf_token is None


async def test_transcription_alignment_and_model_reuse(api):
    client, pipeline, loads, directory, _ = api
    response = await client.post(
        "/v1/audio/transcriptions",
        files=audio(),
        data={
            "response_format": "verbose_json",
            "timestamp_granularities[]": "word",
            "prompt": "财经新闻",
            "hotwords": "标普500",
        },
    )
    assert response.status_code == 200
    assert response.json()["segments"][0]["words"][0]["start"] == 0.1
    assert pipeline.calls[0][1].initial_prompt == "财经新闻"
    assert pipeline.calls[0][2] != threading.get_ident()
    assert pipeline.options.initial_prompt is None
    language = await client.post("/v1/audio/language", files=audio())
    assert language.json() == {"language": "zh"}
    assert loads == ["small"]
    assert list((directory / "uploads").iterdir()) == []


async def test_translation_disables_alignment(api):
    client, pipeline, _, _, _ = api
    response = await client.post("/v1/audio/translations", files=audio())
    assert response.status_code == 200
    assert response.json() == {"text": "你好"}
    assert pipeline.calls[0][0]["task"] == "translate"


async def test_alignment_request(api):
    client, _, _, _, _ = api
    response = await client.post(
        "/v1/audio/alignments",
        files=audio(),
        data={"request": '{"language":"zh","segments":[{"start":0,"end":1,"text":"你好"}]}'},
    )
    assert response.status_code == 200
    assert response.json()["task"] == "align"


@pytest.mark.parametrize("format,marker", [("srt", "00:00:00,200"), ("vtt", "WEBVTT")])
async def test_native_subtitle_writer(api, format, marker):
    client, pipeline, _, _, _ = api
    response = await client.post(
        "/v1/audio/subtitles",
        files=audio(),
        data={
            "response_format": format,
            "prompt": "你好，欢迎收听。",
            "max_line_width": "20",
            "max_line_count": "1",
        },
    )
    assert response.status_code == 200
    assert marker in response.text
    assert "原生分段" in response.text
    assert pipeline.calls == []
    assert pipeline.model.calls[0]["initial_prompt"] == "你好，欢迎收听。"
    assert pipeline.model.calls[0]["condition_on_previous_text"] is True
    assert pipeline.model.calls[0]["without_timestamps"] is False
    assert pipeline.model.calls[0]["vad_filter"] is False


async def test_native_subtitle_word_highlighting_requests_timestamps(api):
    client, pipeline, _, _, _ = api
    response = await client.post(
        "/v1/audio/subtitles",
        files=audio(),
        data={"response_format": "srt", "highlight_words": "true"},
    )
    assert response.status_code == 200
    assert pipeline.model.calls[0]["word_timestamps"] is True


@pytest.mark.parametrize(
    "path,data",
    [
        ("transcriptions", {"model": "unknown"}),
        ("transcriptions", {"temperature": "2"}),
        ("transcriptions", {"return_char_alignments": "true", "align": "false"}),
        ("transcriptions", {"timestamp_granularities[]": "word"}),
        ("alignments", {"request": "invalid json"}),
        ("diarizations", {"min_speakers": "3", "max_speakers": "1"}),
        ("diarizations", {}),
        ("subtitles", {"document_text": "原稿"}),
        ("subtitles", {"document_text": "   "}),
        ("subtitles", {"document_text": ""}),
        ("subtitles", {"document": "原稿"}),
    ],
)
async def test_invalid_options(api, path, data):
    client, _, _, directory, _ = api
    response = await client.post(f"/v1/audio/{path}", files=audio(), data=data)
    assert response.status_code == 422
    assert "error" in response.json()
    assert not list(directory.glob("uploads/*"))


async def test_discovery(api):
    client, _, _, _, app = api
    assert (await client.get("/health/live")).status_code == 200
    assert (await client.get("/health")).json()["status"] == "ready"
    assert (await client.get("/v1/models")).json()["data"][2]["default"] is True
    assert (await client.get("/v1/capabilities")).json()["diarization"] is False
    assert (await client.get("/v1/capabilities")).json()["document_subtitles"] is True
    assert "/health" in app.openapi()["paths"]
    assert "/health/ready" not in app.openapi()["paths"]
    assert "/v1/audio/language" in app.openapi()["paths"]
    assert all("-" not in path for path in app.openapi()["paths"])


async def test_concurrent_inference_is_serialized(api):
    client, pipeline, _, _, _ = api
    responses = await asyncio.gather(
        *[client.post("/v1/audio/transcriptions", files=audio()) for _ in range(3)]
    )
    assert all(response.status_code == 200 for response in responses)
    assert pipeline.maximum_active == 1


async def test_diarization_uses_native_pipeline(api, monkeypatch):
    import pandas as pd

    from whisperx import diarize

    client, _, _, _, app = api
    app.state.settings.hf_token = "server-token"
    app.state.settings.allow_speaker_embeddings = True
    calls = []

    class Diarizer:
        def __init__(self, **kwargs):
            assert kwargs["token"] == "server-token"

        def __call__(self, audio, **kwargs):
            calls.append(kwargs)
            frame = pd.DataFrame([{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}])
            return frame, {"SPEAKER_00": [0.1, 0.2]}

    monkeypatch.setattr(diarize, "DiarizationPipeline", Diarizer)
    response = await client.post(
        "/v1/audio/diarizations",
        files=audio(),
        data={
            "num_speakers": "1",
            "return_speaker_embeddings": "true",
        },
    )
    assert response.status_code == 200
    assert calls[0]["num_speakers"] == 1
    assert response.json()["turns"][0]["speaker"] == "SPEAKER_00"
    assert response.json()["speaker_embeddings"] == {"SPEAKER_00": [0.1, 0.2]}


async def test_inference_failure_restores_options_and_removes_upload(api):
    client, pipeline, _, directory, _ = api

    def fail(*args, **kwargs):
        pipeline.tokenizer = object()
        raise RuntimeError("model failed")

    pipeline.transcribe = fail
    response = await client.post("/v1/audio/transcriptions", files=audio(), data={"prompt": "test"})
    assert response.status_code == 500
    assert pipeline.options.initial_prompt is None
    assert pipeline.tokenizer is None
    assert not list(directory.glob("uploads/*"))


@pytest.mark.parametrize(
    "language,texts,expected",
    [
        ("en", ["Hello", "world"], "Hello world"),
        ("en", [" Hello. ", "", "  Good morning. "], "Hello. Good morning."),
        ("zh", ["你好", "世界"], "你好世界"),
        ("ja", ["こんにちは", "世界"], "こんにちは世界"),
    ],
)
def test_segment_text_preserves_language_boundaries(language, texts, expected):
    from whisperx.api.audio.engine import WhisperXEngine
    from whisperx.api.audio.schemas import AudioTask

    raw = {
        "language": language,
        "segments": [
            {"start": index, "end": index + 1, "text": text} for index, text in enumerate(texts)
        ],
    }
    result = WhisperXEngine._normalize_result(raw, task=AudioTask.TRANSCRIBE, duration=len(texts))
    assert result.text == expected


@pytest.mark.parametrize(
    "error,status",
    [
        (OSError("download unavailable"), 503),
        (TypeError("incorrect model call"), 500),
    ],
)
async def test_model_loading_errors_preserve_classification_and_logs(
    api, monkeypatch, error, status
):
    client, _, _, directory, app = api

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(whisperx, "load_model", fail)
    response = await client.post(
        "/v1/audio/transcriptions",
        files=audio(),
        headers={"X-Request-ID": "model-error"},
    )
    assert response.status_code == status
    logs = app.state.settings.log_file.read_text(encoding="utf-8")
    assert "request_id=model-error" in logs
    assert str(error) in logs
    assert app.state.available_engines.qsize() == 1
    assert not list(directory.glob("uploads/*"))


async def test_invalid_native_result_is_server_error(api):
    client, pipeline, _, _, _ = api
    pipeline.transcribe = lambda *args, **kwargs: {
        "language": "en",
        "segments": [
            {
                "start": 0,
                "end": 1,
                "text": "Hello",
                "words": [{"word": "Hello", "start": "invalid timestamp"}],
            }
        ],
    }
    response = await client.post("/v1/audio/transcriptions", files=audio())
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "internal_error"


async def test_alignment_loading_failure_is_not_input_error(api, monkeypatch):
    client, _, _, _, _ = api

    def fail(**kwargs):
        raise OSError("alignment download unavailable")

    monkeypatch.setattr(whisperx, "load_align_model", fail)
    response = await client.post(
        "/v1/audio/alignments",
        files=audio(),
        data={"request": '{"language":"en","segments":[{"start":0,"end":1,"text":"Hello"}]}'},
    )
    assert response.status_code == 503


async def test_unsupported_alignment_language_is_input_error(api):
    client, _, _, _, _ = api
    response = await client.post(
        "/v1/audio/alignments",
        files=audio(),
        data={"request": '{"language":"zz","segments":[{"start":0,"end":1,"text":"Hello"}]}'},
    )
    assert response.status_code == 422


async def test_diarization_capability_is_checked_before_inference(api):
    client, pipeline, _, _, _ = api
    response = await client.post(
        "/v1/audio/transcriptions", files=audio(), data={"diarize": "true"}
    )
    assert response.status_code == 422
    assert pipeline.calls == []


async def test_invalid_audio_is_input_error(api, monkeypatch):
    client, _, _, _, _ = api

    def fail(path):
        raise RuntimeError("FFmpeg failed to decode audio")

    monkeypatch.setattr(whisperx, "load_audio", fail)
    response = await client.post("/v1/audio/transcriptions", files=audio())
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_audio"


async def test_document_subtitles_align_original_text(api, monkeypatch):
    client, pipeline, _, directory, _ = api
    calls = []

    def align(segments, *args, **kwargs):
        calls.append(segments)
        words = [{"word": "你好！", "start": 0.1, "end": 0.9}]
        return {"segments": [{**segments[0], "words": words}], "word_segments": words}

    monkeypatch.setattr(whisperx, "align", align)
    response = await client.post(
        "/v1/audio/subtitles",
        files=audio(),
        data={"document_text": "你好！", "language": "zh", "response_format": "srt"},
    )
    assert response.status_code == 200
    assert "你好" in response.text
    assert "你好！" not in response.text
    assert calls[0][0]["text"] == "你好！"
    assert len(pipeline.calls) == 1
    assert not list(directory.glob("uploads/*"))


async def test_unaligned_document_is_rejected(api, monkeypatch):
    client, _, _, _, app = api
    monkeypatch.setattr(whisperx, "align", lambda segments, *a, **kw: {"segments": segments})
    response = await client.post(
        "/v1/audio/subtitles",
        files=audio(),
        data={"document_text": "你好！"},
    )
    assert response.status_code == 422
    assert app.state.available_engines.qsize() == 1


async def test_document_json_preserves_original_text(api):
    client, _, _, _, _ = api
    response = await client.post(
        "/v1/audio/subtitles",
        files=audio(),
        data={"document_text": "你好！", "response_format": "json"},
    )
    assert response.status_code == 200
    assert response.json() == {"text": "你好！"}


async def test_document_with_multiple_asr_segments_and_metadata(api, monkeypatch):
    client, pipeline, _, _, _ = api
    captured = []
    monkeypatch.setattr(
        pipeline,
        "transcribe",
        lambda audio, **kw: {
            "language": "zh",
            "segments": [
                {"start": 0, "end": 0.5, "text": "你好", "avg_logprob": -0.1},
                {"start": 0.5, "end": 1, "text": "今天市场上涨", "avg_logprob": -0.2},
            ],
        },
    )

    def align(segments, *args, **kwargs):
        captured.extend(segments)
        return {
            "segments": [
                {
                    **item,
                    "words": [{"word": item["text"], "start": item["start"], "end": item["end"]}],
                }
                for item in segments
            ]
        }

    monkeypatch.setattr(whisperx, "align", align)
    original = "你好！\n今天市场上涨。"
    response = await client.post(
        "/v1/audio/subtitles",
        files=audio(),
        data={"document_text": original, "response_format": "verbose_json"},
    )
    assert response.status_code == 200
    assert response.json()["text"] == original
    assert [item["text"] for item in captured] == ["你好！\n", "今天市场上涨。"]
    assert [(item["start"], item["end"]) for item in captured] == [(0, 0.5), (0.5, 1)]


@pytest.mark.parametrize(
    "text,words",
    [
        ("你好！", [{"word": "你", "start": 0.1, "end": 0.2}]),
        ("你", [{"word": "你", "start": 0.1, "end": 0.2}]),
    ],
)
async def test_incomplete_document_alignment_is_rejected(api, monkeypatch, text, words):
    client, _, _, _, _ = api
    monkeypatch.setattr(
        whisperx,
        "align",
        lambda segments, *a, **kw: {
            "segments": [{"start": 0, "end": 1, "text": text, "words": words}],
        },
    )
    response = await client.post(
        "/v1/audio/subtitles",
        files=audio(),
        data={"document_text": "你好！"},
    )
    assert response.status_code == 422
