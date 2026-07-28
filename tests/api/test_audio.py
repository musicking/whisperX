import json

from httpx import AsyncClient


async def test_openai_compatible_transcription(client: AsyncClient) -> None:
    response = await client.post(
        "/v1/audio/transcriptions",
        files={"file": ("sample.wav", b"fake audio", "audio/wav")},
        data={
            "language": "en",
            "response_format": "verbose_json",
            "timestamp_granularities[]": "word",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["text"] == "Hello world."
    assert payload["segments"][0]["words"][0]["word"] == "Hello"
    assert response.headers["x-request-id"]


async def test_translation_returns_openai_json(client: AsyncClient) -> None:
    response = await client.post(
        "/v1/audio/translations",
        files={"file": ("sample.mp3", b"fake audio", "audio/mpeg")},
        data={"response_format": "json"},
    )

    assert response.status_code == 200
    assert response.json() == {"text": "Hello world."}


async def test_alignment_endpoint(client: AsyncClient) -> None:
    request = {
        "language": "en",
        "segments": [{"start": 0, "end": 1, "text": "Existing transcript"}],
    }
    response = await client.post(
        "/v1/audio/alignments",
        files={"file": ("sample.wav", b"fake audio", "audio/wav")},
        data={"request": json.dumps(request)},
    )

    assert response.status_code == 200
    assert response.json()["task"] == "align"
    assert response.json()["text"] == "Existing transcript"


async def test_srt_rendering(client: AsyncClient) -> None:
    response = await client.post(
        "/v1/audio/transcriptions",
        files={"file": ("sample.wav", b"fake audio", "audio/wav")},
        data={"response_format": "srt"},
    )

    assert response.status_code == 200
    assert "00:00:00,000 --> 00:00:01,250" in response.text


async def test_validation_errors_use_standard_envelope(client: AsyncClient) -> None:
    response = await client.post(
        "/v1/audio/transcriptions",
        files={"file": ("sample.wav", b"fake audio", "audio/wav")},
        data={"temperature": "2"},
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
