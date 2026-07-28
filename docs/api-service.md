# WhisperX API service

This repository includes a FastAPI service around the native WhisperX pipeline. It provides
OpenAI-compatible synchronous endpoints, durable long-running jobs, forced alignment, speaker
diarization, subtitle rendering, health checks, and Prometheus metrics.

The service follows the conventions from
[zhanymkanov/fastapi-best-practices](https://github.com/zhanymkanov/fastapi-best-practices):
domain-oriented packages, thin routers, Pydantic boundary validation, explicit dependencies,
blocking inference outside the event loop, a real worker for long tasks, and async HTTP tests.

## Install

```bash
uv sync --extra api --extra dev
copy .env.example .env
```

Speaker diarization additionally requires a server-side Hugging Face read token and acceptance of
the terms for `pyannote/speaker-diarization-community-1`.

## Run

Use two terminals sharing the same data directory:

```bash
uv run whisperx-api
```

```bash
uv run whisperx-api-worker
```

The API is available at `http://localhost:8000`. Swagger UI is enabled at `/docs` in local, test,
and staging environments. It is hidden by default in production.

The API process uses one Uvicorn worker intentionally. Starting multiple API processes that each
load a GPU model duplicates VRAM. By default, synchronous OpenAI-compatible requests are also
submitted to the durable queue and awaited by the API, so only the dedicated worker loads the ASR
model. Scale by assigning a worker and queue partition to each GPU in a production deployment.

## Configuration

All settings use the `WHISPERX_API_` prefix. Important settings:

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_NAME` | `small` | Default ASR model |
| `ALLOWED_MODELS` | tiny through large-v3 | Request model allowlist |
| `DEVICE` | `cuda` | `cuda` or `cpu` |
| `COMPUTE_TYPE` | `default` | `float16` on GPU, `float32` on CPU |
| `BATCH_SIZE` | `8` | Maximum preferred inference batch |
| `MAX_LOADED_ASR_MODELS` | `1` | Bound ASR model VRAM cache |
| `MAX_LOADED_ALIGN_MODELS` | `3` | Bound language alignment model cache |
| `MAX_UPLOAD_BYTES` | 1 GiB | Streaming upload limit |
| `API_KEY` | empty | Optional shared bearer or `X-API-Key` secret |
| `HF_TOKEN` | empty | Server-only token for diarization |
| `ALLOW_SPEAKER_EMBEDDINGS` | `false` | Explicitly enable sensitive voice embeddings |
| `REQUIRE_WORKER_FOR_READINESS` | `true` | Make readiness fail without a worker heartbeat |
| `SYNC_VIA_WORKER` | `true` | Route synchronous compatibility calls through the GPU worker |
| `SYNC_TIMEOUT_SECONDS` | `900` | Maximum time to await a synchronous compatibility call |

Device selection, cache paths, model downloads, threads, and Hugging Face tokens are deployment
settings, not freely controlled request fields.

## OpenAI-compatible endpoints

### Transcription

```bash
curl http://localhost:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer $API_KEY" \
  -F "file=@audio.mp3" \
  -F "model=small" \
  -F "response_format=verbose_json" \
  -F "timestamp_granularities=word"
```

Additional WhisperX fields include `align`, `diarize`, `min_speakers`, `max_speakers`,
`return_char_alignments`, `return_speaker_embeddings`, and `hotwords`.

### Translation

```bash
curl http://localhost:8000/v1/audio/translations \
  -H "Authorization: Bearer $API_KEY" \
  -F "file=@audio.mp3" \
  -F "response_format=json"
```

WhisperX translation targets English and cannot be combined with forced alignment.

## Durable jobs

Create:

```bash
curl http://localhost:8000/v1/jobs \
  -H "Authorization: Bearer $API_KEY" \
  -F "file=@meeting.mp4" \
  -F "model=large-v3" \
  -F "align=true" \
  -F "diarize=true"
```

Inspect and consume:

```text
GET    /v1/jobs
GET    /v1/jobs/{job_id}
GET    /v1/jobs/{job_id}/events
GET    /v1/jobs/{job_id}/result?format=verbose_json
DELETE /v1/jobs/{job_id}
```

Jobs are stored in SQLite/WAL for the single-host deployment. Inference is claimed transactionally
by the worker and inputs are removed after a terminal inference attempt. For multi-host deployment,
replace the repository with PostgreSQL/Redis and object storage while keeping the domain interface.

## WhisperX-native endpoints

```text
POST /v1/audio/alignments
POST /v1/audio/diarizations
```

Alignment accepts a multipart `request` JSON field:

```json
{
  "language": "en",
  "segments": [
    {"start": 0, "end": 10, "text": "Existing transcript"}
  ]
}
```

## Operations

```text
GET /health/live
GET /health/ready
GET /v1/capabilities
GET /v1/models
GET /version
GET /metrics
```

Readiness checks both storage and a recent worker heartbeat by default. `/metrics`, discovery, and
version endpoints require the configured API key; health probes remain public.

## Test and lint

```bash
uv run --extra api --extra dev pytest
uv run --extra api --extra dev ruff check whisperx_api tests/api
uv run --extra api --extra dev ruff format --check whisperx_api tests/api
```
