# Integrated WhisperX API

The API is part of `whisperx/api`. One FastAPI process directly invokes WhisperX;
there is no separate inference worker, task database, or external message queue.
The existing Python and CLI interfaces remain available without installing the API extra.

## Install and run

For Linux GPU deployment with Docker Compose, see [Docker deployment](docker-deployment.md).

Install FFmpeg and make it available on PATH. For GPU execution, prepare the NVIDIA
driver and CUDA/cuDNN environment required by this repository (CUDA 12.8).

```powershell
uv sync --python 3.12 --extra api
# Only copy this template when .env does not already exist.
Copy-Item .env.example .env
uv run --extra api whisperx-api
```

Alternatively: `uv run --extra api python -m whisperx.api`.
For CPU execution set `WHISPERX_API_DEVICE=cpu` and `WHISPERX_API_COMPUTE_TYPE=int8`.
Default port: 7865. Swagger: http://localhost:7865/docs.
Docs are disabled by default when `WHISPERX_API_ENVIRONMENT=production`.

All settings have defaults. The commented `.env.example` lists common settings;
you do not need to repeat every default in `.env`. Advanced settings remain
available through `WHISPERX_API_` environment variables; see `whisperx/api/config.py`.

## Endpoints

All audio endpoints accept `multipart/form-data` with a required `file` upload.
Responses wait for inference to finish. Internal services call endpoints directly;
no API key or authentication header is required.

| Method | Path | Purpose |
| --- | --- | --- |
| POST | /v1/audio/transcriptions | Transcribe audio |
| POST | /v1/audio/translations | Translate speech into English |
| POST | /v1/audio/alignments | Align existing timed transcript segments |
| POST | /v1/audio/diarizations | Detect anonymous speaker turns |
| POST | /v1/audio/subtitles | Transcribe, align and render subtitles |
| POST | /v1/audio/language | Detect language |
| GET | /health/live | Liveness probe (public) |
| GET | /health/ready | App initialization probe (public) |
| GET | /v1/models | List allowed models |
| GET | /v1/capabilities | Discover configured capabilities |

Transcriptions and translations follow OpenAI-style paths and form field names,
but use local WhisperX model names. This is not a claim of complete OpenAI compatibility.

### Transcription

```powershell
curl.exe http://localhost:7865/v1/audio/transcriptions `
  -F "file=@audio.mp3" `
  -F "language=zh" `
  -F "response_format=verbose_json" `
  -F "timestamp_granularities[]=word"
```

Supported options: `model`, `language`, `prompt`, `hotwords`, `temperature`,
`batch_size`, `chunk_size`, `align`, `diarize`, `min_speakers`, `max_speakers`,
`return_char_alignments`, `return_speaker_embeddings`, `response_format` and
`timestamp_granularities[]` (also accepted without brackets).
Timestamps require `verbose_json`; word timestamps require alignment.
Without an explicit `align` option, alignment is enabled when word timestamps are requested.

Output formats: `json` (text only), `verbose_json` (full segments/words), `text`,
`srt`, `vtt`, `tsv`. Time values in JSON are seconds; TSV timestamps are milliseconds.

Translation accepts `model`, optional source `language`, `prompt`, `temperature`,
`batch_size`, `chunk_size`, and `response_format`. Alignment is disabled for translation.

### Alignment

Send `file` and a form field named `request` containing JSON:

```json
{"language":"zh","segments":[{"start":0,"end":10,"text":"需要对齐的文本"}],"return_char_alignments":false}
```

Each segment needs `start`, `end` and `text`; `end` must exceed `start`.
The response contains the full aligned result.

### Diarization

Options: `num_speakers`, `min_speakers`, `max_speakers`, `return_speaker_embeddings`.
Returns `turns` with `start`, `end`, `speaker`, and optional `speaker_embeddings`.
Set `WHISPERX_API_HF_TOKEN` and accept the terms of
`pyannote/speaker-diarization-community-1` before using diarization.
Embeddings also require `WHISPERX_API_ALLOW_SPEAKER_EMBEDDINGS=true`.

### Subtitles

```powershell
curl.exe http://localhost:7865/v1/audio/subtitles `
  -F "file=@narration.mp3" `
  -F "language=zh" `
  -F "max_line_width=30" `
  -F "max_line_count=1" `
  -F "response_format=srt" `
  -o subtitles.srt
```

Uses upstream `get_writer()` with aligned ASR text. Default format: SRT.
Options: `model`, `language`, `prompt`, `hotwords`, `temperature`, `batch_size`,
`chunk_size`, `response_format`, `max_line_width`, `max_line_count`, `highlight_words`.
`max_line_count` requires `max_line_width`.
This version does not implement document-assisted subtitle matching:
`document` and `document_text` are explicitly rejected.

### Language detection

Send `file` and optional `model`; returns `{"language":"zh"}`.
Uses WhisperX's native detector, which analyzes the opening audio window.

## Execution and errors

Models load lazily and are reused with bounded caches. Enable `PRELOAD_MODEL` to load
the default ASR model during app startup. With lazy loading, readiness does not prove
that model downloads or GPU inference will succeed.
Inference runs in threads, using independent model instances to isolate request state.
`WHISPERX_API_INFERENCE_CONCURRENCY` controls the number of instances (default 1).
Each instance owns its own bounded model caches; additional instances consume additional
VRAM/RAM. Start with 2 and measure throughput on your hardware before increasing further.
An in-memory queue holds available instances, not task records. Requests borrow an
instance and return it after inference, including failures. Waiting requests remain in memory.
Keep the provided single-process launcher: multiple Uvicorn processes would load
separate model caches and could exhaust VRAM. Waiting requests stay in process memory.
This version has no durable jobs, retries or real-time audio streaming.

Uploads are removed after inference, including failures and cancelled uploads.
The internal-service API does not impose a file-size limit.
Errors return `{"error":{"code":"...","message":"...","request_id":"...","details":null}}`.
Common status codes: 422 (invalid options or undecodable audio),
503 (model loading failure), 500 (unexpected server errors).
Input validation is handled only at the request boundary. Server faults are logged
at one error handler with the request ID, path and original exception chain.
Requests and errors carry `X-Request-ID`.

### File logging

Application, model, request and Uvicorn lifecycle logs are written to
`WHISPERX_API_LOG_FILE` (default `./data/logs/api.log`) in UTF-8.
Request logs include request ID, path, HTTP status and elapsed time; inference logs
also include waiting time. Request bodies and prompts are not logged.
`WHISPERX_API_LOG_MAX_BYTES` defaults to 10 MiB and `WHISPERX_API_LOG_BACKUP_COUNT`
defaults to 5. Log files rotate automatically and handlers close at shutdown.

```dotenv
WHISPERX_API_INFERENCE_CONCURRENCY=2
WHISPERX_API_LOG_FILE=./data/logs/api.log
WHISPERX_API_LOG_LEVEL=info
```

## Development

Structure follows domain packages with thin routers, Pydantic schemas, reusable
dependencies and a synchronous WhisperX facade. See the supplied conventions:
https://github.com/zhanymkanov/fastapi-best-practices.

```powershell
uv run --extra api --extra dev pytest tests/api -q
uv run --extra api --extra dev ruff check whisperx/api tests/api
uv run --extra api --extra dev ruff format --check whisperx/api tests/api
```

API tests substitute native model calls to validate routing, options, serialization,
model reuse, concurrency and cleanup without downloading models or requiring a GPU.
