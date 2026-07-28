# WhisperX FastAPI 服务：能力调研与 API 设计建议

> 调研日期：2026-07-28
> 上游基线：`m-bain/whisperX` `main` 分支 `2cfd7b7`；最新稳定标签 `v3.8.6`。生产实现应固定到稳定版本或提交 SHA，而不是跟随 `main`。

## 1. 结论

WhisperX 本身是 Python 库和 CLI，不是 HTTP 服务。它最适合封装为“离线音视频转写任务服务”，核心流水线为：

`音频解码/16 kHz 单声道化 → VAD → 批量 ASR → 可选强制对齐 → 可选说话人分离 → JSON/字幕导出`

推荐同时提供两套入口：

1. **OpenAI 兼容入口**：降低现有 Whisper 客户端的迁移成本。
2. **WhisperX 原生异步任务入口**：完整支持对齐、说话人分离、进度、长音频、结果导出和取消。

首版不应承诺实时流式识别。WhisperX 的实现会先做 VAD、按 chunk 合并后再批量推理，是离线流水线；SSE/WebSocket 最多用于传输**任务状态和阶段性结果**，不等于实时 ASR。

## 2. 上游实际提供的能力

| 能力 | 上游依据 | 可暴露结果 |
|---|---|---|
| 多语言语音转文字 | `FasterWhisperPipeline.transcribe()` 支持语言自动检测、`transcribe` 任务和 batched inference | 文本、语言、句段、句段时间、平均对数概率 |
| 语音翻译为英文 | CLI 的 `task` 支持 `translate` | 英文文本和 ASR 句段 |
| VAD | 可选 `pyannote` 或 `silero`，并可调 onset、offset、chunk size | 服务内部语音区间；也可作为高级诊断结果返回 |
| 词级强制对齐 | `load_align_model()` + `align()` | 词级起止时间、score；可选字符级时间 |
| 独立对齐 | `align()` 接受已有 transcript segments 与音频，不要求先在同一次调用中做 ASR | 用户给定文本/句段的对齐结果 |
| 说话人分离 | `DiarizationPipeline` | speaker turns、每词/每句 speaker ID |
| 说话人嵌入 | diarization 支持 `return_embeddings` | speaker 到 embedding 向量的映射 |
| 多格式导出 | CLI writer 支持 `json`、`txt`、`srt`、`vtt`、`tsv`、`aud` | 下载文件或相应媒体类型的响应 |
| 推理进度 | transcribe、align、diarization 均已有 progress callback 路径 | job progress、SSE 事件 |

主要一手来源：

- [WhisperX README：能力、Python 用法、限制](https://github.com/m-bain/whisperX/blob/main/README.md)
- [ASR 实现：`transcribe()`、语言检测、模型加载](https://github.com/m-bain/whisperX/blob/main/whisperx/asr.py)
- [强制对齐实现与默认语言模型表](https://github.com/m-bain/whisperX/blob/main/whisperx/alignment.py)
- [说话人分离、embedding 与 speaker assignment](https://github.com/m-bain/whisperX/blob/main/whisperx/diarize.py)
- [CLI 的完整参数与输出格式](https://github.com/m-bain/whisperX/blob/main/whisperx/__main__.py)
- [上游 TypedDict 输出结构](https://github.com/m-bain/whisperX/blob/main/whisperx/schema.py)
- [音频解码：ffmpeg、16 kHz、单声道 float32](https://github.com/m-bain/whisperX/blob/main/whisperx/audio.py)
- [当前依赖和 Python/CUDA 版本基线](https://github.com/m-bain/whisperX/blob/main/pyproject.toml)

## 3. 建议的 HTTP API

### 3.1 P0：OpenAI 兼容同步接口

#### `POST /v1/audio/transcriptions`

用途：短音频同步转写；兼容常见 OpenAI Whisper SDK/调用方式。

请求：`multipart/form-data`

| 字段 | 类型 | 建议 |
|---|---|---|
| `file` | file | 必填；服务端限制大小、时长和允许格式 |
| `model` | string | 兼容字段；映射到服务已加载的模型别名，不允许任意下载模型 |
| `language` | string | 可选 ISO 语言码；空值自动检测 |
| `prompt` | string | 映射 `initial_prompt` |
| `response_format` | enum | `json`、`verbose_json`、`text`、`srt`、`vtt` |
| `temperature` | number | 可选 |
| `timestamp_granularities[]` | enum | `segment`、`word`；请求 `word` 时启用 alignment |

WhisperX 扩展字段可使用 `x_*` 前缀，或集中为一个 JSON 字段 `whisperx_options`：

- `align`，默认 `true`
- `diarize`，默认 `false`
- `min_speakers`、`max_speakers`
- `return_char_alignments`，默认 `false`
- `vad_method`、`vad_onset`、`vad_offset`
- `hotwords`

响应：

- `json/text/srt/vtt` 尽量遵循兼容格式；
- `verbose_json` 可增加 `language`、`duration`、`segments`、`words`、`speaker`、`alignment_score`；
- 响应头建议加 `X-Request-ID`、`X-Processing-Time-Ms`。

#### `POST /v1/audio/translations`

用途：将语音翻译为英文。

请求格式与 transcription 基本一致，但必须明确：

- 上游 CLI 在 `task=translate` 时强制 `no_align=True`；
- 因此不能承诺强制对齐后的词级时间戳、字符级时间戳或可靠的词级 speaker assignment；
- 首版应拒绝 `align=true`、`timestamp_granularities[]=word`，返回 `422`，而不是静默降级。

### 3.2 P0：WhisperX 原生异步任务接口

#### `POST /v1/jobs`

适合长音频、视频、说话人分离和批量工作。响应 `202 Accepted`。

请求支持两种输入：

- `multipart/form-data` 上传一个文件；
- JSON 指定已上传对象的 `input_id`。首版不建议直接允许任意公网 URL，避免 SSRF。

建议请求模型：

```json
{
  "input_id": "in_01...",
  "task": "transcribe",
  "language": null,
  "pipeline": {
    "align": true,
    "diarize": false,
    "return_char_alignments": false,
    "return_speaker_embeddings": false
  },
  "inference": {
    "model": "large-v3",
    "batch_size": 16,
    "initial_prompt": null,
    "hotwords": null
  },
  "vad": {
    "method": "pyannote",
    "onset": 0.5,
    "offset": 0.363,
    "chunk_size": 30
  },
  "diarization": {
    "min_speakers": null,
    "max_speakers": null
  }
}
```

服务端必须有独立的部署级白名单和上下限；客户端值不能直接控制设备、模型下载目录、HF token、CUDA device 或线程数。

创建响应：

```json
{
  "id": "job_01...",
  "status": "queued",
  "created_at": "2026-07-28T12:00:00Z",
  "links": {
    "self": "/v1/jobs/job_01...",
    "events": "/v1/jobs/job_01.../events",
    "result": "/v1/jobs/job_01.../result"
  }
}
```

#### `GET /v1/jobs/{job_id}`

返回 `queued | running | succeeded | failed | cancelling | cancelled | expired`，并带：

- `stage`: `decode | vad | transcribe | align | diarize | export`
- `progress`: 0–100
- `created_at`、`started_at`、`finished_at`
- 失败时返回稳定的 `error.code` 和安全的 `error.message`
- 排队信息可提供 `queue_position`，但不承诺精确完成时间

#### `GET /v1/jobs/{job_id}/events`

SSE 事件流，事件类型建议为：

- `status`
- `progress`
- `partial_result`（可选、非实时保证）
- `completed`
- `failed`
- `heartbeat`

断线恢复使用 `Last-Event-ID`。这里传输的是任务事件，不应命名为 realtime transcription。

#### `GET /v1/jobs/{job_id}/result`

默认返回规范化 JSON；支持 `?format=json|txt|srt|vtt|tsv|aud`。大结果或文件可返回短时下载 URL。

规范化 JSON 建议：

```json
{
  "id": "job_01...",
  "task": "transcribe",
  "language": "zh",
  "duration": 123.45,
  "text": "……",
  "segments": [
    {
      "id": 0,
      "start": 0.42,
      "end": 2.81,
      "text": "……",
      "speaker": "SPEAKER_00",
      "avg_logprob": -0.18,
      "words": [
        {
          "word": "你好",
          "start": 0.44,
          "end": 0.92,
          "score": 0.91,
          "speaker": "SPEAKER_00"
        }
      ]
    }
  ],
  "warnings": []
}
```

上游对部分不可对齐词可能省略时间字段，因此 API schema 的 word/char `start`、`end`、`score` 必须允许 `null` 或缺省，不能假设永远存在。

#### `DELETE /v1/jobs/{job_id}`

语义是请求取消：

- 尚未执行的任务可立即取消；
- 已进入 GPU kernel 的工作通常不能硬中断，只能在阶段或 batch 边界协作取消；
- 成功接受取消请求可返回 `202`，已终态则幂等返回当前状态。

#### `GET /v1/jobs`

按调用方分页列出任务；支持 `status`、`created_after` 过滤。生产环境必须按租户隔离。

### 3.3 P1：WhisperX 专项能力

| API | 功能 | 输入 | 输出/备注 |
|---|---|---|---|
| `POST /v1/alignments` | 独立强制对齐 | 音频 + `language` + transcript segments；短任务同步，长任务返回 job | 对齐后的 segments、words、可选 chars；适合已有人工稿 |
| `POST /v1/diarizations` | 独立说话人分离 | 音频、`num_speakers` 或 min/max、是否返回 embedding | speaker turns；embedding 默认关闭且要求额外权限 |
| `POST /v1/uploads` | 大文件预上传 | 流式 multipart/chunked upload | `input_id`、hash、媒体元数据、过期时间 |
| `DELETE /v1/uploads/{input_id}` | 删除未使用输入 | input ID | 幂等删除 |
| `POST /v1/batches` | 批量创建任务 | 多个 input ID + 共享配置 | batch ID 与各 job ID；不要把多个大文件塞进一个请求 |

独立 alignment 的 transcript 不宜只接受一整段字符串。应支持：

```json
{
  "language": "zh",
  "segments": [
    {"start": 0.0, "end": 8.0, "text": "待对齐文本"}
  ]
}
```

如果用户没有粗粒度时间，可允许单段覆盖全音频，但超长文本的可靠性和资源消耗应明确限制。

### 3.4 P0/P1：发现与运维接口

| API | 优先级 | 说明 |
|---|---:|---|
| `GET /v1/capabilities` | P0 | 返回语言、任务、输出格式、是否启用 alignment/diarization、限制值；客户端不必猜配置 |
| `GET /v1/models` | P0 | 只返回服务已配置/已缓存且允许使用的 ASR 模型别名和状态 |
| `GET /health/live` | P0 | 进程是否存活；不做 GPU 推理 |
| `GET /health/ready` | P0 | 队列、存储、模型和 GPU 是否可接受请求 |
| `GET /metrics` | P0 | Prometheus 指标；应放在内网或鉴权后 |
| `GET /version` | P1 | 服务、WhisperX、CUDA、CTranslate2 版本和构建 SHA；避免泄露敏感路径 |

关键指标：

- 队列长度、排队时间、任务时长、各 stage 时长
- 音频秒数/处理秒数（real-time factor）
- batch size、失败/取消/OOM 计数
- GPU 显存使用、模型加载状态
- 上传和结果存储字节数

## 4. 哪些参数不应成为逐请求自由参数

以下应由服务部署配置或白名单控制：

- `device`、`device_index`、`compute_type`
- `model_dir`、`local_files_only`
- 任意 Hugging Face/本地模型路径
- `hf_token`
- Torch/CTranslate2 线程数
- 任意 URL 下载
- 无上限的 `batch_size`、音频时长、文件大小、speaker embeddings

原因是这些字段会导致模型下载、SSRF/路径访问、GPU OOM、跨租户泄露或吞吐抖动。API 只暴露模型别名和受限的质量/速度 preset（例如 `fast | balanced | accurate`）会更稳健。

## 5. 高性能 FastAPI 实现建议

### 5.1 进程与 GPU 模型

- FastAPI 负责 HTTP、校验、鉴权、上传和任务状态；GPU 推理由独立 worker 执行。
- 每张 GPU 通常运行一个模型 worker，并在启动时预加载 ASR/VAD；不要简单启动多个 Uvicorn worker，让每个进程复制一份 GPU 模型。
- alignment 模型按语言做有界 LRU 缓存；diarization 模型按部署配置预热或懒加载一次。
- GPU worker 使用有界队列和显存预算。OOM 后将 batch size 降级重试至有限次数，不能无限重试。
- 可以在队列层做短时间 micro-batching，但只有模型、语言、task 和关键解码参数兼容的请求才能合批。

### 5.2 I/O 与任务

- 上传边读边写临时/对象存储并计算 hash，禁止把整个大文件读进 FastAPI 进程内存。
- ffmpeg 解码放到受限子进程池，设置超时、资源限制和输入探测。
- PostgreSQL/Redis 记录 job 状态；对象存储保存输入和大结果。单机 MVP 可替换为 SQLite + 本地受控目录，但接口保持不变。
- 写状态时用单调 progress，终态幂等；worker 通过 lease/heartbeat 避免任务因进程崩溃永久停在 running。
- 结果设置 TTL，并允许租户主动删除。

### 5.3 安全与可靠性

- 文件类型不能只信 `Content-Type` 或扩展名；用 ffprobe 探测，并限制时长、轨道数、采样率和解码后 PCM 上限。
- 用户上传文件名不得参与最终路径拼接。
- HF token 只从服务端 secret 注入，绝不从请求或响应回显。
- speaker embeddings 属于生物特征敏感数据：默认关闭、单独授权、审计、短 TTL，并避免写入普通日志。
- 返回稳定错误码，例如 `unsupported_media`、`audio_too_long`、`unsupported_language_for_alignment`、`model_unavailable`、`insufficient_gpu_memory`。

## 6. 上游限制必须体现在 API 契约中

1. 含数字、币种等不在 alignment dictionary 中的词可能无法获得时间戳。
2. 重叠语音处理不佳；speaker diarization 并不完美。
3. 对齐依赖特定语言的 wav2vec2/CTC 模型，并非 Whisper 支持的每种语言都一定有默认 alignment 模型。
4. 翻译与 alignment 在当前 CLI 流程中互斥。
5. 说话人结果是匿名 ID，不是人物身份识别。
6. ffmpeg 是音频解码运行时依赖；当前上游项目要求 Python `>=3.10,<3.14`，并固定到一组较新的 PyTorch/pyannote 依赖，部署镜像必须整体锁版本。

## 7. 推荐实现顺序

### 第一阶段：可用 MVP

1. `POST /v1/audio/transcriptions`
2. `POST /v1/audio/translations`
3. `POST /v1/jobs`
4. job 查询、SSE、结果、取消
5. `capabilities`、`models`、health、metrics
6. JSON/text/SRT/VTT 输出

### 第二阶段：WhisperX 差异化

1. 独立 alignment
2. 独立 diarization
3. 大文件 upload API
4. 批量任务
5. 更多导出格式和 webhook

### 暂不纳入首版

- 真正实时 WebSocket ASR
- 任意模型动态下载/逐请求换模型
- 用户传入 HF token
- 默认返回 speaker embeddings
- 任意公网 URL 抓取
- “识别具体人物身份”

## 8. 实现前需要确定的部署决策

后续编码前只需确定下面几项；其余可以按本文默认值实施：

1. 首发环境：Linux + NVIDIA GPU，还是需要 Windows/CPU 同时支持。
2. 默认 ASR 模型及显存预算。
3. 是否首版启用 diarization，以及服务端是否已有获授权的 Hugging Face token。
4. 单机服务还是 Redis/PostgreSQL/对象存储的可扩展部署。
5. OpenAI 兼容优先，还是原生 job API 优先。
