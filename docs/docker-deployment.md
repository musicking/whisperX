# Docker Compose 部署（Linux / NVIDIA GPU）

采用单个 API 容器，直接调用 WhisperX，不需要 Worker、数据库或消息队列。
默认端口为 7865。Dockerfile 内包含 CUDA 12.8、cuDNN、FFmpeg 和 Python 环境。
镜像标签为 `whisperx-api:latest`，容器名固定为 `whisperx-api`。
宿主机需要 NVIDIA 驱动、Docker Engine、Docker Compose v2 和 NVIDIA Container Toolkit。

## 1. 检查 GPU 容器环境

按 [NVIDIA 官方说明](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
安装 NVIDIA Container Toolkit，然后配置 Docker：

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
nvidia-smi
docker compose version
docker run --rm --gpus all nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04 nvidia-smi
```

最后一个命令应能看到 GPU。宿主机不需要另外安装 CUDA Toolkit。
Compose GPU 配置参考 [Docker 官方说明](https://docs.docker.com/compose/how-tos/gpu-support/)。

## 2. 获取代码和配置

```bash
git clone --branch api https://github.com/musicking/whisperX.git WhisperX-API
cd WhisperX-API
if [ ! -f .env ]; then cp .env.example .env; fi
mkdir -p data/models/hf-hub data/logs
```

常用配置已在 `.env.example` 中注释。默认使用 small 模型和一个推理实例。
设置 `WHISPERX_API_INFERENCE_CONCURRENCY=2` 可启用两个独立实例，但会增加显存占用。
需要使用其他 GPU 时，在 `.env` 添加 `WHISPERX_DOCKER_GPU_ID=1`。
Compose 只暴露选中的 GPU，容器内设备索引固定为 0。
说话人区分仍需 `WHISPERX_API_HF_TOKEN` 以及对应模型使用授权。

## 3. 使用已有模型（可选）

如果已按照之前的迁移步骤生成并上传 `models-transfer.tar`：

```bash
tar -xf ~/models-transfer.tar -C data/models
if [ -d data/models/torch-checkpoints ]; then
    cp -a data/models/torch-checkpoints/. data/models/hf-hub/
fi
```

也可以把 Windows 的 Hugging Face `hub` 目录内容整体复制到 `data/models/hf-hub/`。
保留每个模型的 `refs`、`snapshots`、`blobs` 结构；迁移时应保存链接目标的实际文件。

```text
data/
├── models/
│   ├── hf-hub/
│   │   ├── models--Systran--faster-whisper-small/
│   │   └── models--jonatasgrosman--wav2vec2-large-xlsr-53-chinese-zh-cn/
│   └── nltk-data/                 # 可选，镜像内也已准备 punkt_tab
└── logs/
```

只使用已有 Hugging Face 模型、不允许访问 Hub 时，在 `.env` 添加：

```dotenv
WHISPERX_API_MODEL_CACHE_ONLY=true
HF_HUB_OFFLINE=1
```

缓存不完整会报错。以上设置不约束 TorchAudio 等其他下载来源。
不设置时，缺少的模型会在首次请求时下载到挂载目录。
离线模式仅针对运行时模型访问；首次构建镜像仍需联网下载系统和 Python 依赖、NLTK 资源。

## 4. 构建、启动和验证

```bash
docker compose config --quiet
docker compose up -d --build
docker compose ps
docker compose logs -f api
```

```bash
curl http://localhost:7865/health
curl http://localhost:7865/v1/audio/subtitles \
  -F "file=@/path/to/narration.mp3" \
  -F "language=zh" \
  -F "response_format=srt" \
  -o subtitles.srt
```

Swagger：`http://服务器IP:7865/docs`。
健康检查不加载模型，字幕请求才能验证模型缓存及 GPU 推理。
应用日志写入宿主机 `data/logs/api.log`，默认每个文件最多 10 MiB，保留 5 个备份。

## 5. 更新和停止

```bash
git pull --ff-only
docker compose up -d --build

# 修改 .env 后应用配置。
docker compose up -d

# 停止；保留宿主机 data 目录中的模型和日志。
docker compose down
```
