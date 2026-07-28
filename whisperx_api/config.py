from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(StrEnum):
    LOCAL = "local"
    TEST = "test"
    STAGING = "staging"
    PRODUCTION = "production"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="WHISPERX_API_",
        extra="ignore",
    )

    app_name: str = "WhisperX API"
    app_version: str = "0.1.0"
    environment: Environment = Environment.LOCAL
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    log_level: str = "info"
    show_docs: bool | None = None

    data_dir: Path = Path("./data")
    database_path: Path | None = None
    max_upload_bytes: int = Field(default=1_073_741_824, ge=1)
    upload_chunk_bytes: int = Field(default=1_048_576, ge=65_536)
    sync_via_worker: bool = True
    sync_timeout_seconds: float = Field(default=900, ge=1, le=86_400)
    sync_poll_seconds: float = Field(default=0.25, ge=0.05, le=10)

    api_key: str | None = None
    model_name: str = "small"
    allowed_models: tuple[str, ...] = ("tiny", "base", "small", "medium", "large-v3")
    max_loaded_asr_models: int = Field(default=1, ge=1, le=4)
    max_loaded_align_models: int = Field(default=3, ge=1, le=32)
    device: str = "cuda"
    device_index: int = Field(default=0, ge=0)
    compute_type: str = "default"
    batch_size: int = Field(default=8, ge=1, le=128)
    model_dir: Path | None = None
    model_cache_only: bool = False
    vad_method: str = "pyannote"
    hf_token: str | None = None
    diarize_model: str = "pyannote/speaker-diarization-community-1"
    allow_speaker_embeddings: bool = False
    preload_model: bool = False

    worker_poll_seconds: float = Field(default=0.5, ge=0.05, le=60)
    worker_id: str | None = None
    require_worker_for_readiness: bool = True

    @field_validator("compute_type")
    @classmethod
    def valid_compute_type(cls, value: str) -> str:
        allowed = {"default", "float16", "float32", "int8"}
        if value not in allowed:
            raise ValueError(f"compute_type must be one of {sorted(allowed)}")
        return value

    @field_validator("vad_method")
    @classmethod
    def valid_vad_method(cls, value: str) -> str:
        if value not in {"pyannote", "silero"}:
            raise ValueError("vad_method must be pyannote or silero")
        return value

    def model_post_init(self, __context: object) -> None:
        self.data_dir = self.data_dir.resolve()
        if self.database_path is None:
            self.database_path = self.data_dir / "whisperx-api.sqlite3"
        else:
            self.database_path = self.database_path.resolve()
        if self.model_dir is not None:
            self.model_dir = self.model_dir.resolve()

    @property
    def docs_enabled(self) -> bool:
        if self.show_docs is not None:
            return self.show_docs
        return self.environment in {Environment.LOCAL, Environment.TEST, Environment.STAGING}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
