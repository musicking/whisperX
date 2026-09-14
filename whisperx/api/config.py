from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="WHISPERX_API_", extra="ignore")

    host: str = "0.0.0.0"
    port: int = Field(default=7865, ge=1, le=65535)
    log_level: Literal["debug", "info", "warning", "error", "critical"] = "info"
    log_file: Path = Path("./data/logs/api.log")
    log_max_bytes: int = Field(default=10_485_760, ge=1)
    log_backup_count: int = Field(default=5, ge=1)
    inference_concurrency: int = Field(default=1, ge=1, le=8)
    environment: Literal["local", "test", "staging", "production"] = "local"
    show_docs: bool | None = None
    data_dir: Path = Path("./data")
    upload_chunk_bytes: int = Field(default=1_048_576, ge=65_536)
    model_name: str = "small"
    allowed_models: tuple[str, ...] = ("tiny", "base", "small", "medium", "large-v3")
    device: Literal["cpu", "cuda"] = "cuda"
    device_index: int = Field(default=0, ge=0)
    compute_type: Literal["default", "float16", "float32", "int8"] = "default"
    batch_size: int = Field(default=8, ge=1, le=128)
    model_dir: Path | None = None
    model_cache_only: bool = False
    max_loaded_asr_models: int = Field(default=1, ge=1, le=4)
    max_loaded_align_models: int = Field(default=3, ge=1, le=32)
    vad_method: Literal["pyannote", "silero"] = "pyannote"
    hf_token: str | None = None
    diarize_model: str = "pyannote/speaker-diarization-community-1"
    allow_speaker_embeddings: bool = False
    preload_model: bool = False

    @field_validator("hf_token", mode="before")
    @classmethod
    def empty_hf_token(cls, value):
        return None if isinstance(value, str) and not value.strip() else value

    @model_validator(mode="after")
    def validate_model(self) -> "Settings":
        if self.model_name not in self.allowed_models:
            raise ValueError("model_name must be included in allowed_models")
        return self

    @property
    def docs_enabled(self) -> bool:
        return self.show_docs if self.show_docs is not None else self.environment != "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
