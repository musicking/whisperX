import gc
import threading
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, Protocol

from whisperx_api.audio.schemas import (
    AlignmentRequest,
    AudioTask,
    DiarizationResult,
    DiarizationTurn,
    PipelineOptions,
    Segment,
    TranscriptionResult,
    Word,
)
from whisperx_api.config import Settings
from whisperx_api.exceptions import ModelUnavailable, UnsupportedOption

ProgressCallback = Callable[[str, float], None]


class SpeechEngine(Protocol):
    def transcribe(
        self,
        audio_path: Path,
        options: PipelineOptions,
        progress: ProgressCallback | None = None,
    ) -> TranscriptionResult: ...

    def align(
        self,
        audio_path: Path,
        request: AlignmentRequest,
        progress: ProgressCallback | None = None,
    ) -> TranscriptionResult: ...

    def diarize(
        self,
        audio_path: Path,
        *,
        min_speakers: int | None,
        max_speakers: int | None,
        return_embeddings: bool,
        progress: ProgressCallback | None = None,
    ) -> DiarizationResult: ...


class WhisperXEngine:
    """Thread-safe, lazy-loading facade around the synchronous WhisperX pipeline."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = threading.RLock()
        self._asr_models: dict[str, Any] = {}
        self._align_models: dict[str, tuple[Any, dict[str, Any]]] = {}
        self._diarize_model: Any | None = None

    def preload(self) -> None:
        self._get_asr_model(self.settings.model_name)

    def _validate_model(self, model_name: str) -> None:
        if model_name not in self.settings.allowed_models:
            raise UnsupportedOption(
                f"Model '{model_name}' is not enabled.",
                details={"allowed_models": list(self.settings.allowed_models)},
            )

    def _get_asr_model(self, model_name: str) -> Any:
        self._validate_model(model_name)
        if model_name not in self._asr_models:
            while len(self._asr_models) >= self.settings.max_loaded_asr_models:
                self._asr_models.pop(next(iter(self._asr_models)))
                self._release_unused_memory()
            try:
                import whisperx

                self._asr_models[model_name] = whisperx.load_model(
                    model_name,
                    device=self.settings.device,
                    device_index=self.settings.device_index,
                    compute_type=self.settings.compute_type,
                    download_root=(
                        str(self.settings.model_dir) if self.settings.model_dir else None
                    ),
                    local_files_only=self.settings.model_cache_only,
                    vad_method=self.settings.vad_method,
                    use_auth_token=self.settings.hf_token,
                )
            except Exception as exc:
                raise ModelUnavailable(f"Unable to load ASR model '{model_name}': {exc}") from exc
        return self._asr_models[model_name]

    def _get_align_model(self, language: str) -> tuple[Any, dict[str, Any]]:
        if language not in self._align_models:
            while len(self._align_models) >= self.settings.max_loaded_align_models:
                self._align_models.pop(next(iter(self._align_models)))
                self._release_unused_memory()
            try:
                import whisperx

                self._align_models[language] = whisperx.load_align_model(
                    language_code=language,
                    device=self.settings.device,
                    model_dir=str(self.settings.model_dir) if self.settings.model_dir else None,
                    model_cache_only=self.settings.model_cache_only,
                )
            except Exception as exc:
                raise UnsupportedOption(
                    f"No usable alignment model is available for language '{language}'."
                ) from exc
        return self._align_models[language]

    def _get_diarize_model(self) -> Any:
        if self._diarize_model is None:
            if not self.settings.hf_token:
                raise UnsupportedOption(
                    "Speaker diarization is disabled because no server-side HF token is configured."
                )
            try:
                from whisperx.diarize import DiarizationPipeline

                self._diarize_model = DiarizationPipeline(
                    model_name=self.settings.diarize_model,
                    token=self.settings.hf_token,
                    device=self.settings.device,
                    cache_dir=(str(self.settings.model_dir) if self.settings.model_dir else None),
                )
            except Exception as exc:
                raise ModelUnavailable(f"Unable to load diarization model: {exc}") from exc
        return self._diarize_model

    def transcribe(
        self,
        audio_path: Path,
        options: PipelineOptions,
        progress: ProgressCallback | None = None,
    ) -> TranscriptionResult:
        import whisperx

        model_name = options.model or self.settings.model_name
        with self._lock:
            if options.vad_method and options.vad_method != self.settings.vad_method:
                raise UnsupportedOption(
                    "VAD method is fixed by the deployment. "
                    f"This server uses '{self.settings.vad_method}'."
                )
            if progress:
                progress("decode", 2)
            audio = whisperx.load_audio(str(audio_path))
            duration = len(audio) / 16_000

            if progress:
                progress("transcribe", 5)
            model = self._get_asr_model(model_name)
            original_asr_options = model.options
            original_vad_params = dict(model._vad_params)
            model.options = replace(
                model.options,
                temperatures=[options.temperature],
                initial_prompt=options.initial_prompt,
                hotwords=options.hotwords,
            )
            model._vad_params.update(
                {
                    "vad_onset": options.vad_onset,
                    "vad_offset": options.vad_offset,
                }
            )
            try:
                result = model.transcribe(
                    audio,
                    batch_size=options.batch_size or self.settings.batch_size,
                    language=options.language,
                    task=options.task.value,
                    chunk_size=options.chunk_size,
                    progress_callback=(
                        (lambda value: progress("transcribe", 5 + value * 0.6))
                        if progress
                        else None
                    ),
                )
            finally:
                model.options = original_asr_options
                model._vad_params = original_vad_params

            if options.align:
                if progress:
                    progress("align", 67)
                align_model, metadata = self._get_align_model(result["language"])
                result = whisperx.align(
                    result["segments"],
                    align_model,
                    metadata,
                    audio,
                    self.settings.device,
                    return_char_alignments=options.return_char_alignments,
                    progress_callback=(
                        (lambda value: progress("align", 67 + value * 0.18)) if progress else None
                    ),
                )
                result["language"] = options.language or metadata["language"]

            if options.diarize:
                if options.return_speaker_embeddings and not self.settings.allow_speaker_embeddings:
                    raise UnsupportedOption("Speaker embeddings are disabled by this deployment.")
                if progress:
                    progress("diarize", 86)
                diarize_result = self._get_diarize_model()(
                    audio,
                    min_speakers=options.min_speakers,
                    max_speakers=options.max_speakers,
                    return_embeddings=options.return_speaker_embeddings,
                    progress_callback=(
                        (lambda value: progress("diarize", 86 + value * 0.13)) if progress else None
                    ),
                )
                if options.return_speaker_embeddings:
                    diarize_segments, embeddings = diarize_result
                else:
                    diarize_segments, embeddings = diarize_result, None
                result = whisperx.assign_word_speakers(
                    diarize_segments,
                    result,
                    speaker_embeddings=embeddings,
                )

            if progress:
                progress("export", 100)
            return self._normalize_result(
                result,
                task=options.task,
                duration=duration,
            )

    def align(
        self,
        audio_path: Path,
        request: AlignmentRequest,
        progress: ProgressCallback | None = None,
    ) -> TranscriptionResult:
        import whisperx

        with self._lock:
            if progress:
                progress("decode", 5)
            audio = whisperx.load_audio(str(audio_path))
            align_model, metadata = self._get_align_model(request.language)
            if progress:
                progress("align", 10)
            result = whisperx.align(
                [segment.model_dump() for segment in request.segments],
                align_model,
                metadata,
                audio,
                self.settings.device,
                return_char_alignments=request.return_char_alignments,
                progress_callback=(
                    (lambda value: progress("align", 10 + value * 0.9)) if progress else None
                ),
            )
            result["language"] = request.language
            return self._normalize_result(
                result,
                task=AudioTask.ALIGN,
                duration=len(audio) / 16_000,
            )

    def diarize(
        self,
        audio_path: Path,
        *,
        min_speakers: int | None,
        max_speakers: int | None,
        return_embeddings: bool,
        progress: ProgressCallback | None = None,
    ) -> DiarizationResult:
        if return_embeddings and not self.settings.allow_speaker_embeddings:
            raise UnsupportedOption("Speaker embeddings are disabled by this deployment.")
        with self._lock:
            output = self._get_diarize_model()(
                str(audio_path),
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                return_embeddings=return_embeddings,
                progress_callback=(
                    (lambda value: progress("diarize", value)) if progress else None
                ),
            )
            if return_embeddings:
                frame, embeddings = output
            else:
                frame, embeddings = output, None
            turns = [
                DiarizationTurn(
                    start=float(row["start"]),
                    end=float(row["end"]),
                    speaker=str(row["speaker"]),
                )
                for _, row in frame.iterrows()
            ]
            return DiarizationResult(turns=turns, speaker_embeddings=embeddings)

    @staticmethod
    def _normalize_result(
        raw: dict[str, Any],
        *,
        task: AudioTask,
        duration: float,
    ) -> TranscriptionResult:
        segments: list[Segment] = []
        for index, raw_segment in enumerate(raw.get("segments", [])):
            words = None
            if "words" in raw_segment:
                words = [
                    Word(
                        word=str(word.get("word", "")),
                        start=word.get("start"),
                        end=word.get("end"),
                        score=word.get("score"),
                        speaker=word.get("speaker"),
                    )
                    for word in raw_segment["words"]
                ]
            segments.append(
                Segment(
                    id=index,
                    start=float(raw_segment.get("start", 0)),
                    end=float(raw_segment.get("end", 0)),
                    text=str(raw_segment.get("text", "")),
                    speaker=raw_segment.get("speaker"),
                    avg_logprob=raw_segment.get("avg_logprob"),
                    words=words,
                    chars=raw_segment.get("chars"),
                )
            )
        word_segments = None
        if raw.get("word_segments") is not None:
            word_segments = [
                Word(
                    word=str(word.get("word", "")),
                    start=word.get("start"),
                    end=word.get("end"),
                    score=word.get("score"),
                    speaker=word.get("speaker"),
                )
                for word in raw["word_segments"]
            ]
        return TranscriptionResult(
            task=task,
            language=str(raw.get("language", "unknown")),
            duration=duration,
            text="".join(segment.text for segment in segments).strip(),
            segments=segments,
            word_segments=word_segments,
            speaker_embeddings=raw.get("speaker_embeddings"),
        )

    def unload(self) -> None:
        with self._lock:
            self._asr_models.clear()
            self._align_models.clear()
            self._diarize_model = None
            self._release_unused_memory()

    @staticmethod
    def _release_unused_memory() -> None:
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
