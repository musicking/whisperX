import gc
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any

from whisperx.api.audio.schemas import (
    AlignmentRequest,
    AlignmentSegment,
    AudioTask,
    DiarizationResult,
    DiarizationTurn,
    LanguageResult,
    PipelineOptions,
    Segment,
    TranscriptionResult,
    Word,
)
from whisperx.api.config import Settings
from whisperx.api.exceptions import InvalidAudio, ModelUnavailable, UnsupportedOption


def _load_audio(path: Path):
    import whisperx

    try:
        return whisperx.load_audio(str(path))
    except RuntimeError as exc:
        # WhisperX reports FFmpeg decode failures as RuntimeError.
        raise InvalidAudio("Unable to decode the uploaded audio.") from exc


class WhisperXEngine:
    """Thread-safe, lazy-loading facade around the synchronous WhisperX pipeline."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._torch_device = f"cuda:{settings.device_index}" if settings.device == "cuda" else "cpu"
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
                    download_root=str(self.settings.model_dir) if self.settings.model_dir else None,
                    local_files_only=self.settings.model_cache_only,
                    vad_method=self.settings.vad_method,
                    use_auth_token=self.settings.hf_token,
                )
            except (OSError, RuntimeError, ValueError) as exc:
                raise ModelUnavailable(f"Unable to load ASR model '{model_name}': {exc}") from exc
        return self._asr_models[model_name]

    def _get_align_model(self, language: str) -> tuple[Any, dict[str, Any]]:
        from whisperx.alignment import DEFAULT_ALIGN_MODELS_HF, DEFAULT_ALIGN_MODELS_TORCH

        if language not in DEFAULT_ALIGN_MODELS_TORCH and language not in DEFAULT_ALIGN_MODELS_HF:
            raise UnsupportedOption(f"No default alignment model for language '{language}'.")
        if language not in self._align_models:
            while len(self._align_models) >= self.settings.max_loaded_align_models:
                self._align_models.pop(next(iter(self._align_models)))
                self._release_unused_memory()
            try:
                import whisperx

                self._align_models[language] = whisperx.load_align_model(
                    language_code=language,
                    device=self._torch_device,
                    model_dir=str(self.settings.model_dir) if self.settings.model_dir else None,
                    model_cache_only=self.settings.model_cache_only,
                )
            except (OSError, RuntimeError, ValueError) as exc:
                raise ModelUnavailable(
                    f"No usable alignment model is available for language '{language}'."
                ) from exc
        return self._align_models[language]

    def _get_diarize_model(self) -> Any:
        if self._diarize_model is None:
            try:
                from whisperx.diarize import DiarizationPipeline

                self._diarize_model = DiarizationPipeline(
                    model_name=self.settings.diarize_model,
                    token=self.settings.hf_token,
                    device=self._torch_device,
                    cache_dir=str(self.settings.model_dir) if self.settings.model_dir else None,
                )
            except (OSError, RuntimeError, ValueError) as exc:
                raise ModelUnavailable(f"Unable to load diarization model: {exc}") from exc
        return self._diarize_model

    def transcribe(
        self, audio_path: Path, options: PipelineOptions, *, document_text: str | None = None
    ) -> TranscriptionResult:
        import whisperx

        model_name = options.model or self.settings.model_name
        if document_text is not None:
            if not document_text.strip():
                raise UnsupportedOption("document_text must not be blank")
            if not options.align or options.task != AudioTask.TRANSCRIBE:
                raise UnsupportedOption("document_text requires aligned transcription")
        self._validate_model(model_name)
        if options.diarize:
            self._validate_diarization(options.return_speaker_embeddings)
        with self._lock:
            audio = _load_audio(audio_path)
            duration = len(audio) / 16000
            model = self._get_asr_model(model_name)
            # Restore mutable pipeline state so a failed request cannot affect the next.
            original_asr_options = model.options
            original_tokenizer = model.tokenizer
            model.options = replace(
                model.options,
                temperatures=[options.temperature],
                initial_prompt=options.initial_prompt,
                hotwords=options.hotwords,
            )
            try:
                result = model.transcribe(
                    audio,
                    batch_size=options.batch_size or self.settings.batch_size,
                    language=options.language,
                    task=options.task.value,
                    chunk_size=options.chunk_size,
                )
            finally:
                model.options = original_asr_options
                model.tokenizer = original_tokenizer
            if options.align:
                if document_text is not None:
                    from whisperx.api.audio.document import match_document

                    # ASR locates the narration; only original script text goes to align().
                    result["segments"] = [
                        segment.model_dump()
                        for segment in match_document(
                            document_text,
                            [
                                AlignmentSegment(
                                    start=segment["start"], end=segment["end"], text=segment["text"]
                                )
                                for segment in result["segments"]
                            ],
                        )
                    ]
                (align_model, metadata) = self._get_align_model(result["language"])
                result = whisperx.align(
                    result["segments"],
                    align_model,
                    metadata,
                    audio,
                    self._torch_device,
                    return_char_alignments=options.return_char_alignments,
                )
                result["language"] = options.language or metadata["language"]
            if options.diarize:
                diarize_result = self._get_diarize_model()(
                    audio,
                    min_speakers=options.min_speakers,
                    max_speakers=options.max_speakers,
                    return_embeddings=options.return_speaker_embeddings,
                )
                if options.return_speaker_embeddings:
                    (diarize_segments, embeddings) = diarize_result
                else:
                    (diarize_segments, embeddings) = (diarize_result, None)
                result = whisperx.assign_word_speakers(
                    diarize_segments, result, speaker_embeddings=embeddings
                )
            normalized = self._normalize_result(result, task=options.task, duration=duration)
            if document_text is not None:
                from whisperx.api.audio.document import validate_document_alignment

                validate_document_alignment(document_text, normalized)
                normalized.text = document_text
            return normalized

    def align(self, audio_path: Path, request: AlignmentRequest) -> TranscriptionResult:
        import whisperx

        with self._lock:
            audio = _load_audio(audio_path)
            (align_model, metadata) = self._get_align_model(request.language)
            result = whisperx.align(
                [segment.model_dump() for segment in request.segments],
                align_model,
                metadata,
                audio,
                self._torch_device,
                return_char_alignments=request.return_char_alignments,
            )
            result["language"] = request.language
            return self._normalize_result(result, task=AudioTask.ALIGN, duration=len(audio) / 16000)

    def diarize(
        self,
        audio_path: Path,
        *,
        min_speakers: int | None,
        max_speakers: int | None,
        return_embeddings: bool,
        num_speakers: int | None = None,
    ) -> DiarizationResult:
        self._validate_diarization(return_embeddings)
        with self._lock:
            output = self._get_diarize_model()(
                _load_audio(audio_path),
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                return_embeddings=return_embeddings,
            )
            if return_embeddings:
                (frame, embeddings) = output
            else:
                (frame, embeddings) = (output, None)
            turns = [
                DiarizationTurn(
                    start=float(row["start"]), end=float(row["end"]), speaker=str(row["speaker"])
                )
                for (_, row) in frame.iterrows()
            ]
            return DiarizationResult(turns=turns, speaker_embeddings=embeddings)

    def _validate_diarization(self, return_embeddings: bool) -> None:
        if not self.settings.hf_token:
            raise UnsupportedOption("Speaker diarization requires a server-side HF token.")
        if return_embeddings and not self.settings.allow_speaker_embeddings:
            raise UnsupportedOption("Speaker embeddings are disabled by this deployment.")

    def detect_language(self, audio_path: Path, *, model: str | None = None) -> LanguageResult:

        with self._lock:
            pipeline = self._get_asr_model(model or self.settings.model_name)
            audio = _load_audio(audio_path)
            if len(audio) == 0:
                raise UnsupportedOption("Audio is empty")
            return LanguageResult(language=pipeline.detect_language(audio))

    @staticmethod
    def _normalize_result(
        raw: dict[str, Any], *, task: AudioTask, duration: float
    ) -> TranscriptionResult:
        from whisperx.utils import LANGUAGES_WITHOUT_SPACES

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
        language = str(raw.get("language", "unknown"))
        # Sentence slices returned by alignment do not retain their boundary whitespace.
        output_language = "en" if task == AudioTask.TRANSLATE else language
        separator = "" if output_language in LANGUAGES_WITHOUT_SPACES else " "
        text = separator.join(segment.text.strip() for segment in segments if segment.text.strip())
        return TranscriptionResult(
            task=task,
            language=language,
            duration=duration,
            text=text,
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
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
