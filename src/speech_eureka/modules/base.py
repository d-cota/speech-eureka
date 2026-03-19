from abc import ABC, abstractmethod

from speech_eureka.models.data import (
    DiarizationSegment,
    PipelineResult,
    TranscriptionResult,
)


class BaseTranscriber(ABC):
    """Base class for all transcription backends."""

    @abstractmethod
    def transcribe(self, audio_path: str) -> TranscriptionResult:
        ...


class BaseDiarizer(ABC):
    """Base class for all diarization backends."""

    @abstractmethod
    def diarize(self, audio_path: str) -> list[DiarizationSegment]:
        ...


class BaseSpeakerIdentifier(ABC):
    """Base class for speaker identification/verification."""

    @abstractmethod
    def enroll(self, name: str, audio_paths: list[str]) -> None:
        """Enroll a speaker from audio samples."""
        ...

    @abstractmethod
    def identify(self, audio_path: str, start: float, end: float) -> str | None:
        """Identify a speaker from an audio segment. Returns speaker name or None."""
        ...
