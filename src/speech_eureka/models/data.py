from dataclasses import dataclass, field


@dataclass
class AudioSegment:
    """A segment of audio with timing information."""
    start: float
    end: float
    waveform: object = None  # torch.Tensor when loaded
    sample_rate: int = 16000

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class TranscriptionResult:
    """Result from the transcription module."""
    text: str
    segments: list["TranscribedSegment"] = field(default_factory=list)
    language: str | None = None


@dataclass
class TranscribedSegment:
    """A single transcribed segment with timestamps."""
    start: float
    end: float
    text: str
    confidence: float = 0.0


@dataclass
class DiarizationSegment:
    """A segment assigned to a speaker by diarization."""
    start: float
    end: float
    speaker_label: str  # e.g. "SPEAKER_00"


@dataclass
class SpeakerProfile:
    """A known speaker with an embedding for identification."""
    name: str
    embedding: object = None  # torch.Tensor


@dataclass
class IdentifiedSegment:
    """A fully processed segment: transcribed, diarized, and speaker-identified."""
    start: float
    end: float
    text: str
    speaker_label: str
    speaker_name: str | None = None
    confidence: float = 0.0


@dataclass
class PipelineResult:
    """Full pipeline output for an audio file."""
    audio_path: str
    transcription: TranscriptionResult | None = None
    diarization: list[DiarizationSegment] = field(default_factory=list)
    segments: list[IdentifiedSegment] = field(default_factory=list)
