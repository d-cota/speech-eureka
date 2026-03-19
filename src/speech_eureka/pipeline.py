import logging

from hydra.utils import instantiate
from omegaconf import DictConfig

from speech_eureka.models.data import (
    DiarizationSegment,
    IdentifiedSegment,
    PipelineResult,
    TranscribedSegment,
    TranscriptionResult,
)
from speech_eureka.modules.base import BaseDiarizer, BaseSpeakerIdentifier, BaseTranscriber

logger = logging.getLogger(__name__)


class SpeechPipeline:
    """Orchestrates transcription, diarization, and speaker identification."""

    def __init__(
        self,
        cfg: DictConfig,
        transcriber: BaseTranscriber | None = None,
        diarizer: BaseDiarizer | None = None,
        speaker_identifier: BaseSpeakerIdentifier | None = None,
        steps: list[str] | None = None,
    ):
        self.cfg = cfg
        self.steps = steps or ["transcription", "diarization", "speaker_id"]

        self.transcriber = transcriber
        self.diarizer = diarizer
        self.speaker_identifier = speaker_identifier

    def setup(self) -> None:
        """Lazily instantiate modules based on active steps."""
        if "transcription" in self.steps and self.transcriber is None:
            logger.info("Instantiating transcriber...")
            self.transcriber = instantiate(self.cfg.transcription)

        if "diarization" in self.steps and self.diarizer is None:
            logger.info("Instantiating diarizer...")
            self.diarizer = instantiate(self.cfg.diarization)

        if "speaker_id" in self.steps and self.speaker_identifier is None:
            logger.info("Instantiating speaker identifier...")
            self.speaker_identifier = instantiate(self.cfg.speaker_id)
            self.speaker_identifier.enroll_from_dir()

    def process(self, audio_path: str) -> PipelineResult:
        """Run the full pipeline on an audio file."""
        result = PipelineResult(audio_path=audio_path)

        if "transcription" in self.steps:
            result.transcription = self.transcriber.transcribe(audio_path)
            logger.info(f"Transcription done: {len(result.transcription.segments)} segments")

        if "diarization" in self.steps:
            result.diarization = self.diarizer.diarize(audio_path)
            logger.info(f"Diarization done: {len(result.diarization)} segments")

        if result.transcription and result.diarization:
            result.segments = self._align(
                result.transcription, result.diarization, audio_path
            )
            logger.info(f"Alignment done: {len(result.segments)} identified segments")

        return result

    def _align(
        self,
        transcription: TranscriptionResult,
        diarization: list[DiarizationSegment],
        audio_path: str,
    ) -> list[IdentifiedSegment]:
        """Align transcribed segments with diarization and optionally identify speakers."""
        segments = []
        for tseg in transcription.segments:
            speaker_label = self._find_speaker(tseg, diarization)
            speaker_name = None

            if "speaker_id" in self.steps and self.speaker_identifier and speaker_label:
                speaker_name = self.speaker_identifier.identify(
                    audio_path, tseg.start, tseg.end
                )

            segments.append(
                IdentifiedSegment(
                    start=tseg.start,
                    end=tseg.end,
                    text=tseg.text,
                    speaker_label=speaker_label or "UNKNOWN",
                    speaker_name=speaker_name,
                    confidence=tseg.confidence,
                )
            )
        return segments

    @staticmethod
    def _find_speaker(
        tseg: TranscribedSegment,
        diarization: list[DiarizationSegment],
    ) -> str | None:
        """Find the diarization speaker with maximum overlap for a transcribed segment."""
        best_overlap = 0.0
        best_speaker = None
        for dseg in diarization:
            overlap_start = max(tseg.start, dseg.start)
            overlap_end = min(tseg.end, dseg.end)
            overlap = max(0.0, overlap_end - overlap_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = dseg.speaker_label
        return best_speaker
