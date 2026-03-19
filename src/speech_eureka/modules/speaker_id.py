import logging
from pathlib import Path

import torch
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier

from speech_eureka.models.data import SpeakerProfile
from speech_eureka.modules.base import BaseSpeakerIdentifier

logger = logging.getLogger(__name__)


class EcapaSpeakerIdentifier(BaseSpeakerIdentifier):
    """Speaker identification using SpeechBrain ECAPA-TDNN embeddings."""

    def __init__(
        self,
        model_name: str = "speechbrain/spkrec-ecapa-voxceleb",
        device: str = "cuda",
        similarity_threshold: float = 0.65,
        enrollment_dir: str = "data/enrollments",
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.similarity_threshold = similarity_threshold
        self.enrollment_dir = Path(enrollment_dir)
        self.enrolled_speakers: list[SpeakerProfile] = []

        logger.info(f"Loading speaker ID model: {model_name}")
        self.model = EncoderClassifier.from_hparams(
            source=model_name,
            run_opts={"device": self.device},
        )

    def _get_embedding(self, audio_path: str, start: float = 0.0, end: float = 0.0) -> torch.Tensor:
        waveform, sr = torchaudio.load(audio_path)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if end > start:
            start_sample = int(start * 16000)
            end_sample = int(end * 16000)
            waveform = waveform[:, start_sample:end_sample]

        return self.model.encode_batch(waveform.to(self.device)).squeeze()

    def enroll(self, name: str, audio_paths: list[str]) -> None:
        logger.info(f"Enrolling speaker: {name} from {len(audio_paths)} samples")
        embeddings = [self._get_embedding(p) for p in audio_paths]
        avg_embedding = torch.stack(embeddings).mean(dim=0)
        self.enrolled_speakers.append(
            SpeakerProfile(name=name, embedding=avg_embedding)
        )

    def enroll_from_dir(self) -> None:
        """Enroll all speakers from the enrollment directory.
        Expected structure: enrollment_dir/<speaker_name>/*.wav
        """
        if not self.enrollment_dir.exists():
            logger.warning(f"Enrollment dir not found: {self.enrollment_dir}")
            return

        for speaker_dir in sorted(self.enrollment_dir.iterdir()):
            if speaker_dir.is_dir():
                audio_files = list(speaker_dir.glob("*.wav")) + list(speaker_dir.glob("*.flac"))
                if audio_files:
                    self.enroll(speaker_dir.name, [str(f) for f in audio_files])

    def identify(self, audio_path: str, start: float, end: float) -> str | None:
        if not self.enrolled_speakers:
            return None

        segment_embedding = self._get_embedding(audio_path, start, end)

        best_score = -1.0
        best_name = None
        for profile in self.enrolled_speakers:
            score = torch.nn.functional.cosine_similarity(
                segment_embedding.unsqueeze(0),
                profile.embedding.unsqueeze(0),
            ).item()
            if score > best_score:
                best_score = score
                best_name = profile.name

        if best_score >= self.similarity_threshold:
            return best_name
        return None
