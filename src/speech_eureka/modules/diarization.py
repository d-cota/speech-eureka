import logging

import torch
import huggingface_hub

from speech_eureka.models.data import DiarizationSegment
from speech_eureka.modules.base import BaseDiarizer

# --- Compat patches for pyannote 3.x with modern dependencies ---

# 1) PyTorch 2.6 made weights_only=True the default; pyannote checkpoints contain
#    objects (TorchVersion, etc.) not in the default allowlist. Force weights_only=False
#    for all torch.load calls so pyannote can load its checkpoints safely.
#    This is acceptable because pyannote models are loaded from trusted HF Hub sources.
_original_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)


torch.load = _patched_torch_load

# 2) use_auth_token was removed from huggingface_hub >= 1.x
_original_hf_hub_download = huggingface_hub.hf_hub_download


def _patched_hf_hub_download(*args, **kwargs):
    if "use_auth_token" in kwargs:
        kwargs["token"] = kwargs.pop("use_auth_token")
    return _original_hf_hub_download(*args, **kwargs)


huggingface_hub.hf_hub_download = _patched_hf_hub_download

from pyannote.audio import Pipeline as PyannotePipeline  # noqa: E402

logger = logging.getLogger(__name__)


class PyannoteDiarizer(BaseDiarizer):
    """Speaker diarization using pyannote.audio."""

    def __init__(
        self,
        model_name: str = "pyannote/speaker-diarization-3.1",
        device: str = "cuda",
        min_speakers: int = 1,
        max_speakers: int = 25,
    ):
        self.device = torch.device(device)
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers

        logger.info(f"Loading diarization model: {model_name}")
        self.pipeline = PyannotePipeline.from_pretrained(model_name)
        self.pipeline.to(self.device)

    def diarize(self, audio_path: str) -> list[DiarizationSegment]:
        logger.info(f"Diarizing: {audio_path}")
        result = self.pipeline(
            audio_path,
            min_speakers=self.min_speakers,
            max_speakers=self.max_speakers,
        )

        segments = []
        for turn, _, speaker in result.itertracks(yield_label=True):
            segments.append(
                DiarizationSegment(
                    start=turn.start,
                    end=turn.end,
                    speaker_label=speaker,
                )
            )

        logger.info(f"Found {len(set(s.speaker_label for s in segments))} speakers")
        return segments
