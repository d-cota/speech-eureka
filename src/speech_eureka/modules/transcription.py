import logging

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from speech_eureka.models.data import TranscribedSegment, TranscriptionResult
from speech_eureka.modules.base import BaseTranscriber

logger = logging.getLogger(__name__)


class WhisperTranscriber(BaseTranscriber):
    """Transcription using HuggingFace Whisper models."""

    def __init__(
        self,
        model_name: str = "openai/whisper-large-v3",
        device: str = "cuda",
        batch_size: int = 16,
        language: str | None = None,
        compute_type: str = "float16",
        return_timestamps: bool = True,
        chunk_length_s: int = 30,
    ):
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        self.language = language
        self.return_timestamps = return_timestamps
        self.chunk_length_s = chunk_length_s

        dtype = torch.float16 if compute_type == "float16" else torch.float32

        logger.info(f"Loading transcription model: {model_name} on {self.device}")
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name,
            dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(self.device)

        processor = AutoProcessor.from_pretrained(model_name)

        generate_kwargs = {}
        if language:
            generate_kwargs["language"] = language

        pipe_kwargs = dict(
            task="automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            dtype=dtype,
            device=self.device,
            batch_size=batch_size,
            return_timestamps=return_timestamps,
        )
        if generate_kwargs:
            pipe_kwargs["generate_kwargs"] = generate_kwargs

        self.pipe = pipeline(**pipe_kwargs)

    def transcribe(self, audio_path: str) -> TranscriptionResult:
        logger.info(f"Transcribing: {audio_path}")
        result = self.pipe(audio_path)

        segments = []
        if "chunks" in result:
            for chunk in result["chunks"]:
                ts = chunk.get("timestamp", (0.0, 0.0))
                segments.append(
                    TranscribedSegment(
                        start=ts[0] if ts[0] is not None else 0.0,
                        end=ts[1] if ts[1] is not None else 0.0,
                        text=chunk["text"].strip(),
                    )
                )

        return TranscriptionResult(
            text=result["text"].strip(),
            segments=segments,
        )
