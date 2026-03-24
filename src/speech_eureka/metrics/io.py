import json
import re
import string
from pathlib import Path

from speech_eureka.models.data import DiarizationSegment, IdentifiedSegment, PipelineResult, TranscriptionResult


def parse_rttm(path: str) -> list[DiarizationSegment]:
    """Parse a RTTM file into DiarizationSegments.

    Expected format (as written by main.py):
        SPEAKER <file> 1 <start> <duration> <NA> <NA> <speaker> <NA> <NA>
    """
    segments = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or not line.startswith("SPEAKER"):
            continue
        fields = line.split()
        start = float(fields[3])
        duration = float(fields[4])
        speaker = fields[7]
        segments.append(DiarizationSegment(start=start, end=start + duration, speaker_label=speaker))
    return segments


def load_pipeline_result(path: str) -> PipelineResult:
    """Load a pipeline result JSON (as written by main.py) into a PipelineResult."""
    data = json.loads(Path(path).read_text())

    segments = [
        IdentifiedSegment(
            start=s["start"],
            end=s["end"],
            text=s["text"],
            speaker_label=s.get("speaker_label", "UNKNOWN"),
            speaker_name=s.get("speaker_name"),
            confidence=s.get("confidence", 0.0),
        )
        for s in data.get("segments", [])
    ]

    transcription = TranscriptionResult(text=data.get("text", ""), segments=[])

    return PipelineResult(
        audio_path=data.get("audio_path", path),
        transcription=transcription,
        segments=segments,
    )


_FILLERS = {"uh", "um", "hmm", "hm", "uh-huh", "mhm"}


def normalize_text(text: str, strip_fillers: bool = False) -> str:
    """Normalize text for WER/CER computation.

    Steps:
        1. Lowercase
        2. Remove punctuation except apostrophes (preserves contractions)
        3. Optionally strip filler words
        4. Collapse whitespace
    """
    text = text.lower()
    # Keep apostrophes, remove everything else in string.punctuation
    punct = string.punctuation.replace("'", "")
    text = text.translate(str.maketrans("", "", punct))
    if strip_fillers:
        words = [w for w in text.split() if w not in _FILLERS]
        text = " ".join(words)
    text = re.sub(r"\s+", " ", text).strip()
    return text
