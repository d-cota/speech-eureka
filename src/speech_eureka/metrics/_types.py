from dataclasses import dataclass, field


@dataclass
class DERResult:
    """Diarization Error Rate breakdown."""
    der: float                          # overall DER (0–∞, can exceed 1.0)
    missed_speech: float                # missed / total_ref_time
    false_alarm: float                  # false_alarm / total_ref_time
    speaker_confusion: float            # confusion / total_ref_time
    total_ref_time: float               # seconds of reference speech (post-collar)
    collar: float                       # collar applied in seconds
    ref_speaker_count: int
    hyp_speaker_count: int
    speaker_mapping: dict[str, str] = field(default_factory=dict)  # hyp -> ref


@dataclass
class JERResult:
    """Jaccard Error Rate per speaker and overall."""
    jer: float                          # mean JER across reference speakers (0–1)
    per_speaker: dict[str, float] = field(default_factory=dict)  # ref_speaker -> JER


@dataclass
class WERResult:
    """Word and Character Error Rate."""
    wer: float
    cer: float
    insertions: int
    deletions: int
    substitutions: int
    ref_words: int
    ref_chars: int


@dataclass
class SpeakerStats:
    """Per-speaker participation statistics."""
    speaker: str                        # speaker_name if available, else speaker_label
    talk_time: float                    # seconds
    talk_ratio: float                   # fraction of total speech time
    turn_count: int
    avg_turn_duration: float
    question_count: int


@dataclass
class ClassroomMetrics:
    """Classroom-specific evaluation metrics."""
    # Participation
    speakers: list[SpeakerStats] = field(default_factory=list)
    total_speech_time: float = 0.0
    silence_time: float | None = None   # None if audio_duration not provided
    silence_ratio: float | None = None

    # Equity
    gini_coefficient: float = 0.0
    teacher_student_ratio: float | None = None  # teacher_time / total_student_time

    # Discourse
    total_turns: int = 0
    avg_turn_duration: float = 0.0

    # Turn transitions
    interaction_matrix: dict[str, dict[str, int]] = field(default_factory=dict)

    # Overlaps
    overlap_count: int = 0
    overlap_duration: float = 0.0
    overlap_ratio: float = 0.0          # overlap_duration / total_speech_time

    # Questions (heuristic)
    question_count: int = 0
    questions_by_speaker: dict[str, int] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Top-level container for all evaluation metrics."""
    audio_path: str
    der: DERResult | None = None
    jer: JERResult | None = None
    asr: WERResult | None = None
    classroom: ClassroomMetrics | None = None
