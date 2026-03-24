import numpy as np

from speech_eureka.metrics._types import ClassroomMetrics, SpeakerStats
from speech_eureka.models.data import IdentifiedSegment

_QUESTION_WORDS = {
    "what", "where", "when", "who", "why", "how", "which", "whose", "whom",
    "is", "are", "was", "were", "do", "does", "did",
    "can", "could", "will", "would", "should", "have", "has", "had",
}


def _resolve_speaker(seg: IdentifiedSegment) -> str:
    return seg.speaker_name or seg.speaker_label


def _merge_turns(
    segments: list[IdentifiedSegment],
    gap_tolerance: float = 0.3,
) -> list[tuple[str, float, float]]:
    """Merge consecutive same-speaker segments within gap_tolerance into turns.

    Returns list of (speaker, start, end).
    """
    if not segments:
        return []

    sorted_segs = sorted(segments, key=lambda s: s.start)
    turns: list[tuple[str, float, float]] = []
    cur_spk = _resolve_speaker(sorted_segs[0])
    cur_start = sorted_segs[0].start
    cur_end = sorted_segs[0].end

    for seg in sorted_segs[1:]:
        spk = _resolve_speaker(seg)
        if spk == cur_spk and seg.start - cur_end <= gap_tolerance:
            cur_end = max(cur_end, seg.end)
        else:
            turns.append((cur_spk, cur_start, cur_end))
            cur_spk = spk
            cur_start = seg.start
            cur_end = seg.end

    turns.append((cur_spk, cur_start, cur_end))
    return turns


def _gini(values: list[float]) -> float:
    """Gini coefficient of a list of non-negative values (0 = equal, 1 = maximal inequality)."""
    if not values or sum(values) == 0:
        return 0.0
    arr = np.sort(np.array(values, dtype=float))
    n = len(arr)
    idx = np.arange(1, n + 1)
    return float((2 * np.sum(idx * arr)) / (n * np.sum(arr)) - (n + 1) / n)


def _is_question(text: str) -> bool:
    stripped = text.strip()
    if stripped.endswith("?"):
        return True
    words = stripped.lower().split()
    return bool(words) and words[0] in _QUESTION_WORDS and "?" in stripped


def compute_classroom_metrics(
    segments: list[IdentifiedSegment],
    audio_duration: float | None = None,
    teacher_label: str | None = None,
    gap_tolerance: float = 0.3,
    transition_window: float = 5.0,
) -> ClassroomMetrics:
    """Compute all classroom-specific metrics from identified segments.

    Args:
        segments: Pipeline output segments (transcribed + diarized + identified).
        audio_duration: Total audio length in seconds; required for silence_ratio.
        teacher_label: speaker_name or speaker_label of the teacher for ratio computation.
        gap_tolerance: Max gap (s) between same-speaker segments to merge into one turn.
        transition_window: Max time (s) between turns to count as a discourse transition.
    """
    if not segments:
        return ClassroomMetrics()

    turns = _merge_turns(segments, gap_tolerance)
    sorted_segs = sorted(segments, key=lambda s: s.start)

    # --- Per-speaker stats ---
    speaker_time: dict[str, float] = {}
    speaker_turns: dict[str, list[float]] = {}
    for spk, start, end in turns:
        dur = end - start
        speaker_time[spk] = speaker_time.get(spk, 0.0) + dur
        speaker_turns.setdefault(spk, []).append(dur)

    total_speech_time = sum(speaker_time.values())

    speaker_stats = [
        SpeakerStats(
            speaker=spk,
            talk_time=t,
            talk_ratio=t / total_speech_time if total_speech_time > 0 else 0.0,
            turn_count=len(speaker_turns[spk]),
            avg_turn_duration=float(np.mean(speaker_turns[spk])),
            question_count=0,  # filled below
        )
        for spk, t in sorted(speaker_time.items(), key=lambda x: -x[1])
    ]

    # --- Gini ---
    gini = _gini(list(speaker_time.values()))

    # --- Teacher-student ratio ---
    teacher_student_ratio: float | None = None
    if teacher_label is not None:
        teacher_time = speaker_time.get(teacher_label, 0.0)
        student_time = sum(t for spk, t in speaker_time.items() if spk != teacher_label)
        if student_time > 0:
            teacher_student_ratio = teacher_time / student_time

    # --- Silence ---
    silence_time: float | None = None
    silence_ratio: float | None = None
    if audio_duration is not None:
        # Merge overlapping speech intervals to compute actual speech coverage
        intervals = sorted((s.start, s.end) for s in sorted_segs)
        merged_end = -1.0
        merged_total = 0.0
        for start, end in intervals:
            if start > merged_end:
                merged_total += end - start
                merged_end = end
            else:
                if end > merged_end:
                    merged_total += end - merged_end
                    merged_end = end
        silence_time = max(0.0, audio_duration - merged_total)
        silence_ratio = silence_time / audio_duration if audio_duration > 0 else 0.0

    # --- Interaction matrix (who speaks after whom) ---
    interaction_matrix: dict[str, dict[str, int]] = {}
    for i in range(len(turns) - 1):
        spk_a, _, end_a = turns[i]
        spk_b, start_b, _ = turns[i + 1]
        if spk_a != spk_b and start_b - end_a <= transition_window:
            interaction_matrix.setdefault(spk_a, {})
            interaction_matrix[spk_a][spk_b] = interaction_matrix[spk_a].get(spk_b, 0) + 1

    # --- Overlaps (segments from different speakers with overlapping intervals) ---
    overlap_count = 0
    overlap_duration = 0.0
    for i in range(len(sorted_segs)):
        for j in range(i + 1, len(sorted_segs)):
            a, b = sorted_segs[i], sorted_segs[j]
            if b.start >= a.end:
                break
            if _resolve_speaker(a) != _resolve_speaker(b):
                overlap_start = max(a.start, b.start)
                overlap_end = min(a.end, b.end)
                if overlap_end > overlap_start:
                    overlap_count += 1
                    overlap_duration += overlap_end - overlap_start

    overlap_ratio = overlap_duration / total_speech_time if total_speech_time > 0 else 0.0

    # --- Questions ---
    questions_by_speaker: dict[str, int] = {}
    for seg in sorted_segs:
        if _is_question(seg.text):
            spk = _resolve_speaker(seg)
            questions_by_speaker[spk] = questions_by_speaker.get(spk, 0) + 1
    question_count = sum(questions_by_speaker.values())

    # Fill question_count into speaker_stats
    stats_map = {s.speaker: s for s in speaker_stats}
    for spk, qc in questions_by_speaker.items():
        if spk in stats_map:
            stats_map[spk].question_count = qc

    return ClassroomMetrics(
        speakers=speaker_stats,
        total_speech_time=total_speech_time,
        silence_time=silence_time,
        silence_ratio=silence_ratio,
        gini_coefficient=gini,
        teacher_student_ratio=teacher_student_ratio,
        total_turns=len(turns),
        avg_turn_duration=float(np.mean([e - s for _, s, e in turns])) if turns else 0.0,
        interaction_matrix=interaction_matrix,
        overlap_count=overlap_count,
        overlap_duration=overlap_duration,
        overlap_ratio=overlap_ratio,
        question_count=question_count,
        questions_by_speaker=questions_by_speaker,
    )
