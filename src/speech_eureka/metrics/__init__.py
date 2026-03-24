from speech_eureka.metrics._types import (
    ClassroomMetrics,
    DERResult,
    EvaluationResult,
    JERResult,
    SpeakerStats,
    WERResult,
)
from speech_eureka.metrics.classroom import compute_classroom_metrics
from speech_eureka.metrics.diarization import compute_der, compute_jer
from speech_eureka.metrics.io import load_pipeline_result, normalize_text, parse_rttm
from speech_eureka.metrics.report import generate_json_report, generate_text_report, save_report
from speech_eureka.metrics.transcription import compute_wer
from speech_eureka.models.data import DiarizationSegment, PipelineResult


def evaluate(
    result: PipelineResult,
    reference_rttm: list[DiarizationSegment] | None = None,
    reference_transcript: str | None = None,
    teacher_label: str | None = None,
    audio_duration: float | None = None,
    collar: float = 0.25,
    skip_overlap: bool = False,
    strip_fillers: bool = False,
    gap_tolerance: float = 0.3,
    transition_window: float = 5.0,
) -> EvaluationResult:
    """Run all applicable evaluation metrics on a pipeline result.

    Args:
        result: Output from SpeechPipeline.process().
        reference_rttm: Ground-truth diarization segments; enables DER/JER.
        reference_transcript: Ground-truth transcript string; enables WER/CER.
        teacher_label: Speaker name/label identifying the teacher; enables teacher-student ratio.
        audio_duration: Total audio length in seconds; enables silence_ratio.
        collar: DER/JER collar in seconds around reference boundaries.
        skip_overlap: Ignore overlapping regions in DER computation.
        strip_fillers: Strip filler words (uh, um, hmm) before WER/CER.
        gap_tolerance: Max gap (s) to merge same-speaker segments into one turn.
        transition_window: Max gap (s) between turns to count as a discourse transition.
    """
    der_result = None
    jer_result = None
    if reference_rttm is not None:
        # Use raw diarization if available, otherwise reconstruct from identified segments
        hyp_diarization = result.diarization or [
            DiarizationSegment(start=s.start, end=s.end, speaker_label=s.speaker_label)
            for s in result.segments
        ]
        if hyp_diarization:
            der_result = compute_der(reference_rttm, hyp_diarization, collar=collar, skip_overlap=skip_overlap)
            jer_result = compute_jer(reference_rttm, hyp_diarization, collar=collar)

    asr_result = None
    if reference_transcript is not None and result.transcription:
        asr_result = compute_wer(
            reference_transcript,
            result.transcription.text,
            strip_fillers=strip_fillers,
        )

    classroom_result = None
    if result.segments:
        classroom_result = compute_classroom_metrics(
            result.segments,
            audio_duration=audio_duration,
            teacher_label=teacher_label,
            gap_tolerance=gap_tolerance,
            transition_window=transition_window,
        )

    return EvaluationResult(
        audio_path=result.audio_path,
        der=der_result,
        jer=jer_result,
        asr=asr_result,
        classroom=classroom_result,
    )


__all__ = [
    "evaluate",
    "EvaluationResult",
    "DERResult",
    "JERResult",
    "WERResult",
    "SpeakerStats",
    "ClassroomMetrics",
    "compute_der",
    "compute_jer",
    "compute_wer",
    "compute_classroom_metrics",
    "load_pipeline_result",
    "parse_rttm",
    "normalize_text",
    "generate_text_report",
    "generate_json_report",
    "save_report",
]
