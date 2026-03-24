from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate

from speech_eureka.metrics._types import DERResult, JERResult
from speech_eureka.models.data import DiarizationSegment


def _to_annotation(segments: list[DiarizationSegment], uri: str = "utt") -> Annotation:
    annotation = Annotation(uri=uri)
    for seg in segments:
        annotation[Segment(seg.start, seg.end)] = seg.speaker_label
    return annotation


def compute_der(
    reference: list[DiarizationSegment],
    hypothesis: list[DiarizationSegment],
    collar: float = 0.25,
    skip_overlap: bool = False,
) -> DERResult:
    """Compute DER using pyannote.metrics.

    Args:
        collar: Collar in seconds applied around reference speaker boundaries.
        skip_overlap: If True, ignore overlapping reference speech regions.
    """
    ref = _to_annotation(reference)
    hyp = _to_annotation(hypothesis)

    metric = DiarizationErrorRate(collar=collar, skip_overlap=skip_overlap)
    components = metric(ref, hyp, detailed=True)

    total_ref = components["total"]
    missed = components["missed detection"]
    fa = components["false alarm"]
    confusion = components["confusion"]

    # Recover optimal speaker mapping from the metric internals
    mapping = {}
    try:
        mapping = metric.optimal_mapping(ref, hyp)
        mapping = {v: k for k, v in mapping.items()}  # hyp -> ref
    except Exception:
        pass

    return DERResult(
        der=components["diarization error rate"],
        missed_speech=missed / total_ref if total_ref > 0 else 0.0,
        false_alarm=fa / total_ref if total_ref > 0 else 0.0,
        speaker_confusion=confusion / total_ref if total_ref > 0 else 0.0,
        total_ref_time=total_ref,
        collar=collar,
        ref_speaker_count=len(ref.labels()),
        hyp_speaker_count=len(hyp.labels()),
        speaker_mapping=mapping,
    )


def compute_jer(
    reference: list[DiarizationSegment],
    hypothesis: list[DiarizationSegment],
    collar: float = 0.25,
) -> JERResult:
    """Compute JER using pyannote.metrics."""
    ref = _to_annotation(reference)
    hyp = _to_annotation(hypothesis)

    metric = JaccardErrorRate(collar=collar)
    components = metric(ref, hyp, detailed=True)

    # Per-speaker JER via individual evaluation
    per_speaker: dict[str, float] = {}
    for label in ref.labels():
        ref_single = Annotation(uri="utt")
        for seg, _, spk in ref.itertracks(yield_label=True):
            if spk == label:
                ref_single[seg] = spk
        hyp_filtered = hyp.subset(set(components.get("_mapping", {label: label}).values()))
        try:
            per_speaker[label] = float(metric(ref_single, hyp_filtered))
        except Exception:
            per_speaker[label] = 1.0

    return JERResult(
        jer=float(components["jaccard error rate"]),
        per_speaker=per_speaker,
    )
