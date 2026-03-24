import jiwer

from speech_eureka.metrics._types import WERResult
from speech_eureka.metrics.io import normalize_text

_TRANSFORM = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords(),
])


def compute_wer(
    reference: str,
    hypothesis: str,
    strip_fillers: bool = False,
) -> WERResult:
    """Compute WER and CER using jiwer.

    Both strings are normalized before scoring.
    """
    ref = normalize_text(reference, strip_fillers=strip_fillers)
    hyp = normalize_text(hypothesis, strip_fillers=strip_fillers)

    word_out = jiwer.process_words(ref, hyp, reference_transform=_TRANSFORM, hypothesis_transform=_TRANSFORM)
    char_out = jiwer.process_characters(ref, hyp)

    return WERResult(
        wer=word_out.wer,
        cer=char_out.cer,
        insertions=word_out.insertions,
        deletions=word_out.deletions,
        substitutions=word_out.substitutions,
        ref_words=len(ref.split()),
        ref_chars=len(ref.replace(" ", "")),
    )
