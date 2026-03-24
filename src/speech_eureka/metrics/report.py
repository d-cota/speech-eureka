import dataclasses
import json
from pathlib import Path

from speech_eureka.metrics._types import ClassroomMetrics, DERResult, EvaluationResult, JERResult, WERResult


def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def _dur(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s"


def generate_text_report(result: EvaluationResult) -> str:
    lines: list[str] = []
    sep = "=" * 52

    lines += [sep, f"EVALUATION REPORT: {Path(result.audio_path).name}", sep, ""]

    # --- Diarization ---
    if result.der is not None:
        d: DERResult = result.der
        lines += [
            "--- Diarization (DER) ---",
            f"  DER:                {_pct(d.der)}",
            f"  Missed speech:      {_pct(d.missed_speech)}",
            f"  False alarm:        {_pct(d.false_alarm)}",
            f"  Speaker confusion:  {_pct(d.speaker_confusion)}",
            f"  Total ref speech:   {_dur(d.total_ref_time)}",
            f"  Collar:             {d.collar}s",
            f"  Speakers (ref/hyp): {d.ref_speaker_count} / {d.hyp_speaker_count}",
        ]
        if d.speaker_mapping:
            mapping_str = ", ".join(f"{h} -> {r}" for h, r in d.speaker_mapping.items())
            lines.append(f"  Speaker mapping:    {mapping_str}")
        lines.append("")

    if result.jer is not None:
        j: JERResult = result.jer
        lines += [
            "--- Diarization (JER) ---",
            f"  JER (mean):  {_pct(j.jer)}",
        ]
        for spk, val in j.per_speaker.items():
            lines.append(f"    {spk:<20} {_pct(val)}")
        lines.append("")

    # --- ASR ---
    if result.asr is not None:
        a: WERResult = result.asr
        lines += [
            "--- ASR ---",
            f"  WER:  {_pct(a.wer)}",
            f"  CER:  {_pct(a.cer)}",
            f"  Insertions / Deletions / Substitutions: {a.insertions} / {a.deletions} / {a.substitutions}",
            f"  Reference words: {a.ref_words}  chars: {a.ref_chars}",
            "",
        ]

    # --- Classroom ---
    if result.classroom is not None:
        c: ClassroomMetrics = result.classroom
        lines += [
            "--- Classroom Participation ---",
            f"  {'Speaker':<22} {'Talk Time':>9}  {'Ratio':>6}  {'Turns':>5}  {'Avg Turn':>8}  {'Questions':>9}",
            f"  {'-'*22} {'-'*9}  {'-'*6}  {'-'*5}  {'-'*8}  {'-'*9}",
        ]
        for s in c.speakers:
            lines.append(
                f"  {s.speaker:<22} {_dur(s.talk_time):>9}  {_pct(s.talk_ratio):>6}  "
                f"{s.turn_count:>5}  {_dur(s.avg_turn_duration):>8}  {s.question_count:>9}"
            )
        lines += [
            "",
            f"  Gini coefficient:     {c.gini_coefficient:.3f}  (0=equal, 1=one speaker dominates)",
        ]
        if c.teacher_student_ratio is not None:
            lines.append(f"  Teacher:student ratio: {c.teacher_student_ratio:.2f}")
        if c.silence_ratio is not None:
            lines.append(f"  Silence ratio:        {_pct(c.silence_ratio)}"
                         f"  ({_dur(c.silence_time or 0)})")
        lines += [
            f"  Overlap ratio:        {_pct(c.overlap_ratio)}"
            f"  ({c.overlap_count} overlapping segments, {c.overlap_duration:.1f}s)",
            f"  Total turns:          {c.total_turns}  (avg {_dur(c.avg_turn_duration)} each)",
            f"  Questions detected:   {c.question_count}",
        ]
        lines.append("")

        if c.interaction_matrix:
            lines += ["--- Speaker Transitions (who -> who) ---"]
            speakers = sorted(c.interaction_matrix)
            all_targets = sorted({t for v in c.interaction_matrix.values() for t in v})
            from_to = "From \\ To"
            header = f"  {from_to:<20}" + "".join(f"{t[:8]:>10}" for t in all_targets)
            lines.append(header)
            for spk in speakers:
                row = f"  {spk:<20}" + "".join(
                    f"{c.interaction_matrix[spk].get(t, 0):>10}" for t in all_targets
                )
                lines.append(row)
            lines.append("")

    return "\n".join(lines)


def _round_floats(obj, decimals: int = 4):
    """Recursively round floats in a nested dict/list structure."""
    if isinstance(obj, float):
        return round(obj, decimals)
    if isinstance(obj, dict):
        return {k: _round_floats(v, decimals) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_floats(v, decimals) for v in obj]
    return obj


def generate_json_report(result: EvaluationResult) -> dict:
    """Return a JSON-serialisable dict of all metrics."""
    raw = dataclasses.asdict(result)
    return _round_floats(raw)


def save_report(result: EvaluationResult, output_path: str, fmt: str = "both") -> None:
    """Save text and/or JSON report to output_path (without extension)."""
    base = Path(output_path)
    base.parent.mkdir(parents=True, exist_ok=True)

    if fmt in ("text", "both"):
        text_path = base.with_suffix(".txt")
        text_path.write_text(generate_text_report(result))

    if fmt in ("json", "both"):
        json_path = base.with_suffix(".json")
        json_path.write_text(json.dumps(generate_json_report(result), indent=2, ensure_ascii=False))
