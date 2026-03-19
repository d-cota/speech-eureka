import torchaudio


def load_audio(path: str, target_sr: int = 16000) -> tuple:
    """Load audio file and resample to target sample rate.

    Returns (waveform, sample_rate) tuple.
    """
    waveform, sr = torchaudio.load(path)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        sr = target_sr
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform, sr
