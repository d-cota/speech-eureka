import json
import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from speech_eureka.pipeline import SpeechPipeline

logger = logging.getLogger(__name__)


def save_result(result, output_dir: str, fmt: str = "json") -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(result.audio_path).stem

    if fmt == "json":
        data = {
            "audio_path": result.audio_path,
            "text": result.transcription.text if result.transcription else "",
            "segments": [
                {
                    "start": s.start,
                    "end": s.end,
                    "text": s.text,
                    "speaker_label": s.speaker_label,
                    "speaker_name": s.speaker_name,
                }
                for s in result.segments
            ],
        }
        path = out / f"{stem}.json"
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    elif fmt == "txt":
        lines = []
        for s in result.segments:
            name = s.speaker_name or s.speaker_label
            lines.append(f"[{s.start:.1f}-{s.end:.1f}] {name}: {s.text}")
        path = out / f"{stem}.txt"
        path.write_text("\n".join(lines))
    elif fmt == "rttm":
        lines = []
        for s in result.segments:
            duration = s.end - s.start
            label = s.speaker_name or s.speaker_label
            lines.append(
                f"SPEAKER {stem} 1 {s.start:.3f} {duration:.3f} <NA> <NA> {label} <NA> <NA>"
            )
        path = out / f"{stem}.rttm"
        path.write_text("\n".join(lines))

    logger.info(f"Saved output to {path}")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    audio_files = list(cfg.audio.paths) if cfg.audio.paths else []
    if not audio_files:
        logger.error(
            "No audio files provided. Usage: speech-eureka '+audio.paths=[file1.wav,file2.wav]'"
        )
        sys.exit(1)

    pipeline = SpeechPipeline(cfg=cfg, steps=cfg.pipeline.steps)
    pipeline.setup()

    for audio_path in audio_files:
        audio_path = str(audio_path)
        if not Path(audio_path).exists():
            logger.warning(f"File not found, skipping: {audio_path}")
            continue
        logger.info(f"Processing: {audio_path}")
        result = pipeline.process(audio_path)
        save_result(result, cfg.output.dir, cfg.output.format)


if __name__ == "__main__":
    main()
