import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from speech_eureka.metrics import evaluate, load_pipeline_result, parse_rttm, save_report
from speech_eureka.metrics.report import generate_text_report

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="evaluate")
def main(cfg: DictConfig) -> None:
    logger.info(f"Evaluate config:\n{OmegaConf.to_yaml(cfg)}")

    if cfg.result is None:
        logger.error("No result file provided. Usage: speech-eureka-eval result=outputs/sample.json")
        sys.exit(1)

    result_path = str(cfg.result)
    if not Path(result_path).exists():
        logger.error(f"Result file not found: {result_path}")
        sys.exit(1)

    # Load pipeline result
    pipeline_result = load_pipeline_result(result_path)
    logger.info(f"Loaded result: {result_path} ({len(pipeline_result.segments)} segments)")

    # Load optional reference files
    reference_rttm = None
    if cfg.ref_rttm is not None:
        ref_rttm_path = str(cfg.ref_rttm)
        if not Path(ref_rttm_path).exists():
            logger.error(f"Reference RTTM not found: {ref_rttm_path}")
            sys.exit(1)
        reference_rttm = parse_rttm(ref_rttm_path)
        logger.info(f"Loaded reference RTTM: {len(reference_rttm)} segments")

    reference_transcript = None
    if cfg.ref_transcript is not None:
        ref_txt_path = str(cfg.ref_transcript)
        if not Path(ref_txt_path).exists():
            logger.error(f"Reference transcript not found: {ref_txt_path}")
            sys.exit(1)
        reference_transcript = Path(ref_txt_path).read_text(encoding="utf-8")
        logger.info(f"Loaded reference transcript: {len(reference_transcript.split())} words")

    # Run evaluation
    eval_result = evaluate(
        result=pipeline_result,
        reference_rttm=reference_rttm,
        reference_transcript=reference_transcript,
        teacher_label=cfg.teacher_label,
        audio_duration=cfg.audio_duration,
        collar=cfg.collar,
        skip_overlap=cfg.skip_overlap,
        strip_fillers=cfg.strip_fillers,
        gap_tolerance=cfg.gap_tolerance,
        transition_window=cfg.transition_window,
    )

    # Print report to stdout
    print(generate_text_report(eval_result))

    # Save report files
    stem = Path(result_path).stem
    output_path = Path(cfg.output.dir) / stem
    save_report(eval_result, str(output_path), fmt=cfg.output.format)
    logger.info(f"Report saved to {output_path}.*")


if __name__ == "__main__":
    main()
