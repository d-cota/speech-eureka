import json
import logging
import os
import tempfile

import gradio as gr
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

from speech_eureka.pipeline import SpeechPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global pipeline instance (lazy loaded)
_pipeline: SpeechPipeline | None = None
_current_steps: list[str] = []


def get_pipeline(steps: list[str], model_size: str) -> SpeechPipeline:
    """Get or create pipeline, reinitializing if config changed."""
    global _pipeline, _current_steps

    transcription_cfg = "whisper_small" if model_size == "small" else "whisper"

    if _pipeline is not None and _current_steps == steps:
        return _pipeline

    with initialize_config_module(config_module="speech_eureka.configs", version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[f"transcription={transcription_cfg}"],
        )

    _pipeline = SpeechPipeline(cfg=cfg, steps=steps)
    _pipeline.setup()
    _current_steps = steps[:]
    return _pipeline


def process_audio(
    audio_path: str,
    do_transcription: bool,
    do_diarization: bool,
    do_speaker_id: bool,
    model_size: str,
) -> tuple[str, str, str]:
    """Process uploaded audio and return results."""
    if audio_path is None:
        return "No audio provided.", "", ""

    steps = []
    if do_transcription:
        steps.append("transcription")
    if do_diarization:
        steps.append("diarization")
    if do_speaker_id:
        steps.append("speaker_id")

    if not steps:
        return "Select at least one pipeline step.", "", ""

    try:
        pipeline = get_pipeline(steps, model_size)
        result = pipeline.process(audio_path)
    except Exception as e:
        logger.exception("Pipeline error")
        return f"Error: {e}", "", ""

    # Format transcription
    transcription_text = ""
    if result.transcription:
        transcription_text = result.transcription.text
        if result.transcription.segments:
            transcription_text += "\n\n--- Segments ---\n"
            for seg in result.transcription.segments:
                transcription_text += f"[{seg.start:.1f}s - {seg.end:.1f}s] {seg.text}\n"

    # Format diarization
    diarization_text = ""
    if result.diarization:
        for seg in result.diarization:
            diarization_text += f"[{seg.start:.1f}s - {seg.end:.1f}s] {seg.speaker_label}\n"

    # Format aligned segments
    aligned_text = ""
    if result.segments:
        for seg in result.segments:
            name = seg.speaker_name or seg.speaker_label
            aligned_text += f"[{seg.start:.1f}s - {seg.end:.1f}s] {name}: {seg.text}\n"

    # Build JSON output
    json_output = json.dumps(
        {
            "audio_path": result.audio_path,
            "text": result.transcription.text if result.transcription else "",
            "diarization_segments": len(result.diarization),
            "aligned_segments": [
                {
                    "start": s.start,
                    "end": s.end,
                    "text": s.text,
                    "speaker_label": s.speaker_label,
                    "speaker_name": s.speaker_name,
                }
                for s in result.segments
            ],
        },
        indent=2,
        ensure_ascii=False,
    )

    display = transcription_text or diarization_text
    if aligned_text:
        display = aligned_text

    return display, diarization_text, json_output


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Speech Eureka") as app:
        gr.Markdown("# Speech Eureka\nClassroom speech transcription, diarization & speaker ID")

        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    label="Upload Audio",
                    type="filepath",
                )
                with gr.Group():
                    gr.Markdown("### Pipeline Steps")
                    do_transcription = gr.Checkbox(label="Transcription", value=True)
                    do_diarization = gr.Checkbox(label="Diarization", value=False)
                    do_speaker_id = gr.Checkbox(label="Speaker ID", value=False)
                model_size = gr.Radio(
                    choices=["small", "large-v3"],
                    value="small",
                    label="Whisper Model",
                )
                run_btn = gr.Button("Process", variant="primary")

            with gr.Column(scale=2):
                output_text = gr.Textbox(
                    label="Transcription / Aligned Output",
                    lines=12,
                    interactive=False,
                )
                diarization_output = gr.Textbox(
                    label="Diarization",
                    lines=6,
                    interactive=False,
                )
                json_output = gr.Code(
                    label="JSON",
                    language="json",
                )

        run_btn.click(
            fn=process_audio,
            inputs=[audio_input, do_transcription, do_diarization, do_speaker_id, model_size],
            outputs=[output_text, diarization_output, json_output],
        )

    return app


def main():
    import time

    app = build_app()
    port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    root_path = os.environ.get("GRADIO_ROOT_PATH", "")
    _, local_url, share_url = app.launch(
        server_name="0.0.0.0",
        server_port=port,
        root_path=root_path,
        share=True,
        prevent_thread_lock=True,
    )
    if share_url:
        from pathlib import Path
        Path("/workspace/speech-eureka/share_url.txt").write_text(share_url)

    # Keep alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
