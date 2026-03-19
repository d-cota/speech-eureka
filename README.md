# Speech Eureka

Modular pipeline for classroom speech transcription, diarization, and speaker identification. Designed for classrooms of ~20 students using open-source models from HuggingFace and related ecosystems.

## Models

| Stage | Model | Source |
|---|---|---|
| Transcription | Whisper large-v3 | `openai/whisper-large-v3` |
| Diarization | pyannote 3.1 | `pyannote/speaker-diarization-3.1` |
| Speaker ID | ECAPA-TDNN | `speechbrain/spkrec-ecapa-voxceleb` |

## Setup

Requires [uv](https://docs.astral.sh/uv/), Python 3.11+, and `ffmpeg`.

```bash
# Install ffmpeg (Ubuntu/Debian)
sudo apt-get install -y ffmpeg

# Install dependencies
uv sync

# Install with dev tools (pytest, ruff)
uv sync --extra dev
```

### Cache and Data Directories

Model weights and HF datasets are cached under `/workspace/cache/` to avoid re-downloading:

```bash
export HF_HOME=/workspace/cache/huggingface
export TORCH_HOME=/workspace/cache/torch
```

These are also set in `.env` (gitignored). The `data/` directory holds audio files and enrollment samples and is also gitignored.

## Usage

```bash
# Process an audio file with defaults (full pipeline)
uv run speech-eureka '+audio.paths=[/path/to/recording.wav]'

# Use a smaller whisper model
uv run speech-eureka '+audio.paths=[/path/to/recording.wav]' transcription=whisper_small

# Transcription only (skip diarization + speaker ID)
uv run speech-eureka '+audio.paths=[/path/to/recording.wav]' pipeline=transcribe_only

# Change output format (json, txt, rttm)
uv run speech-eureka '+audio.paths=[/path/to/recording.wav]' output.format=txt

# Multiple files
uv run speech-eureka '+audio.paths=[file1.wav,file2.wav]'
```

## Speaker Enrollment

To identify known speakers, place audio samples in the enrollment directory:

```
data/enrollments/
├── alice/
│   ├── sample1.wav
│   └── sample2.wav
├── bob/
│   └── sample1.wav
└── ...
```

The pipeline automatically enrolls all speakers found in this directory and matches them against diarized segments using cosine similarity on ECAPA-TDNN embeddings.

## Project Structure

```
data/                               # Local data (gitignored)
├── audio/                           # Audio files to process
└── enrollments/                     # Speaker enrollment samples

src/speech_eureka/
├── configs/                         # Hydra YAML configs
│   ├── config.yaml                  # Main config (composes defaults)
│   ├── transcription/               # Transcription model configs
│   ├── diarization/                 # Diarization model configs
│   ├── speaker_id/                  # Speaker ID model configs
│   └── pipeline/                    # Pipeline step configs (default, transcribe_only)
├── models/data.py                   # Dataclasses (segments, results, profiles)
├── modules/
│   ├── base.py                      # Abstract base classes
│   ├── transcription.py             # WhisperTranscriber
│   ├── diarization.py               # PyannoteDiarizer
│   └── speaker_id.py                # EcapaSpeakerIdentifier
├── utils/audio.py                   # Audio loading helpers
├── pipeline.py                      # Pipeline orchestrator
└── main.py                          # Hydra entry point
```

## Configuration

All modules are configured via Hydra YAML files using the `_target_` pattern for instantiation. Override any config value from the command line:

```bash
uv run speech-eureka '+audio.paths=[recording.wav]' \
    transcription=whisper_small \
    diarization.max_speakers=25 \
    speaker_id.similarity_threshold=0.7 \
    output.dir=my_results
```

## Adding a New Module

1. Create a class extending the relevant base (`BaseTranscriber`, `BaseDiarizer`, or `BaseSpeakerIdentifier`)
2. Add a YAML config in the corresponding `src/speech_eureka/configs/` subdirectory with a `_target_` pointing to your class
3. Use it: `uv run speech-eureka '+audio.paths=[file.wav]' transcription=your_config`
