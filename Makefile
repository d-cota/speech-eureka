UV := $(shell command -v uv 2>/dev/null || echo ~/.local/bin/uv)
PYTHON := .venv/bin/python
VENV := .venv

# Load .env if present
ifneq (,$(wildcard .env))
  include .env
  export
endif

.PHONY: install sync run eval gui help

## Install uv (if missing) and sync all dependencies
install:
	@if ! command -v uv >/dev/null 2>&1 && [ ! -f ~/.local/bin/uv ]; then \
		echo "Installing uv..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	fi
	$(UV) sync --all-extras

## Sync dependencies from lockfile (fast, no re-solve)
sync:
	$(UV) sync --all-extras

## Run the speech pipeline (override audio paths as needed)
## Usage: make run AUDIO=data/audio/sample.wav
run:
	$(UV) run speech-eureka '+audio.paths=[$(AUDIO)]'

## Run evaluation on a pipeline result JSON
## Usage: make eval RESULT=outputs/sample.json
## Usage: make eval RESULT=outputs/sample.json REF_RTTM=ref/sample.rttm REF_TXT=ref/sample.txt
eval:
	$(UV) run speech-eureka-eval result=$(RESULT) \
		$(if $(REF_RTTM),ref_rttm=$(REF_RTTM),) \
		$(if $(REF_TXT),ref_transcript=$(REF_TXT),) \
		$(if $(TEACHER),teacher_label=$(TEACHER),) \
		$(if $(DURATION),audio_duration=$(DURATION),)

## Launch the Gradio web UI
gui:
	$(UV) run speech-eureka-gui

help:
	@echo ""
	@echo "speech-eureka targets:"
	@echo "  make install              Install uv + sync all dependencies"
	@echo "  make sync                 Sync dependencies from lockfile"
	@echo "  make run AUDIO=<path>     Run pipeline on an audio file"
	@echo "  make eval RESULT=<path>   Evaluate a pipeline result JSON"
	@echo "    Optional: REF_RTTM=<path>  REF_TXT=<path>  TEACHER=<label>  DURATION=<s>"
	@echo "  make gui                  Launch Gradio web UI"
	@echo ""
