# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Requires `IMAGE_ROOT` env var pointing to the parent of the `izzykatt/` source folder:

```bash
export IMAGE_ROOT=/path/to/your/pictures
# or copy .env.example → .env and set IMAGE_ROOT there
```

Ollama must be running locally on `http://localhost:11434` with `qwen3.5:9b-q8_0` (or the fallback `qwen3.5:9b`). `bootstrap.py` will auto-install Ollama via Homebrew and pull the model on first run if missing.

## Commands

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=lora_pipeline -q

# Run a single test file
pytest tests/test_processor.py

# Type-check
mypy src/

# Run the pipeline
lora-pipeline [--dry-run] [--limit N] [--reset] [--retry-failed]
```

## Architecture

The pipeline is a single-pass image processor: it reads images from `SOURCE_DIR`, sends each (as a base64 JPEG thumbnail) to a local Qwen multimodal model via Ollama, parses the JSON response, then writes the image + caption `.txt` + metadata `.json` into one of six output subfolders.

**Data flow:**

```
cli.py → bootstrap.py → processor.py (main loop)
                          ├── image_ops.py   (load, resize, encode, phash)
                          ├── duplicate.py   (in-memory phash dedup)
                          ├── ollama_client.py (call_qwen + backoff)
                          └── file_ops.py    (atomic writes, checkpoint, failed list)
```

**Key design decisions:**

- **Crash-safe writes:** All file writes go through `file_ops.atomic_write()` (write to `.tmp`, then `os.replace()`). The checkpoint (`_metadata/checkpoint.json`) is saved after every successfully processed image.
- **Resumability:** `processor.run()` loads the checkpoint on startup and skips already-completed stems. Pass `--reset` to ignore it.
- **Watchdog:** `ollama_client.call_qwen()` runs the Ollama call in a daemon thread and joins with `CALL_TIMEOUT` (90s). On timeout or connection failure, `analyse_with_backoff()` retries up to `MAX_RETRIES` times with exponential backoff and will attempt an Ollama server restart after the second failure.
- **Duplicate detection:** `duplicate.py` maintains a module-level dict of `phash → stem`. Duplicates are flagged in metadata (`duplicate_of`) but still processed and written; they are not suppressed.
- **Auto-retry:** After the main loop, if `failed.txt` is non-empty and not already in retry mode, `processor.py` re-invokes itself via `subprocess.call` with `--retry-failed`.

**Configuration is centralized** in `config.py`. `IMAGE_ROOT` env var is required at import time — tests set it via `os.environ.setdefault("IMAGE_ROOT", "/tmp")` in `conftest.py`.

**Test isolation:** Tests use `patch_dirs` fixture (`conftest.py`) to monkeypatch all path constants in `config` and `file_ops` to `tmp_path` subdirectories. The `duplicate` module's in-memory state is reset via `dup_module.reset()` between tests.
