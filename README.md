# lora-pipeline

Single-pass dataset processing pipeline for Flux LoRA training. Feeds images through [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) (vision + thinking model) via [Ollama](https://ollama.com), classifies each image, writes a caption and metadata, and organises everything into training-ready subfolders.

---

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) with `qwen3.5:9b` pulled (`ollama pull qwen3.5:9b`)
- `IMAGE_ROOT` env var pointing to the parent folder of your source images

---

## Quick start

```bash
# 1. Clone and set up
git clone <repo>
cd lora-pipeline
make setup

# 2. Configure
cp .env.example .env
# Edit .env and set IMAGE_ROOT=/path/to/your/pictures

# 3. Dry run (classifies 5 images, writes nothing)
make run-dry

# 4. Full run in background
make run-detached

# 5. Watch logs
make logs-all
```

---

## Source layout

The pipeline expects:

```
$IMAGE_ROOT/
  izzykatt/          ← source images (.jpg, .jpeg, .png)
```

And produces:

```
$IMAGE_ROOT/
  izzykatt-dataset/
    solo/            ← single subject, SFW
    solo-nsfw/       ← single subject, NSFW
    group/           ← multiple people, SFW
    group-nsfw/      ← multiple people, NSFW
    no-subject/      ← no identifiable subject
    rejected/        ← low quality or flagged
    _metadata/
      photo-XXXX.json    ← full analysis record per image
      checkpoint.json    ← progress checkpoint (resumable)
      failed.txt         ← stems that errored (for --retry-failed)
      process.log        ← rotating detailed log (includes thinking output)
```

Each processed image produces three files:

```
solo/photo-0001.jpg          ← copy of source
solo/photo-0001.txt          ← "[trigger], <caption>"
_metadata/photo-0001.json    ← full metadata record
```

---

## Make targets

### Local (native Python)

| Target | Description |
|--------|-------------|
| `make setup` | Create `.venv` and install dependencies |
| `make run` | Run pipeline in foreground, logs to `pipeline_run.log` |
| `make run-detached` | Run in background via `nohup`, PID saved to `.pipeline.pid` |
| `make run-dry` | Classify only — no files written |
| `make reset` | Reprocess everything, ignoring checkpoint |
| `make stop` | Kill the detached pipeline |
| `make logs` | `tail -f pipeline_run.log` |
| `make logs-ollama` | `tail -f ~/.ollama/logs/server.log` |
| `make logs-all` | Both logs interleaved |
| `make test` | Run test suite |
| `make coverage` | Run tests with coverage report |
| `make typecheck` | Run mypy |
| `make clean` | Remove venv, caches, PID file |

### Docker

| Target | Description |
|--------|-------------|
| `make docker-build` | Build the pipeline image |
| `make docker-run` | Run pipeline in Docker, Ollama on host (macOS/Metal) |
| `make docker-run-cloud` | Run pipeline + Ollama in Docker with NVIDIA GPU (RunPod/VastAI) |
| `make docker-logs` | Stream container logs |
| `make docker-stop` | Stop and remove containers |

---

## CLI flags

```
lora-pipeline [--dry-run] [--limit N] [--reset] [--retry-failed]
```

| Flag | Description |
|------|-------------|
| `--dry-run` | Classify and log only, write no files |
| `--limit N` | Process only the first N images |
| `--reset` | Ignore checkpoint and reprocess everything |
| `--retry-failed` | Only reprocess images listed in `failed.txt` |

---

## Configuration

All settings live in `src/lora_pipeline/config.py`. Key env vars:

| Env var | Default | Description |
|---------|---------|-------------|
| `IMAGE_ROOT` | *(required)* | Parent folder of source images |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |

Key constants in `config.py`:

| Constant | Default | Description |
|----------|---------|-------------|
| `MODEL` | `qwen3.5:9b-q8_0` | Preferred model (q8 quantisation) |
| `MODEL_FALLBACK` | `qwen3.5:9b` | Used if preferred not pulled |
| `TRIGGER` | `[trigger]` | Prefix prepended to every caption |
| `THUMB_SIZE` | `(1024, 1024)` | Max resolution sent to the model |
| `CALL_TIMEOUT` | `120` | Seconds before a hung call is killed |
| `MAX_RETRIES` | `4` | Attempts per image before marking as failed |
| `NUM_PREDICT` | `8192` | Token budget per call (thinking + JSON) |

---

## Docker deployment

The pipeline image contains only the Python code. Ollama runs separately, either on the host or in a sidecar container.

### macOS (Apple Silicon)

Ollama must run natively to use the Metal GPU — Docker on macOS cannot access it.

```bash
# Ollama running natively on host
ollama serve

# Pipeline in Docker, pointing at host Ollama
IMAGE_ROOT=/path/to/pictures make docker-run
```

### RunPod / VastAI (NVIDIA GPU)

Both services run Linux, so full containerisation with GPU pass-through works.

```bash
IMAGE_ROOT=/workspace/pictures make docker-run-cloud
```

On first run, Ollama pulls `qwen3.5:9b` (~6 GB) into the `ollama_models` named volume. Subsequent runs skip the download.

### Logs

```bash
make docker-logs          # stream pipeline output
docker compose logs -f    # all containers
```

---

## How it works

```
cli.py → bootstrap.py → processor.py
                          ├── image_ops.py     load, resize, encode, phash
                          ├── duplicate.py     in-memory phash dedup
                          ├── ollama_client.py streaming call + thinking log
                          └── file_ops.py      atomic writes, checkpoint, failed list
```

1. **Bootstrap** — ensures Ollama is running and the model is pulled.
2. **Main loop** — for each image: load → thumbnail → base64 encode → send to Qwen3.5.
3. **Streaming** — the model's `<think>...</think>` reasoning is logged line-by-line to `process.log` as it generates, then stripped before JSON parsing.
4. **Atomic writes** — every file write goes through a `.tmp` → `os.replace()` to prevent corruption on crash.
5. **Checkpoint** — saved after every image; re-run resumes automatically from where it left off.
6. **Auto-retry** — after the main loop, any images in `failed.txt` are automatically retried once.

---

## Metadata schema

`_metadata/photo-XXXX.json`:

```json
{
  "source": "/path/to/original.jpg",
  "stem": "photo-0001",
  "folder": "solo",
  "caption": "A woman with brown eyes...",
  "trigger_caption": "[trigger], A woman with brown eyes...",
  "tags": ["portrait", "natural lighting", "close-up"],
  "quality_score": 4,
  "has_face": true,
  "face_visibility": "clear",
  "shot_type": "close-up",
  "subject_angle": "frontal",
  "nsfw": false,
  "lighting": "natural",
  "occlusion": [],
  "background": "simple",
  "compression": "clean",
  "filters_detected": false,
  "rejection_reason": null,
  "width": 1080,
  "height": 1920,
  "megapixels": 2.07,
  "resolution_tier": "high",
  "aspect_ratio": "portrait",
  "phash": "f8c0e0c8f0e0c8f0",
  "duplicate_of": null,
  "processed_at": "2026-05-14 07:00:00"
}
```

---

## Development

```bash
make setup        # install with dev extras
make test         # pytest
make coverage     # pytest + coverage report
make typecheck    # mypy
```

Tests use `tmp_path` fixtures and monkeypatch all path constants — no real images or Ollama connection needed.
