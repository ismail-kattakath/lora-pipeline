# lora-pipeline

Single-pass dataset processing pipeline for Flux LoRA training.

Processes a folder of photos through Qwen3.5 via Ollama — classifies, captions,
tags, and organises each image into subfolders ready for ai-toolkit training.

## Output structure

```
izzykatt-dataset/
  solo/           # single subject, SFW
  solo-nsfw/      # single subject, NSFW
  group/          # subject + others, SFW
  group-nsfw/     # subject + others, NSFW
  no-subject/     # no identifiable subject
  rejected/       # bad quality
  _metadata/      # per-image JSON + logs + checkpoint
```

Each image gets:
- `photo-0001.jpg` — copied from source
- `photo-0001.txt` — caption prefixed with `[trigger],`
- `_metadata/photo-0001.json` — full metadata record

## Install

```bash
cd lora-pipeline
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

```bash
# Test on 5 images without writing
lora-pipeline --dry-run --limit 5

# Full run
lora-pipeline

# Resume after crash (automatic via checkpoint)
lora-pipeline

# Reprocess everything
lora-pipeline --reset

# Reprocess only failed images
lora-pipeline --retry-failed
```

## Configuration

Edit `src/lora_pipeline/config.py`:

| Key | Default | Description |
|-----|---------|-------------|
| `SOURCE_DIR` | `~/Pictures/izzykatt` | Input folder |
| `OUTPUT_DIR` | `~/Pictures/izzykatt-dataset` | Output folder |
| `MODEL` | `qwen3.5:9b-q8_0` | Preferred Ollama model |
| `MODEL_FALLBACK` | `qwen3.5:9b` | Used if preferred not pulled |
| `THUMB_SIZE` | `(1024, 1024)` | Max size sent to Qwen |
| `CALL_TIMEOUT` | `90` | Seconds before watchdog kills a call |
| `MAX_RETRIES` | `4` | Attempts per image before giving up |

## Tests

```bash
pytest --cov=lora_pipeline -q
```

## Metadata schema

Each `_metadata/photo-XXXX.json` contains:

```json
{
  "folder": "solo",
  "caption": "...",
  "trigger_caption": "[trigger], ...",
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
  "tags": ["..."],
  "width": 1080,
  "height": 1920,
  "megapixels": 2.07,
  "resolution_tier": "high",
  "aspect_ratio": "portrait",
  "phash": "...",
  "duplicate_of": null,
  "processed_at": "2026-05-14 04:00:00"
}
```
