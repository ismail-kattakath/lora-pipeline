"""Central configuration — paths, constants, and the Qwen prompt."""

import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
_root = os.environ.get("IMAGE_ROOT")
if not _root:
    raise EnvironmentError(
        "IMAGE_ROOT environment variable must be set (e.g. IMAGE_ROOT=/Users/you/Pictures)"
    )

SOURCE_DIR = Path(_root) / "izzykatt"
OUTPUT_DIR = Path(_root) / "izzykatt-dataset"

CHECKPOINT_FILE = OUTPUT_DIR / "_metadata" / "checkpoint.json"
FAILED_FILE = OUTPUT_DIR / "_metadata" / "failed.txt"
LOG_FILE = OUTPUT_DIR / "_metadata" / "process.log"

FOLDERS = [
    "solo",
    "solo-nsfw",
    "group",
    "group-nsfw",
    "no-subject",
    "rejected",
    "_metadata",
]

# ── Model ──────────────────────────────────────────────────────────────────
MODEL = "qwen3.5:9b-q8_0"
MODEL_FALLBACK = "qwen3.5:9b"
OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# ── Processing ─────────────────────────────────────────────────────────────
TRIGGER = "[trigger]"
THUMB_SIZE = (1024, 1024)
CALL_TIMEOUT = 120  # seconds per Qwen call
MAX_RETRIES = 4
BACKOFF_BASE = 2
SERVER_RESTART_WAIT = 8
NUM_PREDICT = 8192  # thinking + JSON output; qwen3.5 thinks before responding

# ── Prompt ─────────────────────────────────────────────────────────────────
PROMPT = """Analyse this image and return a single JSON object with exactly these keys.
No markdown. No explanation. JSON only.

{
  "folder": one of: solo | solo-nsfw | group | group-nsfw | no-subject | rejected,
  "caption": "Detailed description. If person present: face first (eye color, shape, nose, lips, skin tone, expression), then hair (color, length, texture, style), then clothing and accessories, then pose and body language, then background and setting. Single paragraph.",
  "tags": ["descriptive", "tags"],
  "quality_score": integer 1-5,
  "has_face": true or false,
  "face_visibility": one of: clear | partial | occluded | none,
  "shot_type": one of: close-up | half-body | full-body | behind | no-person | object,
  "subject_angle": one of: frontal | three-quarter | profile | overhead | behind | none,
  "nsfw": true or false,
  "lighting": one of: natural | artificial | mixed | backlit | overexposed | underexposed,
  "occlusion": ["sunglasses", "hat", "hand", "hair", etc or empty list],
  "background": one of: clean | simple | busy | cluttered,
  "compression": one of: clean | mild | heavy,
  "filters_detected": true or false,
  "rejection_reason": null or one of: blurry | dark | low-res | no-subject | corrupt | watermark
}

folder rules:
- rejected   : quality_score <= 2 OR rejection_reason not null
- no-subject : no person clearly visible
- solo       : one person, SFW
- solo-nsfw  : one person, intimate or explicit
- group      : multiple people, SFW
- group-nsfw : multiple people, intimate or explicit
"""
