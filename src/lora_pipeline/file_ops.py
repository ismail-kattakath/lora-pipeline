"""Atomic file writes, checkpoint persistence, failed-image tracking."""

import json
import logging
import os
import time
from pathlib import Path

from . import config

log = logging.getLogger("processor")


def atomic_write(path: Path, content: str):
    """Write to .tmp then os.replace() — crash-safe on POSIX."""
    tmp = path.with_suffix(".tmp")
    try:
        tmp.write_text(content, encoding="utf-8")
        os.replace(tmp, path)
    except Exception:
        try:
            tmp.unlink()
        except Exception:
            pass
        raise


def load_checkpoint() -> dict:
    if config.CHECKPOINT_FILE.exists():
        try:
            return json.loads(config.CHECKPOINT_FILE.read_text())
        except Exception:
            pass
    return {"completed": [], "counts": {}, "errors": 0, "start_time": time.time()}


def save_checkpoint(cp: dict):
    try:
        atomic_write(config.CHECKPOINT_FILE, json.dumps(cp, indent=2))
    except Exception as e:
        log.error(f"Checkpoint save failed: {e}")


def load_failed() -> set:
    if config.FAILED_FILE.exists():
        return set(config.FAILED_FILE.read_text().splitlines())
    return set()


def append_failed(stem: str):
    with open(config.FAILED_FILE, "a") as f:
        f.write(stem + "\n")
