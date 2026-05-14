"""Entry point — argparse, logging setup, signal handler."""

import argparse
import logging
import signal
import threading
from logging.handlers import RotatingFileHandler

from . import config
from .bootstrap import ensure_ollama, ensure_packages
from .processor import run


def setup_logging() -> logging.Logger:
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (config.OUTPUT_DIR / "_metadata").mkdir(exist_ok=True)

    log = logging.getLogger("processor")
    log.setLevel(logging.DEBUG)

    if not log.handlers:
        fh = RotatingFileHandler(
            config.LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-8s %(message)s", "%Y-%m-%d %H:%M:%S")
        )

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter("%(message)s"))

        log.addHandler(fh)
        log.addHandler(ch)

    return log


def main():
    ap = argparse.ArgumentParser(
        description="lora-pipeline dataset processor — single-pass Qwen pipeline"
    )
    ap.add_argument("--dry-run", action="store_true", help="Classify only, do not write files")
    ap.add_argument("--limit", type=int, default=0, help="Process only first N images (0 = all)")
    ap.add_argument("--reset", action="store_true", help="Ignore checkpoint, reprocess everything")
    ap.add_argument(
        "--retry-failed", action="store_true", help="Only reprocess images listed in failed.txt"
    )
    args = ap.parse_args()

    log = setup_logging()

    shutdown = threading.Event()
    signal.signal(
        signal.SIGINT,
        lambda sig, frame: (log.info("\nCtrl+C — finishing current image..."), shutdown.set()),
    )

    ensure_packages()
    model = ensure_ollama(log)
    run(args, model, shutdown)
