"""Core processing loop."""

import json
import logging
import os
import shutil
import sys
import time

from . import config
from .duplicate import check_duplicate
from .file_ops import append_failed, atomic_write, load_checkpoint, load_failed, save_checkpoint
from .image_ops import get_phash, image_to_b64, load_image, resolution_meta
from .ollama_client import analyse_with_backoff

log = logging.getLogger("processor")


def run(args, model: str, shutdown_event):
    for f in config.FOLDERS:
        (config.OUTPUT_DIR / f).mkdir(parents=True, exist_ok=True)

    # Disk space check
    try:
        free_gb = shutil.disk_usage(config.OUTPUT_DIR).free / 1024**3
        if free_gb < 5:
            log.warning(f"Low disk space: {free_gb:.1f}GB free")
        if free_gb < 1:
            sys.exit(f"Aborting: only {free_gb:.1f}GB free")
    except Exception as e:
        log.warning(f"Could not check disk space: {e}")

    cp = {} if args.reset else load_checkpoint()
    done = set(cp.get("completed", []))
    counts = cp.get("counts", {f: 0 for f in config.FOLDERS})
    errors = cp.get("errors", 0)
    t_start = cp.get("start_time", time.time())

    all_images = sorted(
        p for p in config.SOURCE_DIR.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )

    if args.retry_failed:
        failed = load_failed()
        images = [p for p in all_images if p.stem in failed]
        log.info(f"Retry mode: {len(images)} failed images")
        if config.FAILED_FILE.exists():
            config.FAILED_FILE.unlink()
    else:
        images = all_images

    if args.limit:
        images = images[: args.limit]

    total = len(images)
    log.info("\nLoRA Dataset Processor")
    log.info(f"Source  : {config.SOURCE_DIR}  ({total} images)")
    log.info(f"Output  : {config.OUTPUT_DIR}")
    log.info(f"Model   : {model}")
    log.info(f"Mode    : {'DRY RUN' if args.dry_run else 'LIVE'}")
    log.info(f"Resume  : {len(done)} already done\n")

    processed_this_run = 0

    for i, src in enumerate(images, 1):
        if shutdown_event.is_set():
            log.info("Shutdown requested — stopping cleanly.")
            break

        stem = src.stem

        if not args.reset and stem in done:
            meta_path = config.OUTPUT_DIR / "_metadata" / f"{stem}.json"
            if meta_path.exists() and meta_path.stat().st_size > 0:
                log.debug(f"[{i:4}/{total}] SKIP  {src.name}")
                continue
            else:
                log.warning(f"[{i:4}/{total}] REPROCESS (truncated metadata): {src.name}")
                done.discard(stem)

        try:
            img = load_image(src)
        except Exception as e:
            log.error(f"[{i:4}/{total}] LOAD ERROR {src.name}: {e}")
            errors += 1
            append_failed(stem)
            continue

        res = resolution_meta(img)
        ph = get_phash(img)
        b64 = image_to_b64(img)

        dup = check_duplicate(ph, stem)
        if dup:
            log.info(f"[{i:4}/{total}] DUP   {src.name:<45} ≈ {dup}")

        t0 = time.time()
        analysis = analyse_with_backoff(b64, stem, model)
        elapsed = round(time.time() - t0, 1)

        if "_error" in analysis or "_parse_error" in analysis:
            log.error(f"[{i:4}/{total}] FAIL  {src.name}: {analysis}")
            errors += 1
            append_failed(stem)
            cp["errors"] = errors
            save_checkpoint(cp)
            continue

        folder = analysis.get("folder", "rejected")
        score = analysis.get("quality_score", "?")
        caption = analysis.get("caption", "")
        txt = f"{config.TRIGGER}, {caption}" if caption else config.TRIGGER
        nsfw = "NSFW" if analysis.get("nsfw") else "    "
        dup_tag = " DUP" if dup else "    "
        eta_s = elapsed * (total - i)
        eta_h, eta_m = int(eta_s // 3600), int((eta_s % 3600) // 60)

        log.info(
            f"[{i:4}/{total}] {nsfw}{dup_tag} {folder:<14} q={score}  "
            f"{src.name:<40} ({elapsed}s) ETA {eta_h}h{eta_m:02d}m"
        )

        record = {
            "source": str(src),
            "stem": stem,
            "folder": folder,
            "caption": caption,
            "trigger_caption": txt,
            "phash": ph,
            "duplicate_of": dup,
            "processed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            **res,
            **{k: v for k, v in analysis.items() if k not in ("folder", "caption")},
        }

        if not args.dry_run:
            try:
                meta_path = config.OUTPUT_DIR / "_metadata" / f"{stem}.json"
                atomic_write(meta_path, json.dumps(record, indent=2, ensure_ascii=False))

                dest = config.OUTPUT_DIR / folder / f"{stem}{src.suffix.lower()}"
                dest_tmp = dest.with_suffix(".tmp")
                try:
                    shutil.copy2(src, dest_tmp)
                    os.replace(dest_tmp, dest)
                except OSError as e:
                    try:
                        dest_tmp.unlink()
                    except Exception:
                        pass
                    if "No space left" in str(e):
                        sys.exit(f"Disk full while copying {src.name}")
                    raise

                atomic_write(config.OUTPUT_DIR / folder / f"{stem}.txt", txt)

            except Exception as e:
                log.error(f"  [{stem}] File write error: {e}")
                errors += 1
                append_failed(stem)
                continue

        counts[folder] = counts.get(folder, 0) + 1
        done.add(stem)
        processed_this_run += 1
        cp.update(
            {"completed": list(done), "counts": counts, "errors": errors, "start_time": t_start}
        )
        save_checkpoint(cp)

    # Auto-retry failed
    if not args.dry_run and not args.retry_failed and config.FAILED_FILE.exists():
        failed_now = set(config.FAILED_FILE.read_text().splitlines())
        if failed_now:
            log.info(f"\nAuto-retrying {len(failed_now)} failed images...\n")
            import subprocess

            subprocess.call([sys.executable, "-m", "lora_pipeline", "--retry-failed"])

    # Summary
    elapsed_total = round(time.time() - t_start)
    h, m = divmod(elapsed_total // 60, 60)
    log.info(f"\n{'─' * 55}")
    log.info(f"  Processed this run : {processed_this_run}")
    log.info(f"  Total elapsed      : {h}h {m:02d}m")
    for folder in config.FOLDERS:
        if folder == "_metadata":
            continue
        log.info(f"  {folder:<20} {counts.get(folder, 0):5d}")
    log.info(f"  {'errors':<20} {errors:5d}")
    log.info(f"  {'total done':<20} {len(done):5d}")
    log.info(f"{'─' * 55}")
