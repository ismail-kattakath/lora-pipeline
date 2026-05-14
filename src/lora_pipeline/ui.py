"""Gradio UI — pipeline control, per-image retry, Ollama & Modal status."""

import json
import logging
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

import gradio as gr

from . import config
from .bootstrap import ollama_running, restart_ollama
from .file_ops import append_failed, load_checkpoint, load_failed, save_checkpoint
from .processor import process_single

log = logging.getLogger("ui")

# ── Modal helpers ──────────────────────────────────────────────────────────


def _modal_status() -> dict:
    try:
        import modal  # noqa: F401
    except ImportError:
        return {"installed": False, "authenticated": False, "profile": None}
    try:
        r = subprocess.run(
            ["modal", "profile", "current"], capture_output=True, text=True, timeout=5
        )
        if r.returncode == 0:
            return {"installed": True, "authenticated": True, "profile": r.stdout.strip()}
        return {"installed": True, "authenticated": False, "profile": None}
    except Exception:
        return {"installed": True, "authenticated": False, "profile": None}


def _restart_modal() -> tuple[bool, str]:
    try:
        r = subprocess.run(
            ["modal", "app", "restart", "lora-pipeline"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return r.returncode == 0, (r.stdout or r.stderr).strip() or "Done"
    except FileNotFoundError:
        return False, "modal CLI not found — pip install modal"
    except Exception as e:
        return False, str(e)


# ── Pipeline worker ────────────────────────────────────────────────────────


class PipelineWorker:
    def __init__(self):
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._pause = threading.Event()
        self._pause.set()  # not paused
        self._stop = threading.Event()

        cp = load_checkpoint()
        self.done: set[str] = set(cp.get("completed", []))
        self.failed: set[str] = load_failed()
        self.counts: dict = cp.get("counts", {f: 0 for f in config.FOLDERS})
        self.errors: int = cp.get("errors", 0)
        self.current: Optional[str] = None
        self.model: str = config.MODEL
        self.dry_run: bool = False
        self.start_time: Optional[float] = None
        self._meta_cache: dict[str, dict] = {}
        self._last_errors: dict[str, str] = {}

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def is_paused(self) -> bool:
        return not self._pause.is_set()

    def start(self, model: str, dry_run: bool = False):
        if self.is_running:
            return
        self.model = model
        self.dry_run = dry_run
        self.start_time = self.start_time or time.time()
        self._stop.clear()
        self._pause.set()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def pause(self):
        self._pause.clear()

    def resume(self):
        self._pause.set()

    def stop(self):
        self._stop.set()
        self._pause.set()  # unblock if paused

    def retry_stem(self, stem: str) -> str:
        with self._lock:
            self.done.discard(stem)
            self.failed.discard(stem)
            self._last_errors.pop(stem, None)
            self._meta_cache.pop(stem, None)

        # Remove from checkpoint
        cp = load_checkpoint()
        completed = set(cp.get("completed", []))
        completed.discard(stem)
        cp["completed"] = list(completed)
        save_checkpoint(cp)

        # Remove from failed.txt
        if config.FAILED_FILE.exists():
            lines = [
                ln
                for ln in config.FAILED_FILE.read_text().splitlines()
                if ln.strip() and ln.strip() != stem
            ]
            if lines:
                config.FAILED_FILE.write_text("\n".join(lines) + "\n")
            else:
                config.FAILED_FILE.unlink()

        if not self.is_running:
            src = self._find_source(stem)
            if src is None:
                return f"Source file not found: {stem}"
            self._stop.clear()
            self._pause.set()
            self._thread = threading.Thread(target=self._run_one, args=(src,), daemon=True)
            self._thread.start()
            return f"Retrying: {stem}"
        # Running loop will pick it up since it re-scans pending each iteration
        return f"Queued for retry on next pass: {stem}"

    # ── internal ──

    def _find_source(self, stem: str) -> Optional[Path]:
        for ext in (".jpg", ".jpeg", ".png"):
            p = config.SOURCE_DIR / f"{stem}{ext}"
            if p.exists():
                return p
        return None

    def _process_and_update(self, src: Path):
        try:
            record = process_single(src, self.model, self.dry_run)
            with self._lock:
                self.done.add(src.stem)
                self._meta_cache[src.stem] = record
                folder = record.get("folder", "rejected")
                self.counts[folder] = self.counts.get(folder, 0) + 1
            cp = load_checkpoint()
            cp.update(
                {
                    "completed": list(self.done),
                    "counts": self.counts,
                    "errors": self.errors,
                    "start_time": self.start_time,
                }
            )
            save_checkpoint(cp)
        except Exception as e:
            with self._lock:
                self.failed.add(src.stem)
                self.errors += 1
                self._last_errors[src.stem] = str(e)
            append_failed(src.stem)
            log.error(f"[{src.stem}] {e}")

    def _run_one(self, src: Path):
        with self._lock:
            self.current = src.stem
        self._process_and_update(src)
        with self._lock:
            self.current = None

    def _run_loop(self):
        images = (
            sorted(
                p
                for p in config.SOURCE_DIR.iterdir()
                if p.suffix.lower() in (".jpg", ".jpeg", ".png")
            )
            if config.SOURCE_DIR.exists()
            else []
        )

        for src in images:
            if self._stop.is_set():
                break
            self._pause.wait()
            if self._stop.is_set():
                break

            with self._lock:
                if src.stem in self.done or src.stem in self.failed:
                    continue
                self.current = src.stem

            self._process_and_update(src)

            with self._lock:
                self.current = None

        with self._lock:
            self.current = None


_worker = PipelineWorker()


# ── HTML status cards ──────────────────────────────────────────────────────

_CARD = (
    "padding:12px 16px;border:1px solid #e5e7eb;border-radius:8px;"
    "font-family:sans-serif;line-height:1.5"
)
_SUB = "font-size:11px;color:#6b7280;margin-top:2px"


def _ollama_html() -> str:
    up = ollama_running()
    dot, label = (
        ("<span style='color:#22c55e'>●</span>", "Running")
        if up
        else ("<span style='color:#ef4444'>●</span>", "Offline")
    )
    sub = f"<div style='{_SUB}'>{_worker.model}</div>" if up else ""
    return f"<div style='{_CARD}'>{dot} <b>Ollama</b> — {label}{sub}</div>"


def _modal_html() -> str:
    s = _modal_status()
    if not s["installed"]:
        dot, label = "<span style='color:#9ca3af'>●</span>", "not installed"
        sub = f"<div style='{_SUB}'>pip install modal</div>"
    elif not s["authenticated"]:
        dot, label = "<span style='color:#f59e0b'>●</span>", "not authenticated"
        sub = f"<div style='{_SUB}'>modal token new</div>"
    else:
        dot, label = "<span style='color:#22c55e'>●</span>", "connected"
        sub = f"<div style='{_SUB}'>{s.get('profile') or ''}</div>"
    return f"<div style='{_CARD}'>{dot} <b>Modal</b> — {label}{sub}</div>"


# ── Table & stats ──────────────────────────────────────────────────────────


def _source_images() -> list[Path]:
    if not config.SOURCE_DIR.exists():
        return []
    return sorted(
        p for p in config.SOURCE_DIR.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )


def _image_rows(status_filter: str = "All") -> list[list]:
    with _worker._lock:
        done = set(_worker.done)
        failed = set(_worker.failed)
        current = _worker.current
        meta_cache = dict(_worker._meta_cache)
        last_errors = dict(_worker._last_errors)

    rows = []
    for src in _source_images():
        stem = src.stem
        if stem == current:
            status = "🔄 Running"
            if status_filter not in ("All", "Running"):
                continue
            rows.append([src.name, status, "", "", "", "", ""])
        elif stem in done:
            status = "✅ Done"
            if status_filter not in ("All", "Done"):
                continue
            m = meta_cache.get(stem)
            if m is None:
                mp = config.OUTPUT_DIR / "_metadata" / f"{stem}.json"
                try:
                    m = json.loads(mp.read_text()) if mp.exists() else {}
                    with _worker._lock:
                        _worker._meta_cache[stem] = m
                except Exception:
                    m = {}
            rows.append(
                [
                    src.name,
                    status,
                    m.get("folder", "?"),
                    str(m.get("quality_score", "")),
                    "✓" if m.get("nsfw") else "",
                    m.get("processed_at", ""),
                    "",
                ]
            )
        elif stem in failed:
            status = "❌ Failed"
            if status_filter not in ("All", "Failed"):
                continue
            rows.append([src.name, status, "", "", "", "", last_errors.get(stem, "")])
        else:
            status = "⏳ Pending"
            if status_filter not in ("All", "Pending"):
                continue
            rows.append([src.name, status, "", "", "", "", ""])
    return rows


def _stats() -> tuple[int, int, int, int, int]:
    total = len(_source_images())
    with _worker._lock:
        done = len(_worker.done)
        failed = len(_worker.failed)
        running = 1 if _worker.current else 0
    pending = max(0, total - done - failed - running)
    return total, pending, running, done, failed


# ── Gradio app ─────────────────────────────────────────────────────────────


def main():
    logging.basicConfig(level=logging.INFO)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (config.OUTPUT_DIR / "_metadata").mkdir(exist_ok=True)

    with gr.Blocks(title="LoRA Pipeline", theme=gr.themes.Soft()) as demo:
        gr.Markdown("## LoRA Dataset Pipeline")

        # ── Infrastructure status ──────────────────────────────────────
        with gr.Row():
            with gr.Column(scale=3):
                ollama_box = gr.HTML(_ollama_html)
            with gr.Column(scale=1, min_width=160):
                restart_ollama_btn = gr.Button("↻ Restart Ollama", size="sm")
            with gr.Column(scale=3):
                modal_box = gr.HTML(_modal_html)
            with gr.Column(scale=1, min_width=160):
                restart_modal_btn = gr.Button("↻ Restart Modal App", size="sm")

        msg_box = gr.Textbox(label="", interactive=False, max_lines=1, show_label=False)

        # ── Pipeline controls ──────────────────────────────────────────
        with gr.Row():
            model_input = gr.Textbox(value=config.MODEL, label="Model", scale=4)
            dry_run_cb = gr.Checkbox(label="Dry run", value=False, scale=1)
            start_btn = gr.Button("▶ Start", variant="primary", scale=1)
            pause_btn = gr.Button("⏸ Pause / Resume", scale=1)
            stop_btn = gr.Button("⏹ Stop", variant="stop", scale=1)

        # ── Stats ──────────────────────────────────────────────────────
        with gr.Row():
            stat_total = gr.Number(label="Total", interactive=False, precision=0)
            stat_pending = gr.Number(label="Pending", interactive=False, precision=0)
            stat_running = gr.Number(label="Running", interactive=False, precision=0)
            stat_done = gr.Number(label="Done", interactive=False, precision=0)
            stat_failed = gr.Number(label="Failed", interactive=False, precision=0)

        # ── Image table ────────────────────────────────────────────────
        with gr.Row():
            filter_dd = gr.Dropdown(
                choices=["All", "Pending", "Running", "Done", "Failed"],
                value="All",
                label="Filter",
                scale=1,
            )
            selected_display = gr.Textbox(
                label="Selected", interactive=False, scale=4, placeholder="click a row"
            )
            retry_btn = gr.Button("↺ Retry selected", size="sm", scale=1)

        table = gr.Dataframe(
            headers=["File", "Status", "Folder", "Score", "NSFW", "Processed at", "Error"],
            datatype=["str"] * 7,
            value=_image_rows,
            interactive=False,
            wrap=True,
        )

        selected_stem = gr.State(value="")
        timer = gr.Timer(value=3)

        # ── Event handlers ─────────────────────────────────────────────

        def do_start(model, dry_run):
            if not ollama_running():
                return "Ollama is not running — click Restart Ollama first"
            if _worker.is_running:
                return "Pipeline already running"
            _worker.start(model, dry_run)
            return "Pipeline started"

        def do_pause_resume():
            if not _worker.is_running:
                return "Pipeline is not running"
            if _worker.is_paused:
                _worker.resume()
                return "Resumed"
            _worker.pause()
            return "Paused — will stop after current image finishes"

        def do_stop():
            _worker.stop()
            return "Stop signal sent — finishing current image…"

        def do_restart_ollama():
            ok = restart_ollama(log)
            return _ollama_html(), "Ollama restarted ✓" if ok else "Ollama restart failed ✗"

        def do_restart_modal():
            ok, msg = _restart_modal()
            icon = "✓" if ok else "✗"
            return _modal_html(), f"Modal: {msg} {icon}"

        def on_row_select(evt: gr.SelectData, current_filter):
            rows = _image_rows(current_filter)
            if rows and 0 <= evt.index[0] < len(rows):
                name = rows[evt.index[0]][0]
                stem = Path(name).stem
                return stem, name
            return "", ""

        def do_retry(stem):
            if not stem:
                return "No image selected — click a row first"
            return _worker.retry_stem(stem)

        def refresh(f):
            total, pending, running, done, failed = _stats()
            return (
                _ollama_html(),
                _modal_html(),
                total,
                pending,
                running,
                done,
                failed,
                _image_rows(f),
            )

        # wire up
        start_btn.click(do_start, [model_input, dry_run_cb], msg_box)
        pause_btn.click(do_pause_resume, [], msg_box)
        stop_btn.click(do_stop, [], msg_box)
        restart_ollama_btn.click(do_restart_ollama, [], [ollama_box, msg_box])
        restart_modal_btn.click(do_restart_modal, [], [modal_box, msg_box])
        table.select(on_row_select, [filter_dd], [selected_stem, selected_display])
        retry_btn.click(do_retry, [selected_stem], msg_box)
        filter_dd.change(lambda f: _image_rows(f), [filter_dd], table)
        timer.tick(
            refresh,
            [filter_dd],
            [
                ollama_box,
                modal_box,
                stat_total,
                stat_pending,
                stat_running,
                stat_done,
                stat_failed,
                table,
            ],
        )

    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
