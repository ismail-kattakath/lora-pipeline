"""Integration tests for processor.run() with heavy mocking."""

import json
import threading
from unittest.mock import MagicMock, patch

import pytest

import lora_pipeline.config as cfg
from lora_pipeline.processor import run

GOOD = {
    "folder": "solo",
    "caption": "A woman with brown eyes.",
    "quality_score": 4,
    "has_face": True,
    "face_visibility": "clear",
    "shot_type": "close-up",
    "subject_angle": "frontal",
    "nsfw": False,
    "lighting": "natural",
    "occlusion": [],
    "background": "simple",
    "compression": "clean",
    "filters_detected": False,
    "rejection_reason": None,
    "tags": ["woman", "portrait"],
}


def make_args(**kw):
    class Args:
        dry_run = False
        limit = 0
        reset = False
        retry_failed = False

    for k, v in kw.items():
        setattr(Args, k, v)
    return Args()


@pytest.fixture
def mocked_env(patch_dirs):
    with (
        patch("lora_pipeline.processor.analyse_with_backoff", return_value=GOOD),
        patch(
            "lora_pipeline.processor.shutil.disk_usage", return_value=MagicMock(free=10 * 1024**3)
        ),
    ):
        yield patch_dirs


class TestRunHappyPath:
    def test_processes_all_images(self, mocked_env):
        run(make_args(), cfg.MODEL, threading.Event())
        assert len(list((cfg.OUTPUT_DIR / "solo").glob("*.jpg"))) == 5

    def test_writes_txt_captions(self, mocked_env):
        run(make_args(), cfg.MODEL, threading.Event())
        txts = list((cfg.OUTPUT_DIR / "solo").glob("*.txt"))
        assert len(txts) == 5
        assert all("[trigger]" in t.read_text() for t in txts)

    def test_writes_metadata_json(self, mocked_env):
        run(make_args(), cfg.MODEL, threading.Event())
        metas = list((cfg.OUTPUT_DIR / "_metadata").glob("photo-*.json"))
        assert len(metas) == 5
        record = json.loads(metas[0].read_text())
        assert {"folder", "phash", "width", "caption"} <= record.keys()

    def test_dry_run_writes_nothing(self, mocked_env):
        run(make_args(dry_run=True), cfg.MODEL, threading.Event())
        assert not list((cfg.OUTPUT_DIR / "solo").glob("*.jpg"))
        assert not list((cfg.OUTPUT_DIR / "_metadata").glob("photo-*.json"))

    def test_limit_flag(self, mocked_env):
        run(make_args(limit=2), cfg.MODEL, threading.Event())
        assert len(list((cfg.OUTPUT_DIR / "solo").glob("*.jpg"))) == 2


class TestRunResume:
    def test_skips_already_done(self, mocked_env):
        run(make_args(), cfg.MODEL, threading.Event())
        with patch("lora_pipeline.processor.analyse_with_backoff") as mock:
            run(make_args(), cfg.MODEL, threading.Event())
        mock.assert_not_called()

    def test_reset_reprocesses_all(self, mocked_env):
        run(make_args(), cfg.MODEL, threading.Event())
        with patch("lora_pipeline.processor.analyse_with_backoff", return_value=GOOD) as mock:
            run(make_args(reset=True), cfg.MODEL, threading.Event())
        assert mock.call_count == 5

    def test_reprocesses_truncated_metadata(self, mocked_env):
        run(make_args(), cfg.MODEL, threading.Event())
        meta = list((cfg.OUTPUT_DIR / "_metadata").glob("photo-*.json"))[0]
        meta.unlink()
        with patch("lora_pipeline.processor.analyse_with_backoff", return_value=GOOD) as mock:
            run(make_args(), cfg.MODEL, threading.Event())
        assert mock.call_count == 1


class TestRunErrorHandling:
    def test_skips_corrupt_image(self, mocked_env):
        list(cfg.SOURCE_DIR.glob("*.jpg"))[0].write_bytes(b"")
        run(make_args(), cfg.MODEL, threading.Event())
        assert len(list((cfg.OUTPUT_DIR / "solo").glob("*.jpg"))) == 4

    def test_failed_written_to_txt(self, mocked_env):
        list(cfg.SOURCE_DIR.glob("*.jpg"))[0].write_bytes(b"")
        run(make_args(), cfg.MODEL, threading.Event())
        from lora_pipeline.file_ops import load_failed

        assert len(load_failed()) == 1

    def test_analysis_failure_continues_job(self, mocked_env):
        n = {"count": 0}

        def side(*a, **kw):
            n["count"] += 1
            return {"_error": "fail"} if n["count"] == 2 else GOOD

        with patch("lora_pipeline.processor.analyse_with_backoff", side_effect=side):
            run(make_args(), cfg.MODEL, threading.Event())
        assert len(list((cfg.OUTPUT_DIR / "solo").glob("*.jpg"))) == 4

    def test_disk_full_aborts(self, patch_dirs):
        with patch("lora_pipeline.processor.shutil.disk_usage", return_value=MagicMock(free=0)):
            with pytest.raises(SystemExit):
                run(make_args(), cfg.MODEL, threading.Event())

    def test_shutdown_event_stops_loop(self, mocked_env):
        n = {"count": 0}
        shutdown = threading.Event()

        def side(*a, **kw):
            n["count"] += 1
            if n["count"] == 2:
                shutdown.set()
            return GOOD

        with patch("lora_pipeline.processor.analyse_with_backoff", side_effect=side):
            run(make_args(), cfg.MODEL, shutdown)
        assert n["count"] == 2


class TestRunFolderRouting:
    def test_nsfw_routed_correctly(self, patch_dirs):
        nsfw = {**GOOD, "folder": "solo-nsfw", "nsfw": True}
        with (
            patch("lora_pipeline.processor.analyse_with_backoff", return_value=nsfw),
            patch(
                "lora_pipeline.processor.shutil.disk_usage",
                return_value=MagicMock(free=10 * 1024**3),
            ),
        ):
            run(make_args(), cfg.MODEL, threading.Event())
        assert len(list((cfg.OUTPUT_DIR / "solo-nsfw").glob("*.jpg"))) == 5

    def test_rejected_routed_correctly(self, patch_dirs):
        rej = {**GOOD, "folder": "rejected", "quality_score": 1}
        with (
            patch("lora_pipeline.processor.analyse_with_backoff", return_value=rej),
            patch(
                "lora_pipeline.processor.shutil.disk_usage",
                return_value=MagicMock(free=10 * 1024**3),
            ),
        ):
            run(make_args(), cfg.MODEL, threading.Event())
        assert len(list((cfg.OUTPUT_DIR / "rejected").glob("*.jpg"))) == 5
