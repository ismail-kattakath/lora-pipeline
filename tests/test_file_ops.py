"""Tests for file_ops module."""

from pathlib import Path
from unittest.mock import patch

import pytest

import lora_pipeline.config as cfg
from lora_pipeline.file_ops import (
    append_failed,
    atomic_write,
    load_checkpoint,
    load_failed,
    save_checkpoint,
)


class TestAtomicWrite:
    def test_creates_file(self, tmp_path):
        p = tmp_path / "out.json"
        atomic_write(p, '{"a":1}')
        assert p.read_text() == '{"a":1}'

    def test_no_tmp_left_on_success(self, tmp_path):
        p = tmp_path / "out.json"
        atomic_write(p, "hello")
        assert not list(tmp_path.glob("*.tmp"))

    def test_overwrites_existing(self, tmp_path):
        p = tmp_path / "out.txt"
        p.write_text("old")
        atomic_write(p, "new")
        assert p.read_text() == "new"

    def test_raises_and_cleans_on_failure(self, tmp_path):
        p = tmp_path / "missing_parent" / "out.json"
        with pytest.raises(Exception):
            atomic_write(p, "data")
        assert not list(tmp_path.glob("**/*.tmp"))

    def test_unicode_content(self, tmp_path):
        p = tmp_path / "u.txt"
        atomic_write(p, "héllo 🎉")
        assert p.read_text(encoding="utf-8") == "héllo 🎉"


class TestCheckpoint:
    def test_defaults_when_missing(self, patch_dirs):
        cp = load_checkpoint()
        assert cp["errors"] == 0
        assert "completed" in cp

    def test_save_and_reload(self, patch_dirs):
        cp = {"completed": ["a", "b"], "errors": 2, "counts": {}, "start_time": 0.0}
        save_checkpoint(cp)
        loaded = load_checkpoint()
        assert loaded["completed"] == ["a", "b"]
        assert loaded["errors"] == 2

    def test_recovers_from_corrupt(self, patch_dirs):
        cfg.CHECKPOINT_FILE.write_text("NOT JSON {{")
        cp = load_checkpoint()
        assert cp["errors"] == 0

    def test_save_failure_logged(self, patch_dirs, caplog):
        import logging

        bad_path = Path("/nonexistent/cp.json")
        with patch("lora_pipeline.file_ops.config") as mock_cfg:
            mock_cfg.CHECKPOINT_FILE = bad_path
            with caplog.at_level(logging.ERROR, logger="processor"):
                save_checkpoint({"completed": []})
        assert any("Checkpoint save failed" in r.message for r in caplog.records)


class TestFailedList:
    def test_empty_when_missing(self, patch_dirs):
        assert load_failed() == set()

    def test_append_and_load(self, patch_dirs):
        append_failed("photo-0001")
        append_failed("photo-0002")
        assert {"photo-0001", "photo-0002"} == load_failed()

    def test_creates_file(self, patch_dirs):
        assert not cfg.FAILED_FILE.exists()
        append_failed("x")
        assert cfg.FAILED_FILE.exists()

    def test_set_deduplicates(self, patch_dirs):
        append_failed("photo-0001")
        append_failed("photo-0001")
        assert len(load_failed()) == 1
