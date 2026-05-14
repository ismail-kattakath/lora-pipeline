"""Shared fixtures."""
import os
import pytest
from pathlib import Path
from helpers import make_rgb_image, save_image

# Provide a dummy root so config imports without error; patch_dirs overrides per test.
os.environ.setdefault("IMAGE_ROOT", "/tmp")

import lora_pipeline.config as cfg
import lora_pipeline.duplicate as dup_module


@pytest.fixture
def tmp_src(tmp_path):
    src = tmp_path / "source"
    src.mkdir()
    for i in range(5):
        save_image(make_rgb_image(800, 1000), src / f"photo-{i:04d}.jpg")
    return src


@pytest.fixture
def tmp_out(tmp_path):
    out = tmp_path / "dataset"
    for f in cfg.FOLDERS:
        (out / f).mkdir(parents=True)
    return out


@pytest.fixture
def patch_dirs(tmp_src, tmp_out, monkeypatch):
    monkeypatch.setattr(cfg, "SOURCE_DIR",      tmp_src)
    monkeypatch.setattr(cfg, "OUTPUT_DIR",       tmp_out)
    monkeypatch.setattr(cfg, "CHECKPOINT_FILE",  tmp_out / "_metadata" / "checkpoint.json")
    monkeypatch.setattr(cfg, "FAILED_FILE",      tmp_out / "_metadata" / "failed.txt")
    monkeypatch.setattr(cfg, "LOG_FILE",         tmp_out / "_metadata" / "process.log")
    # also patch file_ops module-level references
    import lora_pipeline.file_ops as fo
    monkeypatch.setattr(fo, "config", cfg)
    dup_module.reset()
    return cfg
