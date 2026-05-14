"""Tests for image_ops module."""
import base64, io
from pathlib import Path
from unittest.mock import patch
import pytest
from PIL import Image
from lora_pipeline.image_ops import (
    load_image, image_to_b64, resolution_meta, get_phash,
)
from helpers import make_rgb_image, save_image


class TestLoadImage:
    def test_loads_valid_jpeg(self, tmp_path):
        p = tmp_path / "img.jpg"
        save_image(make_rgb_image(100, 150), p)
        img = load_image(p)
        assert img.mode == "RGB"
        assert img.size == (100, 150)

    def test_loads_valid_png(self, tmp_path):
        p = tmp_path / "img.png"
        save_image(make_rgb_image(80, 80), p, fmt="PNG")
        assert load_image(p).mode == "RGB"

    def test_raises_on_empty_file(self, tmp_path):
        p = tmp_path / "empty.jpg"
        p.write_bytes(b"")
        with pytest.raises(ValueError, match="empty"):
            load_image(p)

    def test_raises_on_corrupt(self, tmp_path):
        p = tmp_path / "corrupt.jpg"
        p.write_bytes(b"\xff\xd8\xff" + b"\x00" * 100)
        with pytest.raises(ValueError):
            load_image(p)

    def test_raises_on_permission_error(self, tmp_path):
        p = tmp_path / "locked.jpg"
        save_image(make_rgb_image(), p)
        with patch("lora_pipeline.image_ops.Path.stat",
                   side_effect=PermissionError("denied")):
            with pytest.raises(PermissionError, match="Cannot read"):
                load_image(p)

    def test_exif_rotation_applied(self, tmp_path):
        p = tmp_path / "rotated.jpg"
        img = Image.new("RGB", (200, 100))
        exif = img.getexif()
        exif[274] = 6
        img.save(p, exif=exif.tobytes())
        loaded = load_image(p)
        assert loaded.size == (100, 200)

    def test_rgba_converted_to_rgb(self, tmp_path):
        p = tmp_path / "rgba.png"
        Image.new("RGBA", (50, 50), (255, 0, 0, 128)).save(p)
        assert load_image(p).mode == "RGB"


class TestImageToB64:
    def test_returns_valid_base64_jpeg(self):
        b64 = image_to_b64(make_rgb_image(200, 300))
        decoded = base64.b64decode(b64)
        assert Image.open(io.BytesIO(decoded)).format == "JPEG"

    def test_thumbnails_large_image(self):
        import lora_pipeline.config as cfg
        b64 = image_to_b64(make_rgb_image(2000, 3000))
        restored = Image.open(io.BytesIO(base64.b64decode(b64)))
        assert max(restored.size) <= max(cfg.THUMB_SIZE)

    def test_small_image_not_upscaled(self):
        b64 = image_to_b64(make_rgb_image(50, 50))
        assert Image.open(io.BytesIO(base64.b64decode(b64))).size == (50, 50)


class TestResolutionMeta:
    @pytest.mark.parametrize("w,h,tier,ratio", [
        (1920, 1080, "high",   "landscape"),
        (1080, 1920, "high",   "portrait"),
        (1280, 720,  "medium", "landscape"),
        (640,  480,  "low",    "landscape"),
        (512,  512,  "low",    "square"),
    ])
    def test_tier_and_ratio(self, w, h, tier, ratio):
        meta = resolution_meta(make_rgb_image(w, h))
        assert meta["resolution_tier"] == tier
        assert meta["aspect_ratio"] == ratio

    def test_megapixels(self):
        assert resolution_meta(make_rgb_image(1000, 1000))["megapixels"] == 1.0

    def test_boundary_high(self):
        assert resolution_meta(make_rgb_image(1920, 100))["resolution_tier"] == "high"

    def test_boundary_medium(self):
        assert resolution_meta(make_rgb_image(1280, 100))["resolution_tier"] == "medium"

    def test_boundary_low(self):
        assert resolution_meta(make_rgb_image(1279, 100))["resolution_tier"] == "low"


class TestGetPhash:
    def test_returns_string(self):
        assert isinstance(get_phash(make_rgb_image()), str)

    def test_same_image_same_hash(self):
        img = make_rgb_image(100, 100, color=(200, 100, 50))
        assert get_phash(img) == get_phash(img)

    def test_different_images_different_hash(self):
        from PIL import Image, ImageDraw
        # Checkerboard vs gradient — structurally distinct, phash will differ
        img1 = Image.new("RGB", (64, 64))
        draw = ImageDraw.Draw(img1)
        for x in range(0, 64, 8):
            for y in range(0, 64, 8):
                fill = (0, 0, 0) if (x//8 + y//8) % 2 == 0 else (255, 255, 255)
                draw.rectangle([x, y, x+7, y+7], fill=fill)
        img2 = Image.new("RGB", (64, 64))
        px = img2.load()
        for x in range(64):
            for y in range(64):
                px[x, y] = (x * 4, y * 4, 128)
        assert get_phash(img1) != get_phash(img2)
