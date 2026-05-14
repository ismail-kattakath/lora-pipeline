"""Shared test helpers — importable from any test file."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from PIL import Image

def make_rgb_image(w=100, h=150, color=(120, 80, 60)) -> Image.Image:
    return Image.new("RGB", (w, h), color=color)

def save_image(img: Image.Image, path: Path, fmt="JPEG") -> Path:
    img.save(path, format=fmt)
    return path
