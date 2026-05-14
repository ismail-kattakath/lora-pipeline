"""Image loading, encoding, and metadata extraction."""
import base64, io
from pathlib import Path
from PIL import Image, ImageOps
from . import config


def load_image(path: Path) -> Image.Image:
    try:
        stat = path.stat()
    except PermissionError as e:
        raise PermissionError(f"Cannot read {path.name}: {e}")
    if stat.st_size == 0:
        raise ValueError(f"{path.name} is empty (0 bytes)")
    try:
        with Image.open(path) as probe:
            probe.verify()
    except Exception as e:
        raise ValueError(f"{path.name} failed integrity check: {e}")
    try:
        img = Image.open(path)
        img.load()
        return ImageOps.exif_transpose(img).convert("RGB")
    except OSError as e:
        raise ValueError(f"{path.name} could not be decoded: {e}")


def image_to_b64(img: Image.Image) -> str:
    thumb = img.copy()
    thumb.thumbnail(config.THUMB_SIZE, Image.LANCZOS)
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode()


def resolution_meta(img: Image.Image) -> dict:
    w, h   = img.size
    mp     = round((w * h) / 1_000_000, 2)
    long_e = max(w, h)
    tier   = "high" if long_e >= 1920 else ("medium" if long_e >= 1280 else "low")
    ratio  = "portrait" if h > w else ("landscape" if w > h else "square")
    return dict(width=w, height=h, megapixels=mp,
                resolution_tier=tier, aspect_ratio=ratio)


def get_phash(img: Image.Image) -> str:
    import imagehash
    return str(imagehash.phash(img))
