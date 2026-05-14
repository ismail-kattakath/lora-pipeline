"""Perceptual hash based near-duplicate detection."""

import imagehash

_seen_hashes: dict[str, str] = {}  # phash → stem

HAMMING_THRESHOLD = 4


def check_duplicate(ph: str, stem: str) -> str | None:
    """Return stem of near-duplicate if found, else None. Registers ph."""
    h = imagehash.hex_to_hash(ph)
    for seen_hash, seen_stem in _seen_hashes.items():
        if abs(h - imagehash.hex_to_hash(seen_hash)) <= HAMMING_THRESHOLD:
            return seen_stem
    _seen_hashes[ph] = stem
    return None


def reset():
    _seen_hashes.clear()
