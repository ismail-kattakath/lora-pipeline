"""Tests for duplicate module."""

import imagehash
from helpers import make_rgb_image

from lora_pipeline import duplicate as dup_module
from lora_pipeline.duplicate import HAMMING_THRESHOLD, check_duplicate, reset


class TestCheckDuplicate:
    def setup_method(self):
        reset()

    def test_first_image_not_duplicate(self):
        ph = str(imagehash.phash(make_rgb_image(100, 100, color=(10, 20, 30))))
        assert check_duplicate(ph, "photo-0001") is None

    def test_identical_hash_is_duplicate(self):
        ph = str(imagehash.phash(make_rgb_image(100, 100, color=(10, 20, 30))))
        check_duplicate(ph, "photo-0001")
        assert check_duplicate(ph, "photo-0002") == "photo-0001"

    def test_far_apart_hashes_not_duplicate(self):
        ph1 = "0000000000000000"  # all zeros
        ph2 = "ffffffffffffffff"  # all ones — hamming distance = 64
        assert abs(imagehash.hex_to_hash(ph1) - imagehash.hex_to_hash(ph2)) > HAMMING_THRESHOLD
        dup_module._seen_hashes[ph1] = "photo-0001"
        assert check_duplicate(ph2, "photo-0002") is None

    def test_registers_in_seen_hashes(self):
        ph = "0000000000000001"
        check_duplicate(ph, "photo-0001")
        assert "photo-0001" in dup_module._seen_hashes.values()

    def test_reset_clears_state(self):
        dup_module._seen_hashes["abc"] = "photo-0001"
        reset()
        assert len(dup_module._seen_hashes) == 0
