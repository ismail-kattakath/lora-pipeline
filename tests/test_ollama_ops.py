"""Tests for ollama_client module."""

import json
from unittest.mock import patch

import pytest

import lora_pipeline.config as cfg
from lora_pipeline.ollama_client import TimeoutError, analyse_with_backoff, call_qwen


def _stream(content):
    """Yield content as single-chunk stream, matching ollama streaming API."""
    yield {"message": {"content": content}}


class TestCallQwen:
    def test_returns_parsed_json(self):
        payload = json.dumps({"folder": "solo", "quality_score": 4})
        with patch("lora_pipeline.ollama_client._ollama.chat", return_value=_stream(payload)):
            assert call_qwen("b64", "stem", cfg.MODEL)["folder"] == "solo"

    def test_strips_markdown_fences(self):
        payload = "```json\n" + json.dumps({"folder": "group"}) + "\n```"
        with patch("lora_pipeline.ollama_client._ollama.chat", return_value=_stream(payload)):
            assert call_qwen("b64", "stem", cfg.MODEL)["folder"] == "group"

    def test_raises_on_bad_json(self):
        with patch("lora_pipeline.ollama_client._ollama.chat", return_value=_stream("NOT JSON")):
            with pytest.raises(json.JSONDecodeError):
                call_qwen("b64", "stem", cfg.MODEL)

    def test_raises_timeout_when_hung(self, monkeypatch):
        monkeypatch.setattr(cfg, "CALL_TIMEOUT", 0.1)

        def hang(*a, **kw):
            import time

            time.sleep(10)

        with patch("lora_pipeline.ollama_client._ollama.chat", side_effect=hang):
            with pytest.raises(TimeoutError):
                call_qwen("b64", "stem", cfg.MODEL)

    def test_raises_on_ollama_exception(self):
        with patch(
            "lora_pipeline.ollama_client._ollama.chat", side_effect=RuntimeError("model not found")
        ):
            with pytest.raises(RuntimeError):
                call_qwen("b64", "stem", cfg.MODEL)


GOOD = {
    "folder": "solo",
    "caption": "test",
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
    "tags": [],
}


class TestAnalyseWithBackoff:
    def test_returns_on_first_try(self):
        with patch("lora_pipeline.ollama_client.call_qwen", return_value=GOOD):
            assert analyse_with_backoff("b64", "photo-0001", cfg.MODEL)["folder"] == "solo"

    def test_retries_on_json_error_then_succeeds(self):
        calls = [json.JSONDecodeError("bad", "", 0), json.JSONDecodeError("bad", "", 0), GOOD]
        with patch("lora_pipeline.ollama_client.call_qwen", side_effect=calls):
            with patch("time.sleep"):
                assert analyse_with_backoff("b64", "x", cfg.MODEL)["folder"] == "solo"

    def test_returns_error_after_max_retries(self, monkeypatch):
        monkeypatch.setattr(cfg, "MAX_RETRIES", 2)
        with patch(
            "lora_pipeline.ollama_client.call_qwen", side_effect=json.JSONDecodeError("bad", "", 0)
        ):
            with patch("time.sleep"):
                result = analyse_with_backoff("b64", "x", cfg.MODEL)
        assert "_parse_error" in result

    def test_attempts_ollama_restart_on_connection_error(self, monkeypatch):
        monkeypatch.setattr(cfg, "MAX_RETRIES", 2)
        calls = [ConnectionRefusedError(), ConnectionRefusedError(), GOOD]
        with patch("lora_pipeline.ollama_client.call_qwen", side_effect=calls):
            with patch("lora_pipeline.ollama_client.ollama_running", return_value=False):
                with patch(
                    "lora_pipeline.ollama_client.restart_ollama", return_value=True
                ) as mock_restart:
                    with patch("time.sleep"):
                        analyse_with_backoff("b64", "x", cfg.MODEL)
        mock_restart.assert_called()

    def test_backoff_increases(self, monkeypatch):
        monkeypatch.setattr(cfg, "MAX_RETRIES", 3)
        monkeypatch.setattr(cfg, "BACKOFF_BASE", 2)
        with patch(
            "lora_pipeline.ollama_client.call_qwen", side_effect=json.JSONDecodeError("x", "", 0)
        ):
            with patch("time.sleep") as mock_sleep:
                analyse_with_backoff("b64", "x", cfg.MODEL)
        sleeps = [c.args[0] for c in mock_sleep.call_args_list]
        assert sleeps == sorted(sleeps)
        assert len(set(sleeps)) > 1

    def test_handles_unexpected_exception(self, monkeypatch):
        monkeypatch.setattr(cfg, "MAX_RETRIES", 1)
        with patch("lora_pipeline.ollama_client.call_qwen", side_effect=MemoryError("oom")):
            with patch("time.sleep"):
                result = analyse_with_backoff("b64", "x", cfg.MODEL)
        assert "_error" in result
