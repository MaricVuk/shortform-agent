"""Izolovani testovi za korak 3 (TTS) — bez mreze i bez ffmpeg-a."""
from __future__ import annotations

from pathlib import Path

import pytest

from pipeline import tts


def _fake_runner_factory(recorder: dict):
    """Runner koji zapise dummy fajl i zabelezi argumente."""

    def runner(text, voice, out_path: Path):
        recorder["text"] = text
        recorder["voice"] = voice
        recorder["out_path"] = out_path
        out_path.write_bytes(b"fake-audio")

    return runner


def test_synthesize_writes_file_and_returns_duration(tmp_path):
    rec: dict = {}
    out = tmp_path / "audio.mp3"
    result = tts.synthesize(
        "Hello world",
        out,
        tts_runner=_fake_runner_factory(rec),
        duration_probe=lambda p: 12.5,
    )
    assert Path(result.audio_path) == out
    assert out.exists()
    assert result.duration == 12.5


def test_synthesize_creates_parent_dirs(tmp_path):
    out = tmp_path / "nested" / "deep" / "audio.mp3"
    tts.synthesize(
        "hi", out, tts_runner=_fake_runner_factory({}), duration_probe=lambda p: 1.0
    )
    assert out.exists()


def test_synthesize_uses_config_default_voice(tmp_path):
    rec: dict = {}
    tts.synthesize(
        "hi",
        tmp_path / "a.mp3",
        tts_runner=_fake_runner_factory(rec),
        duration_probe=lambda p: 1.0,
    )
    import config

    assert rec["voice"] == config.TTS_VOICE


def test_synthesize_passes_explicit_voice(tmp_path):
    rec: dict = {}
    tts.synthesize(
        "hi",
        tmp_path / "a.mp3",
        voice="en-GB-RyanNeural",
        tts_runner=_fake_runner_factory(rec),
        duration_probe=lambda p: 1.0,
    )
    assert rec["voice"] == "en-GB-RyanNeural"


def test_synthesize_strips_text(tmp_path):
    rec: dict = {}
    tts.synthesize(
        "  hi there  ",
        tmp_path / "a.mp3",
        tts_runner=_fake_runner_factory(rec),
        duration_probe=lambda p: 1.0,
    )
    assert rec["text"] == "hi there"


def test_empty_text_raises(tmp_path):
    with pytest.raises(ValueError):
        tts.synthesize("   ", tmp_path / "a.mp3")


def test_runner_that_writes_no_file_raises(tmp_path):
    def bad_runner(text, voice, out_path):
        pass  # ne pise nista

    with pytest.raises(RuntimeError):
        tts.synthesize(
            "hi", tmp_path / "a.mp3", tts_runner=bad_runner, duration_probe=lambda p: 1
        )


def test_synthesize_defaults_to_empty_captions_when_runner_returns_none(tmp_path):
    result = tts.synthesize(
        "hi", tmp_path / "a.mp3",
        tts_runner=_fake_runner_factory({}), duration_probe=lambda p: 1.0,
    )
    assert result.captions == []


def test_synthesize_captures_word_level_captions(tmp_path):
    def runner(text, voice, out_path: Path):
        out_path.write_bytes(b"fake-audio")
        return [
            tts.Caption(text="Hello", start=0.0, end=0.4),
            tts.Caption(text="world", start=0.4, end=0.9),
        ]

    result = tts.synthesize(
        "Hello world", tmp_path / "a.mp3", tts_runner=runner, duration_probe=lambda p: 1.0,
    )
    assert len(result.captions) == 2
    assert result.captions[0].text == "Hello"
    assert result.captions[1].end == 0.9
