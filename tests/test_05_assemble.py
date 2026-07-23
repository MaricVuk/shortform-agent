"""Izolovani testovi za korak 5 (montaza) — fake renderer, bez ffmpeg-a."""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from pipeline import assemble


@dataclass
class _Cap:
    text: str
    start: float
    end: float


# --- plan_slides (cista funkcija) ---

def test_plan_slides_sums_to_duration():
    durations = assemble.plan_slides(4, 10.0)
    assert len(durations) == 4
    assert round(sum(durations), 3) == 10.0


def test_plan_slides_even_split():
    durations = assemble.plan_slides(5, 10.0)
    assert durations[0] == 2.0


def test_plan_slides_last_absorbs_rounding():
    durations = assemble.plan_slides(3, 10.0)
    assert round(sum(durations), 3) == 10.0  # 3.333*2 + ostatak


def test_plan_slides_zero_assets_raises():
    with pytest.raises(ValueError):
        assemble.plan_slides(0, 10.0)


def test_plan_slides_zero_duration_raises():
    with pytest.raises(ValueError):
        assemble.plan_slides(3, 0)


# --- group_captions / _active_chunk (ciste funkcije) ---

def test_group_captions_groups_by_chunk_size():
    caps = [_Cap("a", 0.0, 0.2), _Cap("b", 0.2, 0.4), _Cap("c", 0.4, 0.6), _Cap("d", 0.6, 0.8)]
    groups = assemble.group_captions(caps, chunk_size=3)
    assert groups == [("a b c", 0.0, 0.6), ("d", 0.6, 0.8)]


def test_group_captions_empty_list():
    assert assemble.group_captions([]) == []


def test_group_captions_uses_config_default(monkeypatch):
    import config
    monkeypatch.setattr(config, "CAPTION_CHUNK_SIZE", 2)
    caps = [_Cap("a", 0.0, 0.1), _Cap("b", 0.1, 0.2), _Cap("c", 0.2, 0.3)]
    groups = assemble.group_captions(caps)
    assert groups[0] == ("a b", 0.0, 0.2)


def test_active_chunk_finds_matching_window():
    chunks = [("hello", 0.0, 1.0), ("world", 1.0, 2.0)]
    assert assemble._active_chunk(chunks, 0.5) == "hello"
    assert assemble._active_chunk(chunks, 1.5) == "world"


def test_active_chunk_none_in_gap():
    chunks = [("hello", 0.0, 1.0), ("world", 2.0, 3.0)]
    assert assemble._active_chunk(chunks, 1.5) is None


# --- assemble_video (injektovan renderer) ---

def test_assemble_calls_renderer_with_planned_durations(tmp_path):
    captured = {}

    def fake_renderer(audio, images, durations, out_path, captions, size, fps):
        captured["images"] = images
        captured["durations"] = durations
        captured["size"] = size
        captured["captions"] = captions
        out_path.write_bytes(b"mp4")

    out = tmp_path / "final.mp4"
    caps = [_Cap("hi", 0.0, 0.5)]
    result = assemble.assemble_video(
        tmp_path / "a.mp3", 9.0, ["i1.jpg", "i2.jpg", "i3.jpg"], out,
        captions=caps, renderer=fake_renderer,
    )
    assert result == str(out)
    assert out.exists()
    assert len(captured["durations"]) == 3
    assert round(sum(captured["durations"]), 3) == 9.0
    assert captured["captions"] == caps
    import config
    assert captured["size"] == (config.VIDEO_WIDTH, config.VIDEO_HEIGHT)


def test_assemble_defaults_captions_to_empty_list(tmp_path):
    captured = {}

    def fake_renderer(audio, images, durations, out_path, captions, size, fps):
        captured["captions"] = captions
        out_path.write_bytes(b"mp4")

    assemble.assemble_video(
        tmp_path / "a.mp3", 5.0, ["i1.jpg"], tmp_path / "f.mp4", renderer=fake_renderer
    )
    assert captured["captions"] == []


def test_assemble_no_images_raises(tmp_path):
    with pytest.raises(ValueError):
        assemble.assemble_video(tmp_path / "a.mp3", 9.0, [], tmp_path / "f.mp4")


def test_assemble_creates_output_parent(tmp_path):
    def fake_renderer(audio, images, durations, out_path, captions, size, fps):
        out_path.write_bytes(b"mp4")

    out = tmp_path / "sub" / "dir" / "final.mp4"
    assemble.assemble_video(
        tmp_path / "a.mp3", 5.0, ["i1.jpg"], out, renderer=fake_renderer
    )
    assert out.exists()


def test_assemble_raises_if_renderer_writes_nothing(tmp_path):
    def bad_renderer(audio, images, durations, out_path, captions, size, fps):
        pass

    with pytest.raises(RuntimeError):
        assemble.assemble_video(
            tmp_path / "a.mp3", 5.0, ["i1.jpg"], tmp_path / "f.mp4",
            renderer=bad_renderer,
        )
