"""Izolovani testovi za korak 5 (montaza) — fake renderer, bez ffmpeg-a."""
from __future__ import annotations

import pytest

from pipeline import assemble


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


# --- assemble_video (injektovan renderer) ---

def test_assemble_calls_renderer_with_planned_durations(tmp_path):
    captured = {}

    def fake_renderer(audio, images, durations, out_path, title, size, fps):
        captured["images"] = images
        captured["durations"] = durations
        captured["size"] = size
        out_path.write_bytes(b"mp4")

    out = tmp_path / "final.mp4"
    result = assemble.assemble_video(
        tmp_path / "a.mp3", 9.0, ["i1.jpg", "i2.jpg", "i3.jpg"], out,
        title="Hi", renderer=fake_renderer,
    )
    assert result == str(out)
    assert out.exists()
    assert len(captured["durations"]) == 3
    assert round(sum(captured["durations"]), 3) == 9.0
    import config
    assert captured["size"] == (config.VIDEO_WIDTH, config.VIDEO_HEIGHT)


def test_assemble_no_images_raises(tmp_path):
    with pytest.raises(ValueError):
        assemble.assemble_video(tmp_path / "a.mp3", 9.0, [], tmp_path / "f.mp4")


def test_assemble_creates_output_parent(tmp_path):
    def fake_renderer(audio, images, durations, out_path, title, size, fps):
        out_path.write_bytes(b"mp4")

    out = tmp_path / "sub" / "dir" / "final.mp4"
    assemble.assemble_video(
        tmp_path / "a.mp3", 5.0, ["i1.jpg"], out, renderer=fake_renderer
    )
    assert out.exists()


def test_assemble_raises_if_renderer_writes_nothing(tmp_path):
    def bad_renderer(audio, images, durations, out_path, title, size, fps):
        pass

    with pytest.raises(RuntimeError):
        assemble.assemble_video(
            tmp_path / "a.mp3", 5.0, ["i1.jpg"], tmp_path / "f.mp4",
            renderer=bad_renderer,
        )
