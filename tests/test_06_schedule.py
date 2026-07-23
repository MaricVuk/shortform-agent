"""Izolovani testovi za korak 6 (mock scheduling)."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from pipeline import schedule


def test_writes_metadata_json(tmp_path):
    md = schedule.prepare_schedule(
        "run123", "My Title", tmp_path / "v.mp4", tmp_path,
        now=datetime(2026, 7, 23, 10, 0, 0),
    )
    path = Path(md["metadata_path"])
    assert path.exists()
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk["run_id"] == "run123"
    assert on_disk["title"] == "My Title"
    assert on_disk["status"] == "ready_to_publish"


def test_suggested_time_is_next_day_1800(tmp_path):
    md = schedule.prepare_schedule(
        "r", "t", "v.mp4", tmp_path, now=datetime(2026, 7, 23, 10, 0, 0),
    )
    assert md["suggested_publish_time"] == "2026-07-24T18:00:00"


def test_default_platforms_used(tmp_path):
    md = schedule.prepare_schedule("r", "t", "v.mp4", tmp_path)
    import config
    assert md["platforms"] == config.DEFAULT_PLATFORMS


def test_explicit_platforms_override(tmp_path):
    md = schedule.prepare_schedule(
        "r", "t", "v.mp4", tmp_path, platforms=["tiktok"]
    )
    assert md["platforms"] == ["tiktok"]


def test_empty_run_id_raises(tmp_path):
    with pytest.raises(ValueError):
        schedule.prepare_schedule("  ", "t", "v.mp4", tmp_path)


def test_no_real_upload_note_present(tmp_path):
    md = schedule.prepare_schedule("r", "t", "v.mp4", tmp_path)
    assert "app review" in md["note"]
