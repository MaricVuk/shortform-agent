"""Testovi LangGraph grafa — grananje retry/continue za oba checkpointa.

Sve zavisnosti su fake (bez mreze, ffmpeg-a, ni kvota). Fokus je na tome da
graf DONOSI ISPRAVNE ODLUKE: retry na nisku ocenu, nastavak na visoku, i
zaustavljanje na max broju pokusaja.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import config
from orchestration import graph as G


class ScriptedModel:
    """Fake Gemini: vraca odgovor po TIPU prompta; ocene iz zadatih listi."""

    def __init__(self, script_scores, media_scores):
        self._script_scores = list(script_scores)
        self._media_scores = list(media_scores)

    def _next(self, queue):
        # kad ponestane, ponavljaj poslednju vrednost (za max-attempts test)
        return queue.pop(0) if len(queue) > 1 else (queue[0] if queue else 1)

    def generate_content(self, prompt):
        if "Rate the script" in prompt:
            score = self._next(self._script_scores)
            text = json.dumps({"score": score, "feedback": "fix the hook"})
        elif "checking whether stock visuals" in prompt:
            score = self._next(self._media_scores)
            text = json.dumps({"score": score, "feedback": "try space imagery"})
        elif "stock-footage search terms" in prompt:
            text = json.dumps({"keywords": ["stars", "galaxy"]})
        elif "scriptwriter" in prompt:
            text = json.dumps({"title": "Cosmos", "script": "Look up tonight."})
        else:
            text = "{}"
        return type("Resp", (), {"text": text})()


class FakeTavily:
    def search(self, **kwargs):
        return {"results": [{"title": "Space", "content": "big", "url": "u"}]}


class FakePexels:
    def search_photos(self, query, per_page):
        return [
            {"id": i, "alt": f"{query} {i}", "url": f"u{i}",
             "src": {"large": f"img{i}.jpg"}}
            for i in range(per_page)
        ]

    def download(self, url, dest: Path):
        dest.write_bytes(b"img")


def _fake_tts_runner(text, voice, out_path: Path):
    out_path.write_bytes(b"audio")


def _fake_renderer(audio, images, durations, out_path, title, size, fps):
    out_path.write_bytes(b"mp4")


def _deps(script_scores, media_scores):
    model = ScriptedModel(script_scores, media_scores)
    return G.Deps(
        tavily_client=FakeTavily(),
        script_model=model,
        media_model=model,
        pexels_client=FakePexels(),
        tts_runner=_fake_tts_runner,
        duration_probe=lambda p: 8.0,
        renderer=_fake_renderer,
        now=datetime(2026, 7, 23, 10, 0, 0),
        media_count=3,
    )


def test_full_pipeline_happy_path(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    state = G.run_pipeline("space", "run_happy", deps=_deps([9], [9]))

    assert state["script_attempts"] == 1
    assert state["media_attempts"] == 1
    assert state["metadata"]["status"] == "ready_to_publish"
    assert Path(state["video_path"]).exists()
    assert (tmp_path / "run_happy" / "metadata.json").exists()


def test_script_retries_until_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    # prva ocena niska (4), druga prolazi (9) -> tacno 2 generacije skripte
    state = G.run_pipeline("space", "run_retry", deps=_deps([4, 9], [9]))
    assert state["script_attempts"] == 2
    assert state["script_score"] == 9


def test_script_stops_at_max_attempts(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    # ocena uvek niska -> staje na SCRIPT_MAX_ATTEMPTS, ali ipak nastavlja dalje
    state = G.run_pipeline("space", "run_max", deps=_deps([2], [9]))
    assert state["script_attempts"] == config.SCRIPT_MAX_ATTEMPTS
    assert state["metadata"]["status"] == "ready_to_publish"  # nastavio do kraja


def test_media_retries_then_passes(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    # vizuali: prvo lose (4), pa dobro (9) -> refine_keywords -> ponovni fetch
    state = G.run_pipeline("space", "run_media", deps=_deps([9], [4, 9]))
    assert state["media_attempts"] == 2
    assert state["media_score"] == 9


def test_media_stops_at_max_attempts(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    state = G.run_pipeline("space", "run_media_max", deps=_deps([9], [1]))
    assert state["media_attempts"] == config.MEDIA_MAX_ATTEMPTS
    assert Path(state["video_path"]).exists()
