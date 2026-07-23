"""Izolovani testovi za korak 4 (media) — fake Pexels klijent i fake LLM."""
from __future__ import annotations

from pathlib import Path

import pytest

from pipeline import media


class FakeModel:
    def __init__(self, responses):
        self._responses = list(responses)
        self.prompts = []

    def generate_content(self, prompt):
        self.prompts.append(prompt)
        text = self._responses.pop(0) if self._responses else ""
        return type("Resp", (), {"text": text})()


class FakePexels:
    """Vraca pripremljene fotke po keyword-u i pise dummy fajlove pri download-u."""

    def __init__(self, photos_by_keyword):
        self._photos = photos_by_keyword
        self.searched = []
        self.downloaded = []

    def search_photos(self, query, per_page):
        self.searched.append((query, per_page))
        return self._photos.get(query, [])[:per_page]

    def download(self, url, dest: Path):
        self.downloaded.append((url, dest))
        dest.write_bytes(b"img")


def _photo(pid, alt="a cat"):
    return {
        "id": pid,
        "alt": alt,
        "url": f"https://pexels.com/{pid}",
        "src": {"large": f"https://img/{pid}.jpg"},
    }


# --- extract_keywords ---

def test_extract_keywords_parses_list():
    model = FakeModel(['{"keywords": ["night sky", "stars", " "]}'])
    kws = media.extract_keywords("Space", "about stars", model=model)
    assert kws == ["night sky", "stars"]  # prazan preskocen


def test_extract_keywords_empty_on_missing():
    model = FakeModel(['{}'])
    assert media.extract_keywords("t", "s", model=model) == []


# --- fetch_media ---

def test_fetch_downloads_up_to_count(tmp_path):
    client = FakePexels({"cats": [_photo(1), _photo(2), _photo(3)]})
    assets = media.fetch_media(["cats"], count=2, out_dir=tmp_path, client=client)
    assert len(assets) == 2
    assert all(Path(a.path).exists() for a in assets)
    assert len(client.downloaded) == 2


def test_fetch_spans_multiple_keywords(tmp_path):
    client = FakePexels({"cats": [_photo(1)], "dogs": [_photo(2), _photo(3)]})
    assets = media.fetch_media(["cats", "dogs"], count=3, out_dir=tmp_path, client=client)
    assert len(assets) == 3
    assert {a.keyword for a in assets} == {"cats", "dogs"}


def test_fetch_stops_early_when_count_reached(tmp_path):
    client = FakePexels({"cats": [_photo(1)], "dogs": [_photo(2)]})
    media.fetch_media(["cats", "dogs"], count=1, out_dir=tmp_path, client=client)
    # drugi keyword se ne pretrazuje kad je count vec dostignut
    assert client.searched == [("cats", 1)]


def test_fetch_captures_description_and_source(tmp_path):
    client = FakePexels({"cats": [_photo(1, alt="fluffy cat")]})
    assets = media.fetch_media(["cats"], count=1, out_dir=tmp_path, client=client)
    assert assets[0].description == "fluffy cat"
    assert assets[0].source_url == "https://pexels.com/1"


def test_fetch_skips_photos_without_usable_url(tmp_path):
    client = FakePexels({"cats": [{"id": 9, "alt": "x", "url": "u", "src": {}}]})
    assets = media.fetch_media(["cats"], count=1, out_dir=tmp_path, client=client)
    assert assets == []


def test_fetch_empty_keywords_raises(tmp_path):
    with pytest.raises(ValueError):
        media.fetch_media([], count=1, out_dir=tmp_path, client=FakePexels({}))


def test_fetch_bad_count_raises(tmp_path):
    with pytest.raises(ValueError):
        media.fetch_media(["cats"], count=0, out_dir=tmp_path, client=FakePexels({}))


# --- evaluate_media ---

def test_evaluate_media_parses_score():
    model = FakeModel(['{"score": 9, "feedback": "great match"}'])
    assets = [media.MediaAsset("p", "u", "a cat", "cats")]
    ev = media.evaluate_media("script about cats", assets, model=model)
    assert ev.score == 9


def test_evaluate_media_no_assets_scores_one():
    ev = media.evaluate_media("script", [], model=FakeModel([]))
    assert ev.score == 1
    assert "Nijedan" in ev.feedback


def test_evaluate_media_description_in_prompt():
    model = FakeModel(['{"score": 5, "feedback": "meh"}'])
    assets = [media.MediaAsset("p", "u", "a fluffy cat", "cats")]
    media.evaluate_media("script", assets, model=model)
    assert "a fluffy cat" in model.prompts[0]


# --- suggest_keywords_from_feedback ---

def test_suggest_keywords_uses_feedback_and_returns_list():
    model = FakeModel(['{"keywords": ["galaxy", "nebula"]}'])
    kws = media.suggest_keywords_from_feedback(
        "try space imagery", "Space", "script", model=model
    )
    assert kws == ["galaxy", "nebula"]
    assert "try space imagery" in model.prompts[0]
