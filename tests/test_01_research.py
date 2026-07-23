"""Izolovani testovi za korak 1 (research) — mock Tavily klijent, bez mreze."""
from __future__ import annotations

import pytest

from pipeline import research


class FakeTavilyClient:
    """Minimalni mock: belezi poziv i vraca unapred pripremljen odgovor."""

    def __init__(self, response: dict):
        self._response = response
        self.last_call: dict | None = None

    def search(self, **kwargs):
        self.last_call = kwargs
        return self._response


def _sample_response() -> dict:
    return {
        "results": [
            {
                "title": "Fact One",
                "content": "The sun is very big.",
                "url": "https://example.com/1",
            },
            {
                "title": "Fact Two",
                "content": "Space is mostly empty.",
                "url": "https://example.com/2",
            },
        ]
    }


def test_returns_structured_references():
    client = FakeTavilyClient(_sample_response())
    refs = research.run_research("space facts", max_results=5, client=client)

    assert len(refs) == 2
    assert refs[0].title == "Fact One"
    assert refs[0].snippet == "The sun is very big."
    assert refs[0].url == "https://example.com/1"


def test_passes_query_and_max_results_to_client():
    client = FakeTavilyClient(_sample_response())
    research.run_research("  space facts  ", max_results=3, client=client)

    assert client.last_call["query"] == "space facts"  # trimovano
    assert client.last_call["max_results"] == 3


def test_respects_max_results_limit():
    client = FakeTavilyClient(_sample_response())
    refs = research.run_research("space", max_results=1, client=client)
    assert len(refs) == 1


def test_skips_empty_entries():
    resp = {"results": [{"title": "", "content": "", "url": "https://x"}]}
    refs = research.run_research("space", client=FakeTavilyClient(resp))
    assert refs == []


def test_snippet_falls_back_to_snippet_key():
    resp = {"results": [{"title": "T", "snippet": "alt text", "url": "u"}]}
    refs = research.run_research("space", client=FakeTavilyClient(resp))
    assert refs[0].snippet == "alt text"


def test_empty_topic_raises():
    with pytest.raises(ValueError):
        research.run_research("   ", client=FakeTavilyClient(_sample_response()))


def test_missing_results_key_returns_empty():
    refs = research.run_research("space", client=FakeTavilyClient({}))
    assert refs == []


def test_reference_to_dict_roundtrip():
    ref = research.Reference(title="T", snippet="S", url="U")
    assert ref.to_dict() == {"title": "T", "snippet": "S", "url": "U"}
