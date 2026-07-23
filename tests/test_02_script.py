"""Izolovani testovi za korak 2 (skripta + self-eval) — fake LLM model."""
from __future__ import annotations

import pytest

from pipeline import script


class FakeModel:
    """Vraca redom pripremljene tekstualne odgovore; belezi prompt-ove."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.prompts: list[str] = []

    def generate_content(self, prompt):
        self.prompts.append(prompt)
        text = self._responses.pop(0) if self._responses else ""
        return type("Resp", (), {"text": text})()


class _Ref:
    def __init__(self, title, snippet):
        self.title = title
        self.snippet = snippet


# --- generate_script ---

def test_generate_parses_title_and_script():
    model = FakeModel(['{"title": "Space!", "script": "Did you know..."}'])
    draft = script.generate_script("space", [], model=model)
    assert draft.title == "Space!"
    assert draft.script == "Did you know..."


def test_generate_handles_json_code_fence():
    model = FakeModel(['```json\n{"title": "T", "script": "S"}\n```'])
    draft = script.generate_script("space", [], model=model)
    assert draft.title == "T"
    assert draft.script == "S"


def test_generate_includes_references_in_prompt():
    model = FakeModel(['{"title": "T", "script": "S"}'])
    script.generate_script("space", [_Ref("Sun", "big star")], model=model)
    assert "Sun" in model.prompts[0]
    assert "big star" in model.prompts[0]


def test_generate_injects_feedback_on_retry():
    model = FakeModel(['{"title": "T", "script": "S"}'])
    script.generate_script("space", [], feedback="hook is weak", model=model)
    assert "hook is weak" in model.prompts[0]


def test_generate_empty_topic_raises():
    with pytest.raises(ValueError):
        script.generate_script("  ", [], model=FakeModel(['{}']))


def test_generate_missing_script_raises():
    model = FakeModel(['{"title": "only title"}'])
    with pytest.raises(ValueError):
        script.generate_script("space", [], model=model)


def test_generate_title_falls_back_to_topic():
    model = FakeModel(['{"script": "S"}'])
    draft = script.generate_script("mytopic", [], model=model)
    assert draft.title == "mytopic"


# --- evaluate_script ---

def test_evaluate_parses_score_and_feedback():
    model = FakeModel(['{"score": 8, "feedback": "solid hook"}'])
    ev = script.evaluate_script(script.ScriptDraft("T", "S"), model=model)
    assert ev.score == 8
    assert ev.feedback == "solid hook"


def test_evaluate_clamps_out_of_range_score():
    model = FakeModel(['{"score": 99, "feedback": "x"}'])
    ev = script.evaluate_script(script.ScriptDraft("T", "S"), model=model)
    assert ev.score == 10


def test_evaluate_invalid_score_defaults_to_one():
    model = FakeModel(['{"score": "not a number", "feedback": "x"}'])
    ev = script.evaluate_script(script.ScriptDraft("T", "S"), model=model)
    assert ev.score == 1


def test_evaluate_unparseable_output_is_safe():
    model = FakeModel(["totally not json"])
    ev = script.evaluate_script(script.ScriptDraft("T", "S"), model=model)
    assert ev.score == 1
    assert ev.feedback == ""


# --- Evaluation.passes ---

def test_passes_threshold():
    assert script.Evaluation(7, "").passes(7) is True
    assert script.Evaluation(6, "").passes(7) is False
