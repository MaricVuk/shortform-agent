"""Zajednicki LLM helperi (Groq) i self-eval tipovi.

Koraci 2 (skripta) i 4 (media) koriste isti obrazac: pozovi LLM, procitaj
tekst, izvuci JSON, i self-eval sa ocenom 1-10. Ova logika zivi ovde da se
ne duplira izmedju modula.

LLM provajder je Groq (OpenAI-kompatibilan chat completions API, free tier sa
znatno velikodusnijim rate limitom od Gemini free tier-a — bitno jer graf u
jednom run-u pravi 4-5 uzastopnih LLM poziva). `build_model()` vraca tanak
wrapper koji izlaze `.generate_content(prompt) -> object.text`, isti interfejs
koji su moduli i testovi vec ocekivali (lak provider-swap bez diranja poziva).

Ovo je obican (ne-numerisan) submodul pa se importuje standardno:
    from pipeline.llm_utils import build_model, generate_text, extract_json, Evaluation
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import config


@dataclass
class Evaluation:
    """Rezultat LLM self-eval-a: numericka ocena + tekstualni feedback.

    Deljeno izmedju koraka 2 (kvalitet skripte) i 4 (relevantnost vizuala).
    """

    score: int
    feedback: str

    def passes(self, threshold: int) -> bool:
        return self.score >= threshold


class _GroqTextResponse:
    """Minimalni response-wrapper: samo `.text`, kao Gemini SDK odgovor."""

    def __init__(self, text: str):
        self.text = text


class _GroqModel:
    """Tanak adapter oko Groq chat completions -> `.generate_content(prompt).text`.

    Drzi isti interfejs koji su moduli/testovi vec pisali za Gemini, tako da
    je zamena provajdera lokalizovana u ovoj klasi (a ne razbacana po pipeline-u).
    """

    def __init__(self, client: Any, model_name: str):
        self._client = client
        self._model_name = model_name

    def generate_content(self, prompt: str) -> _GroqTextResponse:
        completion = self._client.chat.completions.create(
            model=self._model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        text = completion.choices[0].message.content or ""
        return _GroqTextResponse(text)


def build_model() -> Any:
    """Napravi Groq model (lokalni import da testovi ne zavise od paketa)."""
    from groq import Groq

    client = Groq(api_key=config.require("GROQ_API_KEY"))
    return _GroqModel(client, config.GROQ_MODEL)


def generate_text(model: Any, prompt: str) -> str:
    """Jedinstvena tacka za sve LLM pozive: vrati (ocisceni) tekst odgovora."""
    response = model.generate_content(prompt)
    return (getattr(response, "text", None) or "").strip()


def extract_json(text: str) -> dict[str, Any]:
    """Izvuci prvi JSON objekat iz LLM odgovora (tolerantno na ```json fence)."""
    if not text:
        return {}
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text
    brace = re.search(r"\{.*\}", candidate, re.DOTALL)
    if not brace:
        return {}
    try:
        result = json.loads(brace.group(0))
        return result if isinstance(result, dict) else {}
    except json.JSONDecodeError:
        return {}


def coerce_score(value: Any) -> int:
    """Pretvori ocenu u int i clampuj na [1, 10]; nevalidno -> 1 (najgore)."""
    try:
        score = int(round(float(value)))
    except (TypeError, ValueError):
        return 1
    return max(1, min(10, score))
