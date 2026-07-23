"""Korak 2 — Naslov + skripta (Gemini) sa self-eval agent logikom.

Ovaj modul izlaze CISTE funkcije; petlju "generisi -> oceni -> po potrebi
regenerisi" vodi LangGraph graf (orchestration/graph.py). Time je agentska
odluka (retry ili nastavi) eksplicitna u grafu, a ne skrivena u modulu.

- `generate_script(topic, references, feedback=None)` -> ScriptDraft
- `evaluate_script(draft)` -> Evaluation (ocena 1-10 + obrazlozenje)

Zajednicki LLM helperi (poziv modela, JSON parsing, Evaluation) su u
`pipeline/llm_utils.py` i dele se sa korakom 4.

Dizajn za testiranje: obe funkcije primaju opcioni `model` (bilo sta sa
`.generate_content(prompt).text`). U testu se prosledi fake model, pa nema
mreznog poziva ni trosenja kvote.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pipeline.llm_utils import (
    Evaluation,
    build_model,
    coerce_score,
    extract_json,
    generate_text,
)

import config

# Re-export da `script.Evaluation` i dalje radi za pozivaoce/testove.
__all__ = ["ScriptDraft", "Evaluation", "generate_script", "evaluate_script"]


@dataclass
class ScriptDraft:
    """Jedan draft: naslov + telo skripte (spremno za TTS)."""

    title: str
    script: str


# --- Prompt-ovi ---

def _format_references(references: list[Any]) -> str:
    lines = []
    for i, ref in enumerate(references, 1):
        title = getattr(ref, "title", "") or ""
        snippet = getattr(ref, "snippet", "") or ""
        lines.append(f"{i}. {title}: {snippet}")
    return "\n".join(lines) if lines else "(nema referenci)"


def _generation_prompt(topic: str, references: list[Any], feedback: str | None) -> str:
    refs = _format_references(references)
    feedback_block = ""
    if feedback:
        feedback_block = (
            "\nPrethodni pokusaj je dobio ovu kritiku — popravi je u novoj verziji:\n"
            f"{feedback}\n"
        )
    return (
        "You are a scriptwriter for short-form vertical videos "
        "(YouTube Shorts / TikTok / Reels), 20-40 seconds when read aloud.\n"
        f"Topic: {topic}\n\n"
        f"Research references:\n{refs}\n"
        f"{feedback_block}\n"
        "Write a punchy script with a strong hook in the first 3 seconds. "
        "No stage directions, no emojis, just spoken narration.\n"
        'Return ONLY valid JSON: {"title": "<catchy title>", '
        '"script": "<narration text>"}'
    )


def _evaluation_prompt(draft: ScriptDraft) -> str:
    return (
        "You are a critical short-form video editor. Rate the script below.\n"
        f"Title: {draft.title}\n"
        f"Script: {draft.script}\n\n"
        "Judge mainly: is the hook in the first 3 seconds strong enough to stop "
        "a scroll? Also pacing and clarity.\n"
        "Give an integer score 1-10 and one or two sentences of concrete, "
        "actionable feedback on what to fix.\n"
        'Return ONLY valid JSON: {"score": <int 1-10>, "feedback": "<text>"}'
    )


# --- Javne funkcije ---

def generate_script(
    topic: str,
    references: list[Any],
    feedback: str | None = None,
    model: Any | None = None,
) -> ScriptDraft:
    """Generisi naslov + skriptu. `feedback` (iz prethodnog eval-a) se ubacuje
    u prompt na retry-u da model popravi konkretne slabosti."""
    topic = (topic or "").strip()
    if not topic:
        raise ValueError("Tema (topic) ne sme biti prazna.")

    if model is None:
        model = build_model()

    text = generate_text(model, _generation_prompt(topic, references, feedback))
    data = extract_json(text)
    title = (data.get("title") or "").strip()
    script = (data.get("script") or "").strip()
    if not script:
        raise ValueError(f"Model nije vratio skriptu. Sirovi odgovor: {text[:200]}")
    return ScriptDraft(title=title or topic, script=script)


def evaluate_script(draft: ScriptDraft, model: Any | None = None) -> Evaluation:
    """LLM ocenjuje sopstveni draft (hook/pacing) -> ocena 1-10 + feedback."""
    if model is None:
        model = build_model()

    text = generate_text(model, _evaluation_prompt(draft))
    data = extract_json(text)
    return Evaluation(score=coerce_score(data.get("score")), feedback=(data.get("feedback") or "").strip())
