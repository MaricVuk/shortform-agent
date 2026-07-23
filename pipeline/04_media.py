"""Korak 4 — Stock vizuali (Pexels) + self-eval relevantnosti.

Tok:
  1. `extract_keywords(title, script)` — LLM izvlaci vizuelne search termine.
  2. `fetch_media(keywords, count, out_dir)` — Pexels pretraga + download slika.
  3. `evaluate_media(script, assets)` — DRUGI agent checkpoint: LLM ocenjuje da
     li su preuzeti vizuali relevantni skripti (1-10) i predlaze nove keywords.

Petlju (ako ocena < prag -> nove keywords -> ponovi fetch) vodi LangGraph graf,
isti obrazac kao self-eval skripte u koraku 2.

Dizajn za testiranje: `fetch_media` prima injektabilni `client` (Pexels-
kompatibilan: `search_photos` + `download`), a LLM funkcije primaju `model`.
Nijedan test ne dira mrezu ni ne trosi kvotu.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from pipeline.llm_utils import (
    Evaluation,
    build_model,
    coerce_score,
    extract_json,
    generate_text,
)

import config


@dataclass
class MediaAsset:
    """Jedan preuzet vizual + metapodaci za self-eval i montazu."""

    path: str
    source_url: str
    description: str  # Pexels 'alt' tekst — koristi se za eval relevantnosti
    keyword: str  # koji keyword ga je pronasao

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


# --- 1. Keyword ekstrakcija (LLM) ---

def _keywords_prompt(title: str, script: str) -> str:
    return (
        "Extract 3-5 concrete, visual stock-footage search terms for the video "
        "below. Prefer nouns/scenes a stock library would have (e.g. 'night sky', "
        "'city traffic'), not abstract words.\n"
        f"Title: {title}\n"
        f"Script: {script}\n"
        'Return ONLY valid JSON: {"keywords": ["term1", "term2", ...]}'
    )


def extract_keywords(
    title: str,
    script: str,
    model: Any | None = None,
) -> list[str]:
    """LLM izvlaci listu vizuelnih search termina iz naslova + skripte."""
    if model is None:
        model = build_model()
    text = generate_text(model, _keywords_prompt(title, script))
    data = extract_json(text)
    keywords = data.get("keywords") or []
    return [str(k).strip() for k in keywords if str(k).strip()]


# --- 2. Pexels fetch ---

class _PexelsClient:
    """Podrazumevani Pexels klijent preko requests-a."""

    SEARCH_URL = "https://api.pexels.com/v1/search"

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key or config.require("PEXELS_API_KEY")

    def search_photos(self, query: str, per_page: int) -> list[dict]:
        import requests

        resp = requests.get(
            self.SEARCH_URL,
            headers={"Authorization": self._api_key},
            params={"query": query, "per_page": per_page},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json().get("photos", [])

    def download(self, url: str, dest: Path) -> None:
        import requests

        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        dest.write_bytes(resp.content)


def _pick_image_url(photo: dict) -> str:
    """Izaberi URL slike iz Pexels 'src' mape (fallback kroz velicine)."""
    src = photo.get("src") or {}
    for size in ("large2x", "large", "original", "medium"):
        if src.get(size):
            return src[size]
    return next(iter(src.values()), "")


def fetch_media(
    keywords: list[str],
    count: int,
    out_dir: str | Path,
    client: Any | None = None,
) -> list[MediaAsset]:
    """Preuzmi do `count` slika sa Pexels-a za date keywords.

    Prolazi kroz keywords redom dok ne skupi `count` slika. Vraca preuzete
    `MediaAsset` objekte (moze i manje od `count` ako Pexels nema dovoljno).
    """
    if not keywords:
        raise ValueError("Nema keywords za pretragu medija.")
    if count <= 0:
        raise ValueError("count mora biti > 0.")

    if client is None:
        client = _PexelsClient()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    assets: list[MediaAsset] = []
    for keyword in keywords:
        if len(assets) >= count:
            break
        remaining = count - len(assets)
        photos = client.search_photos(keyword, per_page=remaining)
        for photo in photos:
            if len(assets) >= count:
                break
            url = _pick_image_url(photo)
            if not url:
                continue
            dest = out_dir / f"media_{len(assets):02d}.jpg"
            client.download(url, dest)
            assets.append(
                MediaAsset(
                    path=str(dest),
                    source_url=photo.get("url", ""),
                    description=(photo.get("alt") or "").strip(),
                    keyword=keyword,
                )
            )
    return assets


# --- 3. Self-eval relevantnosti (DRUGI agent checkpoint) ---

def _media_eval_prompt(script: str, assets: list[MediaAsset]) -> str:
    listing = "\n".join(
        f"- {a.description or '(no description)'} [keyword: {a.keyword}]"
        for a in assets
    ) or "(no assets)"
    return (
        "You are a video editor checking whether stock visuals fit a script.\n"
        f"Script: {script}\n\n"
        f"Downloaded visuals (descriptions):\n{listing}\n\n"
        "Rate how well these visuals match the script's topic and mood, integer "
        "1-10. If below 7, suggest better stock search keywords in the feedback.\n"
        'Return ONLY valid JSON: {"score": <int 1-10>, "feedback": "<text, may '
        'include suggested keywords>"}'
    )


def evaluate_media(
    script: str,
    assets: list[MediaAsset],
    model: Any | None = None,
) -> Evaluation:
    """LLM ocenjuje relevantnost preuzetih vizuala u odnosu na skriptu."""
    if not assets:
        # Nema sta da se oceni — najniza ocena da graf pokusa ponovo.
        return Evaluation(score=1, feedback="Nijedan vizual nije preuzet.")
    if model is None:
        model = build_model()
    text = generate_text(model, _media_eval_prompt(script, assets))
    data = extract_json(text)
    return Evaluation(
        score=coerce_score(data.get("score")),
        feedback=(data.get("feedback") or "").strip(),
    )


def suggest_keywords_from_feedback(
    feedback: str,
    title: str,
    script: str,
    model: Any | None = None,
) -> list[str]:
    """Na osnovu eval feedback-a generisi nove keywords za retry fetch.

    Feedback se ubacuje kao dodatni kontekst u keyword prompt.
    """
    if model is None:
        model = build_model()
    prompt = (
        _keywords_prompt(title, script)
        + f"\nPrevious visuals were rated poor. Editor feedback: {feedback}\n"
        "Return DIFFERENT, more relevant terms than before."
    )
    text = generate_text(model, prompt)
    data = extract_json(text)
    keywords = data.get("keywords") or []
    return [str(k).strip() for k in keywords if str(k).strip()]
