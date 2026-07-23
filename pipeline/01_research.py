"""Korak 1 — Web research (Tavily).

Uzima temu i vraca strukturiranu listu referenci (naslov, snippet, url)
koje sledeci korak (skripta) koristi kao kontekst.

Dizajn za testiranje: `run_research` prima opcioni `client` parametar. U
produkciji je None pa se pravi pravi TavilyClient; u testovima se prosledi
mock, tako da nije potrebno patch-ovati importe i ne trosi se free-tier kvota.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import config


@dataclass
class Reference:
    """Jedna referenca iz web research-a."""

    title: str
    snippet: str
    url: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def _build_client() -> Any:
    """Napravi pravi Tavily klijent (lokalni import da test ne zavisi od paketa)."""
    from tavily import TavilyClient

    return TavilyClient(api_key=config.require("TAVILY_API_KEY"))


def run_research(
    topic: str,
    max_results: int = 5,
    client: Any | None = None,
) -> list[Reference]:
    """Pretrazi web za `topic` i vrati listu `Reference`.

    Args:
        topic: tema videa (input od korisnika).
        max_results: koliko referenci maksimalno vratiti.
        client: opcioni Tavily-kompatibilan klijent (za testove). Ako je None,
            pravi se pravi klijent iz `TAVILY_API_KEY`.

    Raises:
        ValueError: ako je tema prazna.
    """
    topic = (topic or "").strip()
    if not topic:
        raise ValueError("Tema (topic) ne sme biti prazna.")

    if client is None:
        client = _build_client()

    response = client.search(
        query=topic,
        max_results=max_results,
        search_depth="basic",
    )

    return _parse_results(response, max_results)


def _parse_results(response: dict[str, Any], max_results: int) -> list[Reference]:
    """Izvuci `Reference` objekte iz Tavily odgovora (defanzivno na oblik)."""
    raw_results = response.get("results", []) if isinstance(response, dict) else []
    references: list[Reference] = []
    for item in raw_results[:max_results]:
        title = (item.get("title") or "").strip()
        # Tavily vraca tekst pod 'content'; fallback na 'snippet'.
        snippet = (item.get("content") or item.get("snippet") or "").strip()
        url = (item.get("url") or "").strip()
        if not (title or snippet):
            continue  # preskoci prazne
        references.append(Reference(title=title, snippet=snippet, url=url))
    return references


if __name__ == "__main__":  # rucni smoke test (trosi pravu Tavily kvotu)
    import sys

    query = sys.argv[1] if len(sys.argv) > 1 else "3 interesting facts about space"
    for i, ref in enumerate(run_research(query), 1):
        print(f"{i}. {ref.title}\n   {ref.url}\n   {ref.snippet[:120]}...\n")
