"""Korak 3 — Text-to-Speech (edge-tts).

Pretvara finalni tekst skripte u audio fajl, vraca trajanje (za sync sa
montazom), i **sinhronizovane titlove po recima** (word-level timestamps) —
edge-tts prilikom streaminga saljе WordBoundary dogadjaje koje hvatamo preko
`edge_tts.SubMaker`. Ovo omogucava korak 5 (montaza) da upali titlove tacno
kad se svaka rec izgovori, kao u CapCut/TikTok stilu, umesto statickog
naslova preko celog videa.

edge-tts je async i besplatan (bez API kljuca). Trajanje se cita iz snimljenog
fajla preko MoviePy/ffmpeg-a.

Dizajn za testiranje: `synthesize` prima injektabilne `tts_runner` i
`duration_probe`, pa test moze da napravi dummy fajl i fiksno trajanje/titlove
bez mreznog poziva i bez ffmpeg-a.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import config


@dataclass
class Caption:
    """Jedna rec sa tacnim vremenom izgovora (sekunde, od pocetka audija)."""

    text: str
    start: float
    end: float


@dataclass
class TTSResult:
    """Rezultat sinteze govora."""

    audio_path: str
    duration: float  # sekunde
    captions: list[Caption] = field(default_factory=list)


def _edge_tts_runner(text: str, voice: str, out_path: Path) -> list[Caption]:
    """Podrazumevani runner: pozovi edge-tts, snimi mp3, vrati word-level titlove."""
    import edge_tts

    async def _run() -> list[Caption]:
        # boundary="WordBoundary" je obavezan -- podrazumevano edge-tts salje
        # samo SentenceBoundary, sto ne daje dovoljnu granularnost za
        # sinhronizovane titlove po recima.
        communicate = edge_tts.Communicate(text, voice, boundary="WordBoundary")
        submaker = edge_tts.SubMaker()
        with open(out_path, "wb") as audio_file:
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_file.write(chunk["data"])
                elif chunk["type"] == "WordBoundary":
                    submaker.feed(chunk)
        return [
            Caption(
                text=cue.content,
                start=cue.start.total_seconds(),
                end=cue.end.total_seconds(),
            )
            for cue in submaker.cues
        ]

    return asyncio.run(_run())


def _probe_duration(out_path: Path) -> float:
    """Podrazumevani probe: procitaj trajanje audija preko MoviePy/ffmpeg."""
    from moviepy.editor import AudioFileClip

    clip = AudioFileClip(str(out_path))
    try:
        return float(clip.duration)
    finally:
        clip.close()


def synthesize(
    text: str,
    out_path: str | Path,
    voice: str | None = None,
    *,
    tts_runner: Callable[[str, str, Path], Any] | None = None,
    duration_probe: Callable[[Path], float] | None = None,
) -> TTSResult:
    """Sintetizuj `text` u audio fajl na `out_path` i vrati putanju + trajanje + titlove.

    Args:
        text: tekst za izgovor (telo skripte).
        out_path: gde snimiti audio (npr. output/<run_id>/audio.mp3).
        voice: edge-tts glas; ako None, uzima se config.TTS_VOICE.
        tts_runner: injektabilna funkcija sinteze koja vraca listu `Caption`
            (za testove; ako vrati None, titlovi su prazna lista).
        duration_probe: injektabilna funkcija za citanje trajanja (za testove).

    Raises:
        ValueError: ako je tekst prazan.
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("Tekst za TTS ne sme biti prazan.")

    voice = voice or config.TTS_VOICE
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    runner = tts_runner or _edge_tts_runner
    probe = duration_probe or _probe_duration

    captions = runner(text, voice, out_path) or []

    if not out_path.exists():
        raise RuntimeError(f"TTS nije napravio fajl na {out_path}")

    duration = float(probe(out_path))
    return TTSResult(audio_path=str(out_path), duration=duration, captions=list(captions))


if __name__ == "__main__":  # rucni smoke test (pravi edge-tts poziv + ffmpeg)
    import sys

    sample = sys.argv[1] if len(sys.argv) > 1 else "Hello from the shortform agent."
    result = synthesize(sample, config.OUTPUT_DIR / "_smoke" / "audio.mp3")
    print(f"Audio: {result.audio_path}  ({result.duration:.2f}s)")
    print(f"Captions: {len(result.captions)} reci")
