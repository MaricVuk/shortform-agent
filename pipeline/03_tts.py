"""Korak 3 — Text-to-Speech (edge-tts).

Pretvara finalni tekst skripte u audio fajl i vraca trajanje (potrebno da
montaza sinhronizuje vizuale sa naracijom).

edge-tts je async i besplatan (bez API kljuca). Trajanje se cita iz snimljenog
fajla preko MoviePy/ffmpeg-a.

Dizajn za testiranje: `synthesize` prima injektabilne `tts_runner` i
`duration_probe`, pa test moze da napravi dummy fajl i fiksno trajanje bez
mreznog poziva i bez ffmpeg-a.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import config


@dataclass
class TTSResult:
    """Rezultat sinteze govora."""

    audio_path: str
    duration: float  # sekunde


def _edge_tts_runner(text: str, voice: str, out_path: Path) -> None:
    """Podrazumevani runner: pozovi edge-tts i snimi mp3 na `out_path`."""
    import edge_tts

    async def _run() -> None:
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(str(out_path))

    asyncio.run(_run())


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
    """Sintetizuj `text` u audio fajl na `out_path` i vrati putanju + trajanje.

    Args:
        text: tekst za izgovor (telo skripte).
        out_path: gde snimiti audio (npr. output/<run_id>/audio.mp3).
        voice: edge-tts glas; ako None, uzima se config.TTS_VOICE.
        tts_runner: injektabilna funkcija sinteze (za testove).
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

    runner(text, voice, out_path)

    if not out_path.exists():
        raise RuntimeError(f"TTS nije napravio fajl na {out_path}")

    duration = float(probe(out_path))
    return TTSResult(audio_path=str(out_path), duration=duration)


if __name__ == "__main__":  # rucni smoke test (pravi edge-tts poziv + ffmpeg)
    import sys

    sample = sys.argv[1] if len(sys.argv) > 1 else "Hello from the shortform agent."
    result = synthesize(sample, config.OUTPUT_DIR / "_smoke" / "audio.mp3")
    print(f"Audio: {result.audio_path}  ({result.duration:.2f}s)")
