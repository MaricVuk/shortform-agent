"""Korak 5 — Montaza finalnog MP4 (MoviePy).

Spaja preuzete slike u vertikalni (9:16) slideshow sinhronizovan sa duzinom
naracije, dodaje audio i opcioni naslov-overlay, i renderuje MP4.

Dizajn:
- `plan_slides(n, audio_duration)` je CISTA funkcija (raspodela trajanja po
  slikama) — potpuno testabilna bez ffmpeg-a.
- `assemble_video(...)` prima injektabilni `renderer`, pa se validacija i
  raspodela testiraju bez pravog renderovanja. Podrazumevani `_moviepy_render`
  radi pravu montazu (verifikuje se u end-to-end testu).

FFmpeg dolazi preko `imageio-ffmpeg` (bundlovan), nije potreban sistemski.
Title overlay se crta preko PIL-a (bez ImageMagick zavisnosti) i fail-safe je.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import config


def plan_slides(num_assets: int, audio_duration: float) -> list[float]:
    """Rasporedi `audio_duration` sekundi ravnomerno na `num_assets` slika.

    Poslednja slika dobija ostatak (da suma tacno bude audio_duration).
    """
    if num_assets <= 0:
        raise ValueError("Nema slika za montazu (num_assets <= 0).")
    if audio_duration <= 0:
        raise ValueError("audio_duration mora biti > 0.")

    per = audio_duration / num_assets
    durations = [round(per, 3) for _ in range(num_assets)]
    # koriguj poslednju da suma bude tacna (izbegni drift zaokruzivanja)
    durations[-1] = round(audio_duration - sum(durations[:-1]), 3)
    return durations


def _cover_fit(img: "Any", size: tuple[int, int]) -> "Any":
    """Skaliraj sliku da PREKRIJE ciljni format, pa centralno iseci na `size`.

    Radi se preko PIL-a (LANCZOS), ne preko MoviePy resize fx-a — MoviePy 1.0.3
    zove uklonjeni `Image.ANTIALIAS` pa puca na Pillow 10+.
    """
    from PIL import Image

    tw, th = size
    w, h = img.size
    scale = max(tw / w, th / h)
    resized = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    rw, rh = resized.size
    left, top = (rw - tw) // 2, (rh - th) // 2
    return resized.crop((left, top, left + tw, top + th))


def _draw_caption(img: "Any", text: str) -> None:
    """Nacrtaj naslov (poluprovidna traka + tekst) na dnu slike, in-place."""
    from PIL import ImageDraw, ImageFont

    tw, th = img.size
    pad, band_h = 48, 260
    draw = ImageDraw.Draw(img, "RGBA")
    try:
        font = ImageFont.truetype("arialbd.ttf", 66)
    except OSError:
        font = ImageFont.load_default()

    # wrap na sirinu (max 3 linije)
    words, lines, line = text.split(), [], ""
    for word in words:
        trial = f"{line} {word}".strip()
        if draw.textlength(trial, font=font) <= tw - 2 * pad:
            line = trial
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    block = "\n".join(lines[:3])

    y0 = th - band_h
    draw.rectangle([0, y0, tw, th], fill=(0, 0, 0, 150))
    draw.multiline_text(
        (pad, y0 + pad), block, font=font, fill=(255, 255, 255, 255), spacing=12
    )


def _prepare_frame(src_path: str, size: tuple[int, int], title: str | None) -> "Any":
    """Ucitaj sliku, cover-fit na ciljni format i (opciono) upeci naslov.

    Vraca numpy RGB array spreman za MoviePy ImageClip.
    """
    import numpy as np
    from PIL import Image

    img = Image.open(src_path).convert("RGB")
    img = _cover_fit(img, size)
    if title:
        _draw_caption(img, title)
    return np.array(img)


def _moviepy_render(
    audio_path: str,
    image_paths: list[str],
    durations: list[float],
    out_path: Path,
    title: str | None,
    size: tuple[int, int],
    fps: int,
) -> None:
    """Podrazumevani renderer: slideshow (cover-fit + naslov) + audio -> MP4.

    Frame-ovi se pripremaju u PIL-u pa se prosledjuju kao numpy array; MoviePy
    sluzi samo za concat, audio mux i ffmpeg zapis (izbegnut krhki resize fx).
    """
    from moviepy.editor import AudioFileClip, ImageClip, concatenate_videoclips

    clips = [
        ImageClip(_prepare_frame(path, size, title)).set_duration(dur)
        for path, dur in zip(image_paths, durations)
    ]

    video = concatenate_videoclips(clips, method="chain")
    audio = AudioFileClip(audio_path)
    video = video.set_audio(audio).set_duration(audio.duration)

    video.write_videofile(
        str(out_path),
        fps=fps,
        codec="libx264",
        audio_codec="aac",
        logger=None,
    )
    audio.close()
    video.close()
    for clip in clips:
        clip.close()


def assemble_video(
    audio_path: str | Path,
    audio_duration: float,
    image_paths: list[str],
    out_path: str | Path,
    title: str | None = None,
    *,
    renderer: Callable[..., Any] | None = None,
) -> str:
    """Sastavi finalni MP4 i vrati putanju do njega.

    Args:
        audio_path: putanja do naracije (iz koraka 3).
        audio_duration: trajanje naracije u sekundama (za sync).
        image_paths: putanje do preuzetih slika (iz koraka 4).
        out_path: gde snimiti finalni mp4.
        title: opcioni naslov za overlay.
        renderer: injektabilni renderer (za testove).

    Raises:
        ValueError: ako nema slika ili je trajanje nevalidno.
    """
    if not image_paths:
        raise ValueError("Nema slika za montazu.")

    durations = plan_slides(len(image_paths), audio_duration)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    render = renderer or _moviepy_render
    render(
        str(audio_path),
        list(image_paths),
        durations,
        out_path,
        title,
        (config.VIDEO_WIDTH, config.VIDEO_HEIGHT),
        config.VIDEO_FPS,
    )

    if not out_path.exists():
        raise RuntimeError(f"Montaza nije napravila fajl na {out_path}")
    return str(out_path)
