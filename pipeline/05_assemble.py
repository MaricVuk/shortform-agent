"""Korak 5 — Montaza finalnog MP4 (MoviePy).

Spaja preuzete slike u vertikalni (9:16) slideshow sinhronizovan sa duzinom
naracije, dodaje audio i **sinhronizovane titlove po recima** (iz koraka 3),
i renderuje MP4.

Dizajn:
- `plan_slides(n, audio_duration)` je CISTA funkcija (raspodela trajanja po
  slikama) — potpuno testabilna bez ffmpeg-a.
- `group_captions(...)` i `_active_chunk(...)` su takodje ciste funkcije —
  grupisu pojedinacne reci (iz edge-tts word-boundary timestamp-a) u citljive
  "chunk"-ove i nalaze koji je aktivan u datom trenutku.
- `assemble_video(...)` prima injektabilni `renderer`, pa se validacija i
  raspodela testiraju bez pravog renderovanja. Podrazumevani `_moviepy_render`
  radi pravu montazu (verifikuje se u end-to-end testu).

FFmpeg dolazi preko `imageio-ffmpeg` (bundlovan), nije potreban sistemski.
Titlovi se crtaju preko PIL-a (bez ImageMagick zavisnosti) kao providan
overlay preko cele montaze — vidljivi samo dok se odgovarajuce reci izgovaraju
(za razliku od statickog naslova preko celog videa).
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


def group_captions(captions: list[Any], chunk_size: int | None = None) -> list[tuple[str, float, float]]:
    """Grupisi reci (sa .text/.start/.end) u citljive chunk-ove za prikaz.

    Prikazivanje reci jednu po jednu je previse "trzavo"; grupisanje po
    `chunk_size` reci (podrazumevano config.CAPTION_CHUNK_SIZE) daje kratke
    fraze sinhronizovane sa govorom, kao u TikTok/CapCut stilu titlova.

    Vraca listu (text, start, end) — obicni tuple-ovi, ne zavisi od
    konkretnog Caption tipa (duck typing, isti obrazac kao ostatak pipeline-a).
    """
    chunk_size = chunk_size or config.CAPTION_CHUNK_SIZE
    groups: list[tuple[str, float, float]] = []
    for i in range(0, len(captions), chunk_size):
        chunk = captions[i : i + chunk_size]
        if not chunk:
            continue
        text = " ".join(c.text for c in chunk)
        groups.append((text, chunk[0].start, chunk[-1].end))
    return groups


def _active_chunk(chunks: list[tuple[str, float, float]], t: float) -> str | None:
    """Nadji tekst chunk-a aktivnog u trenutku `t` (sekunde), ili None."""
    for text, start, end in chunks:
        if start <= t < end:
            return text
    return None


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


def _prepare_frame(src_path: str, size: tuple[int, int]) -> "Any":
    """Ucitaj sliku i cover-fit na ciljni format. Vraca numpy RGB array."""
    import numpy as np
    from PIL import Image

    img = Image.open(src_path).convert("RGB")
    img = _cover_fit(img, size)
    return np.array(img)


_CAPTION_BAND_HEIGHT = 220
_CAPTION_FONT_SIZE = 68


def _load_caption_font(size: int = _CAPTION_FONT_SIZE) -> "Any":
    from PIL import ImageFont

    try:
        return ImageFont.truetype("arialbd.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _render_caption_band(width: int, text: str, font: "Any") -> "Any":
    """Nacrtaj jedan caption chunk na providnoj traci sirine `width`."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (width, _CAPTION_BAND_HEIGHT), (18, 18, 18))
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = max(0, (width - text_w) // 2)
    y = max(0, (_CAPTION_BAND_HEIGHT - text_h) // 2)
    draw.text((x, y), text, font=font, fill=(255, 224, 90))
    return img


def _caption_overlay_clip(
    chunks: list[tuple[str, float, float]], size: tuple[int, int], duration: float
) -> "Any":
    """Napravi providan overlay-clip koji prikazuje aktivan caption chunk.

    Providnost (mask) je TAKODJE vremenski promenljiva: 0 kad nijedan chunk
    nije aktivan (izmedju reci/fraza), 1 kad jeste — titlovi se pale/gase
    tacno u ritmu govora, ne stoje stalno na ekranu.
    """
    import numpy as np
    from moviepy.editor import VideoClip

    tw, _th = size
    font = _load_caption_font()
    blank = np.array(_render_caption_band(tw, "", font))
    rendered_cache: dict[str, "np.ndarray"] = {}

    def make_frame(t: float) -> "np.ndarray":
        text = _active_chunk(chunks, t)
        if text is None:
            return blank
        if text not in rendered_cache:
            rendered_cache[text] = np.array(_render_caption_band(tw, text, font))
        return rendered_cache[text]

    def make_mask(t: float) -> "np.ndarray":
        opacity = 0.72 if _active_chunk(chunks, t) is not None else 0.0
        return np.full((_CAPTION_BAND_HEIGHT, tw), opacity)

    clip = VideoClip(make_frame, duration=duration)
    mask_clip = VideoClip(make_mask, duration=duration, ismask=True)
    return clip.set_mask(mask_clip).set_position(("center", "bottom"))


def _moviepy_render(
    audio_path: str,
    image_paths: list[str],
    durations: list[float],
    out_path: Path,
    captions: list[Any],
    size: tuple[int, int],
    fps: int,
) -> None:
    """Podrazumevani renderer: slideshow (cover-fit) + audio + sync titlovi -> MP4.

    Frame-ovi slika se pripremaju u PIL-u pa se prosledjuju kao numpy array;
    MoviePy sluzi za concat, caption compositing, audio mux i ffmpeg zapis
    (izbegnut krhki MoviePy resize fx, vidi `_cover_fit`).
    """
    from moviepy.editor import AudioFileClip, CompositeVideoClip, ImageClip, concatenate_videoclips

    clips = [
        ImageClip(_prepare_frame(path, size)).set_duration(dur)
        for path, dur in zip(image_paths, durations)
    ]

    video = concatenate_videoclips(clips, method="chain")
    audio = AudioFileClip(audio_path)
    video = video.set_audio(audio).set_duration(audio.duration)

    chunks = group_captions(captions) if captions else []
    if chunks:
        caption_clip = _caption_overlay_clip(chunks, size, video.duration)
        video = CompositeVideoClip([video, caption_clip], size=size)

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
    captions: list[Any] | None = None,
    *,
    renderer: Callable[..., Any] | None = None,
) -> str:
    """Sastavi finalni MP4 i vrati putanju do njega.

    Args:
        audio_path: putanja do naracije (iz koraka 3).
        audio_duration: trajanje naracije u sekundama (za sync).
        image_paths: putanje do preuzetih slika (iz koraka 4).
        out_path: gde snimiti finalni mp4.
        captions: lista objekata sa .text/.start/.end (iz koraka 3) za
            sinhronizovane titlove; None/prazna lista -> bez titlova.
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
        captions or [],
        (config.VIDEO_WIDTH, config.VIDEO_HEIGHT),
        config.VIDEO_FPS,
    )

    if not out_path.exists():
        raise RuntimeError(f"Montaza nije napravila fajl na {out_path}")
    return str(out_path)
