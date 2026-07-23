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


_KEN_BURNS_ZOOM = 1.15  # koliko je "platno" vece od ciljnog formata (prostor za pan)
_TRANSITION_DURATION = 0.5  # sekunde, crossfade izmedju slajdova

# Cetiri pravca panovanja (start_x, start_y, end_x, end_y kao udeo 0..1
# dostupnog prostora za pomeranje) — ciklicno se biraju po indeksu slajda,
# deterministicki (ne random) radi ponovljivosti i testabilnosti.
_PAN_PATHS: list[tuple[float, float, float, float]] = [
    (0.0, 0.0, 1.0, 1.0),  # dijagonalno gore-levo -> dole-desno
    (1.0, 0.0, 0.0, 1.0),  # dijagonalno gore-desno -> dole-levo
    (0.5, 0.0, 0.5, 1.0),  # vertikalno, centrirano
    (0.0, 0.5, 1.0, 0.5),  # horizontalno, centrirano
]


def _pan_window(oversized: "Any", size: tuple[int, int], progress: float, path: tuple[float, float, float, float]) -> "Any":
    """Iseci prozor velicine `size` iz `oversized` slike, pomeren po `path`."""
    ow, oh = oversized.size
    tw, th = size
    max_dx, max_dy = max(0, ow - tw), max(0, oh - th)
    x0, y0, x1, y1 = path
    x = round((x0 + (x1 - x0) * progress) * max_dx)
    y = round((y0 + (y1 - y0) * progress) * max_dy)
    return oversized.crop((x, y, x + tw, y + th))


def _ken_burns_clip(src_path: str, size: tuple[int, int], duration: float, path_index: int) -> "Any":
    """Blagi zoom+pan efekat: slika je unapred uvecana (`_KEN_BURNS_ZOOM`), pa
    se svaki frame samo isece (jeftino) umesto da se skalira (skupo) —
    performantno cak i na desetinama frejmova po slajdu.
    """
    import numpy as np
    from moviepy.editor import VideoClip
    from PIL import Image

    img = Image.open(src_path).convert("RGB")
    big_size = (round(size[0] * _KEN_BURNS_ZOOM), round(size[1] * _KEN_BURNS_ZOOM))
    oversized = _cover_fit(img, big_size)
    path = _PAN_PATHS[path_index % len(_PAN_PATHS)]

    def make_frame(t: float) -> "np.ndarray":
        progress = 0.0 if duration <= 0 else min(1.0, max(0.0, t / duration))
        return np.array(_pan_window(oversized, size, progress, path))

    return VideoClip(make_frame, duration=duration)


def _slideshow_with_crossfade(
    image_paths: list[str], durations: list[float], size: tuple[int, int], transition: float
) -> "Any":
    """Spoji Ken Burns slajdove sa crossfade tranzicijom, BEZ MoviePy
    `concatenate_videoclips(..., method="compose")` — ta putanja gradi jedan
    CompositeVideoClip preko svih slajdova i evaluira ga za SVAKI frejm cele
    montaze (izmereno ~8x sporije po frejmu na 1080x1920 nego direktan poziv).

    Umesto toga: direktan lookup segmenta po vremenu (koji slajd je aktivan),
    i numpy alpha-blend RUCNO samo unutar uskih zona preklapanja (`transition`
    sekundi oko svake granice) — van tih zona je to jedan jeftin direktan poziv.
    """
    import numpy as np
    from moviepy.editor import VideoClip

    n = len(image_paths)
    has_transition = transition > 0 and n > 1
    render_durations = [d + transition for d in durations] if has_transition else list(durations)
    clips = [
        _ken_burns_clip(path, size, dur, i)
        for i, (path, dur) in enumerate(zip(image_paths, render_durations))
    ]

    starts = [0.0]
    for dur in render_durations[:-1]:
        step = dur - transition if has_transition else dur
        starts.append(starts[-1] + step)
    total_duration = starts[-1] + render_durations[-1]

    def make_frame(t: float) -> "np.ndarray":
        if has_transition:
            for i in range(n - 1):
                overlap_start = starts[i + 1]
                overlap_end = starts[i] + render_durations[i]
                if overlap_start <= t < overlap_end:
                    local_out = t - starts[i]
                    local_in = t - starts[i + 1]
                    alpha = local_in / transition
                    frame_out = clips[i].get_frame(local_out).astype(np.float32)
                    frame_in = clips[i + 1].get_frame(local_in).astype(np.float32)
                    return ((1 - alpha) * frame_out + alpha * frame_in).astype(np.uint8)
        for i in range(n - 1, -1, -1):
            if t >= starts[i]:
                local_t = min(t - starts[i], render_durations[i] - 1e-6)
                return clips[i].get_frame(local_t)
        return clips[0].get_frame(0.0)

    return VideoClip(make_frame, duration=total_duration)


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


_CAPTION_OPACITY = 0.72


def _apply_captions(video_clip: "Any", chunks: list[tuple[str, float, float]], size: tuple[int, int]) -> "Any":
    """Upeci sinhronizovane titlove direktno u frejmove `video_clip`.

    Namerno NE koristi MoviePy `CompositeVideoClip` (ista razlika u brzini kao
    kod `_slideshow_with_crossfade`) — svaki frejm se ucita iz osnovnog klipa,
    pa se (ako je caption aktivan) traka na dnu alpha-blenduje numpy-jem
    direktno u taj frejm. Van aktivnih trenutaka je ovo samo `get_frame`
    passthrough, skoro besplatno.
    """
    import numpy as np
    from moviepy.editor import VideoClip

    tw, th = size
    font = _load_caption_font()
    band_cache: dict[str, "np.ndarray"] = {}
    band_y0 = th - _CAPTION_BAND_HEIGHT

    def make_frame(t: float) -> "np.ndarray":
        frame = video_clip.get_frame(t)
        text = _active_chunk(chunks, t)
        if text is None:
            return frame
        if text not in band_cache:
            band_cache[text] = np.array(_render_caption_band(tw, text, font), dtype=np.float32)
        band = band_cache[text]
        base = frame[band_y0:th, 0:tw].astype(np.float32)
        blended = (1 - _CAPTION_OPACITY) * base + _CAPTION_OPACITY * band
        frame = frame.copy()
        frame[band_y0:th, 0:tw] = blended.astype(np.uint8)
        return frame

    return VideoClip(make_frame, duration=video_clip.duration)


def _moviepy_render(
    audio_path: str,
    image_paths: list[str],
    durations: list[float],
    out_path: Path,
    captions: list[Any],
    size: tuple[int, int],
    fps: int,
) -> None:
    """Podrazumevani renderer: Ken Burns slideshow + crossfade + audio + sync
    titlovi -> MP4.

    Svaki slajd je blago zumiran/pomeren (`_ken_burns_clip`); susedni slajdovi
    se preklapaju kroz crossfade (`_slideshow_with_crossfade`), a titlovi se
    upisu preko `_apply_captions` — obe funkcije namerno zaobilaze MoviePy
    `CompositeVideoClip`/`method="compose"`, koji je na 1080x1920 izmereno
    ~8x sporiji po frejmu od direktnog pristupa (vidi docstring-ove).
    """
    from moviepy.editor import AudioFileClip

    transition = _TRANSITION_DURATION if len(image_paths) > 1 else 0.0
    video = _slideshow_with_crossfade(image_paths, durations, size, transition)

    chunks = group_captions(captions) if captions else []
    if chunks:
        video = _apply_captions(video, chunks, size)

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
