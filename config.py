"""Centralna konfiguracija: putanje, konstante, pragovi za agent self-eval.

Sve tajne se citaju iz .env (nikad hardkodovane). Ovaj fajl drzi samo
ne-tajne parametre koje je korisno menjati na jednom mestu.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Ucitaj .env iz korena projekta (ako postoji).
load_dotenv()

# --- Putanje ---
ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "output"

# --- API kljucevi (iz okruzenja; None ako nisu postavljeni) ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")

# --- LLM (Groq — OpenAI-kompatibilan API, velikodusan free-tier rate limit) ---
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# --- TTS ---
# Microsoft edge-tts glas. Lista glasova: `edge-tts --list-voices`.
TTS_VOICE = os.getenv("TTS_VOICE", "en-US-AriaNeural")

# --- Agent self-eval pragovi ---
# Korak 2 (skripta): LLM ocenjuje sopstveni draft 1-10; ispod praga -> regenerisi.
SCRIPT_EVAL_THRESHOLD = int(os.getenv("SCRIPT_EVAL_THRESHOLD", "7"))
SCRIPT_MAX_ATTEMPTS = int(os.getenv("SCRIPT_MAX_ATTEMPTS", "3"))

# Korak 4 (vizuali): LLM ocenjuje relevantnost preuzetih vizuala; ispod praga
# -> nove keywords i ponovni Pexels poziv.
MEDIA_EVAL_THRESHOLD = int(os.getenv("MEDIA_EVAL_THRESHOLD", "7"))
MEDIA_MAX_ATTEMPTS = int(os.getenv("MEDIA_MAX_ATTEMPTS", "3"))

# --- Media ---
PEXELS_CLIP_COUNT = int(os.getenv("PEXELS_CLIP_COUNT", "5"))  # broj vizuala po videu

# --- Video (short-form: 9:16 vertikalni) ---
VIDEO_WIDTH = 1080
VIDEO_HEIGHT = 1920
VIDEO_FPS = 30

# --- Scheduling (mock) ---
DEFAULT_PLATFORMS = ["youtube_shorts", "tiktok", "instagram_reels"]


def require(key_name: str) -> str:
    """Vrati vrednost obaveznog API kljuca ili baci jasnu gresku.

    Koristi se u modulima koji zovu prave API-je, da greska bude citljiva
    umesto tihe None-propagacije.
    """
    value = os.getenv(key_name)
    if not value:
        raise RuntimeError(
            f"Nedostaje {key_name}. Kopiraj .env.example u .env i popuni kljuc."
        )
    return value
