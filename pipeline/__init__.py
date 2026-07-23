"""Pipeline paket.

Moduli su namerno numerisani (`01_research.py` ... `06_schedule.py`) radi
citljivosti redosleda, ali imena koja pocinju cifrom nisu validni Python
identifikatori pa se ne mogu importovati standardnim `import`-om.

Ovaj `__init__` ih ucitava lenjo (PEP 562 `__getattr__`) preko `importlib`
i izlaze pod cistim imenima, tako da ostatak koda (graf, testovi) radi
jednostavno i da svaki modul moze da se testira izolovano (ucitava se samo
onaj koji se zatrazi):

    from pipeline import research, script, tts, media, assemble, schedule
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

_PKG_DIR = Path(__file__).resolve().parent

# mapiranje: cisto ime -> ime fajla (bez .py)
_MODULES = {
    "research": "01_research",
    "script": "02_script",
    "tts": "03_tts",
    "media": "04_media",
    "assemble": "05_assemble",
    "schedule": "06_schedule",
}

_cache: dict[str, ModuleType] = {}


def _load(clean_name: str) -> ModuleType:
    """Ucitaj numerisani modul iz fajla i registruj ga pod cistim imenom."""
    if clean_name in _cache:
        return _cache[clean_name]
    file_stem = _MODULES[clean_name]
    path = _PKG_DIR / f"{file_stem}.py"
    qualified = f"{__name__}.{clean_name}"
    spec = importlib.util.spec_from_file_location(qualified, path)
    if spec is None or spec.loader is None:  # pragma: no cover - odbrana
        raise ImportError(f"Ne mogu da ucitam {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified] = module
    spec.loader.exec_module(module)
    _cache[clean_name] = module
    return module


def __getattr__(name: str) -> ModuleType:
    """PEP 562: lenjo ucitavanje modula na prvi pristup `pipeline.<name>`."""
    if name in _MODULES:
        return _load(name)
    raise AttributeError(f"module {__name__!r} nema atribut {name!r}")


def __dir__() -> list[str]:
    return sorted(_MODULES)


__all__ = list(_MODULES)
