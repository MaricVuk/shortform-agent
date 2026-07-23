"""Korak 6 — Mock scheduled objava.

NE zove prave upload API-je (TikTok/YouTube/IG zahtevaju app review — van
scope-a MVP-a). Umesto toga priprema `metadata.json` sa svim sto bi pravi
scheduler dobio: putanja do videa, platforme, predlozeno vreme objave, status.

`now` je injektabilan da predlog vremena bude deterministican u testu.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import config


def _suggest_publish_time(now: datetime) -> datetime:
    """Predlozi sledeci termin objave: sutradan u 18:00 (dobar engagement slot)."""
    tomorrow = (now + timedelta(days=1)).date()
    return datetime(tomorrow.year, tomorrow.month, tomorrow.day, 18, 0, 0)


def prepare_schedule(
    run_id: str,
    title: str,
    video_path: str | Path,
    out_dir: str | Path,
    platforms: list[str] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Napravi i snimi `metadata.json`; vrati metadata dict.

    Args:
        run_id: id ovog run-a (koristi se i kao ime foldera).
        title: naslov videa.
        video_path: putanja do finalnog MP4.
        out_dir: folder run-a (gde se pise metadata.json).
        platforms: ciljne platforme; ako None -> config.DEFAULT_PLATFORMS.
        now: trenutno vreme (injektabilno za testove).

    Raises:
        ValueError: ako run_id prazan.
    """
    run_id = (run_id or "").strip()
    if not run_id:
        raise ValueError("run_id ne sme biti prazan.")

    now = now or datetime.now()
    platforms = platforms or list(config.DEFAULT_PLATFORMS)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "run_id": run_id,
        "title": title,
        "video_path": str(video_path),
        "platforms": platforms,
        "suggested_publish_time": _suggest_publish_time(now).isoformat(),
        "created_at": now.isoformat(),
        "status": "ready_to_publish",  # mock — nista se zaista ne kaci
        "note": "Mock scheduling: pravi upload API-ji nisu pozvani (zahtevaju app review).",
    }

    metadata_path = out_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metadata["metadata_path"] = str(metadata_path)
    return metadata
