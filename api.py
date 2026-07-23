"""FastAPI webhook sloj — ulaz koji okida pipeline.

Endpoints:
  POST /generate        -> {topic} pokrece pipeline asinhrono, vraca run_id
  GET  /status/{run_id} -> status run-a (processing / ready_to_publish / failed)
  GET  /                -> health/info

Ovo je "webhook" ulaz: eksterni alat (n8n, Zapier, Make, ili obican HTTP
poziv) posalje temu i kasnije poll-uje status. Pipeline se izvrsava u pozadini
(FastAPI BackgroundTasks -> starlette threadpool), pa POST odmah vraca odgovor.

Status se cuva u output/<run_id>/status.json (preziveljava proces i moze da se
inspektuje) uz brzi in-memory kes.
"""
from __future__ import annotations

import json
import traceback
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field

import config
from orchestration.graph import run_pipeline

app = FastAPI(
    title="Shortform Agent",
    description="Agentic pipeline: tema -> gotov short-form video (mock objava).",
    version="1.0.0",
)

# Brzi in-memory kes statusa (izvor istine je status.json na disku).
_status_cache: dict[str, dict] = {}


class GenerateRequest(BaseModel):
    topic: str = Field(..., min_length=1, description="Tema videa.")
    run_id: Optional[str] = Field(None, description="Opcioni eksplicitni run_id.")


class GenerateResponse(BaseModel):
    run_id: str
    status: str


def _status_path(run_id: str) -> Path:
    return config.OUTPUT_DIR / run_id / "status.json"


def _write_status(run_id: str, status: str, **extra: Any) -> dict:
    """Upisi status i na disk i u kes."""
    payload = {"run_id": run_id, "status": status, **extra}
    path = _status_path(run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _status_cache[run_id] = payload
    return payload


def _read_status(run_id: str) -> Optional[dict]:
    if run_id in _status_cache:
        return _status_cache[run_id]
    path = _status_path(run_id)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def _run_and_track(topic: str, run_id: str) -> None:
    """Pozadinski task: pokreni pipeline i azuriraj status (uspeh/greska)."""
    _write_status(run_id, "processing", topic=topic)
    try:
        state = run_pipeline(topic, run_id)
        metadata = state.get("metadata", {})
        _write_status(
            run_id,
            "ready_to_publish",
            topic=topic,
            video_path=state.get("video_path"),
            metadata=metadata,
        )
    except Exception as exc:  # zabelezi gresku umesto tihog pada
        _write_status(
            run_id,
            "failed",
            topic=topic,
            error=str(exc),
            traceback=traceback.format_exc(),
        )


@app.get("/")
def root() -> dict:
    return {
        "service": "Shortform Agent",
        "usage": "POST /generate {\"topic\": \"...\"} pa GET /status/{run_id}",
    }


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest, background: BackgroundTasks) -> GenerateResponse:
    """Okini pipeline za datu temu; vrati run_id i pocetni status."""
    run_id = (req.run_id or uuid.uuid4().hex[:12]).strip()
    _write_status(run_id, "queued", topic=req.topic)
    background.add_task(_run_and_track, req.topic, run_id)
    return GenerateResponse(run_id=run_id, status="queued")


@app.get("/status/{run_id}")
def status(run_id: str) -> dict:
    """Vrati trenutni status run-a."""
    data = _read_status(run_id)
    if data is None:
        raise HTTPException(status_code=404, detail=f"Nepoznat run_id: {run_id}")
    return data


if __name__ == "__main__":  # `python api.py` kao alternativa uvicorn CLI-ju
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
