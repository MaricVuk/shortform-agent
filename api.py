"""FastAPI webhook sloj — ulaz koji okida pipeline.

Endpoints:
  POST /generate           -> {topic} pokrece pipeline asinhrono, vraca run_id
  GET  /status/{run_id}    -> status run-a (processing / ready_to_publish / failed)
  GET  /dashboard/{run_id} -> vizuelni live prikaz napretka (poll-uje /status)
  GET  /files/...          -> staticki serviran output/ folder (video, slike, metadata)
  GET  /                   -> health/info
  GET  /docs               -> Swagger UI (automatski od FastAPI-ja)

Ovo je "webhook" ulaz: eksterni alat (n8n, Zapier, Make, ili obican HTTP
poziv) posalje temu i kasnije poll-uje status. Pipeline se izvrsava u pozadini
(FastAPI BackgroundTasks -> starlette threadpool), pa POST odmah vraca odgovor.

Status se cuva u output/<run_id>/status.json (preziveljava proces i moze da se
inspektuje) uz brzi in-memory kes. Pipeline se pokrece preko
`run_pipeline_streaming` — svaki LangGraph cvor odmah azurira status, sto
dashboard koristi za live prikaz napretka kroz faze.
"""
from __future__ import annotations

import json
import re
import traceback
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import config
import dashboard as dashboard_module
from orchestration.graph import run_pipeline_streaming

app = FastAPI(
    title="Shortform Agent",
    description="Agentic pipeline: tema -> gotov short-form video (mock objava).",
    version="1.0.0",
)

config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/files", StaticFiles(directory=str(config.OUTPUT_DIR)), name="files")

# Brzi in-memory kes statusa (izvor istine je status.json na disku).
_status_cache: dict[str, dict] = {}

# run_id zavrsava u putanjama na disku (output/<run_id>/...) i u dashboard
# HTML/JS-u, pa je striktna validacija obavezna: sprecava path traversal
# (npr. run_id="../../x") i XSS u dashboard stranici.
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _validate_run_id(run_id: str) -> str:
    if not _RUN_ID_RE.match(run_id):
        raise HTTPException(
            status_code=400,
            detail="run_id sme sadrzati samo slova, brojeve, '-' i '_' (max 64 znaka).",
        )
    return run_id


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


def _on_step(run_id: str, topic: str):
    """Napravi callback koji posle svakog LangGraph cvora azurira status.json.

    Ovo je izvor "live" napretka za /dashboard — svaka self-eval petlja
    (skripta, vizuali) odmah pise attempt/score, ne ceka se kraj pipeline-a.
    """

    def _callback(node_name: str, state: dict) -> None:
        _write_status(
            run_id,
            "processing",
            topic=topic,
            current_node=node_name,
            script_attempts=state.get("script_attempts", 0),
            script_score=state.get("script_score"),
            media_attempts=state.get("media_attempts", 0),
            media_score=state.get("media_score"),
        )

    return _callback


def _run_and_track(topic: str, run_id: str) -> None:
    """Pozadinski task: pokreni pipeline (streaming) i azuriraj status (uspeh/greska)."""
    _write_status(run_id, "processing", topic=topic, current_node=None)
    try:
        state = run_pipeline_streaming(topic, run_id, on_step=_on_step(run_id, topic))
        metadata = state.get("metadata", {})
        _write_status(
            run_id,
            "ready_to_publish",
            topic=topic,
            video_path=state.get("video_path"),
            metadata=metadata,
            script_attempts=state.get("script_attempts", 0),
            media_attempts=state.get("media_attempts", 0),
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
        "dashboard": "GET /dashboard/{run_id} za vizuelni live prikaz",
        "docs": "GET /docs za interaktivni Swagger UI",
    }


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest, background: BackgroundTasks) -> GenerateResponse:
    """Okini pipeline za datu temu; vrati run_id i pocetni status."""
    run_id = _validate_run_id((req.run_id or uuid.uuid4().hex[:12]).strip())
    _write_status(run_id, "queued", topic=req.topic)
    background.add_task(_run_and_track, req.topic, run_id)
    return GenerateResponse(run_id=run_id, status="queued")


@app.get("/status/{run_id}")
def status(run_id: str) -> dict:
    """Vrati trenutni status run-a."""
    run_id = _validate_run_id(run_id)
    data = _read_status(run_id)
    if data is None:
        raise HTTPException(status_code=404, detail=f"Nepoznat run_id: {run_id}")
    return data


@app.get("/dashboard/{run_id}", response_class=HTMLResponse)
def dashboard(run_id: str) -> str:
    """Vizuelni live prikaz napretka jednog run-a (poll-uje /status)."""
    run_id = _validate_run_id(run_id)
    return dashboard_module.render(run_id)


if __name__ == "__main__":  # `python api.py` kao alternativa uvicorn CLI-ju
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
