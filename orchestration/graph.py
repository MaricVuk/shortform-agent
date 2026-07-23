"""LangGraph orkestracija celog pipeline-a.

Graf povezuje korake 1-6 i drzi DVE agentske petlje (conditional edges):

  research
     -> generate_script -> eval_script --(ocena < prag i ima pokusaja)--> generate_script
                                        --(prosao / nema vise pokusaja)--> tts
     -> tts
     -> extract_keywords -> fetch_media -> eval_media --(losi vizuali)--> refine_keywords -> fetch_media
                                                       --(ok / max)------> assemble
     -> assemble -> schedule -> END

Odluke "retry vs nastavi" su EKSPLICITNE grane u grafu (ne skrivene u modulima)
— to je razlika izmedju obicne automatizacije i pravog agenta.

`Deps` drzi sve injektabilne zavisnosti (LLM modeli, API klijenti, TTS runner,
renderer, `now`). U produkciji su None pa moduli prave prave klijente; u
testovima se proslede fake-ovi, pa grananje grafa moze da se testira bez mreze.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, TypedDict

from langgraph.graph import END, StateGraph

import config
from pipeline import assemble, media, research, schedule, script, tts


@dataclass
class Deps:
    """Injektabilne zavisnosti (sve None u produkciji -> pravi klijenti)."""

    tavily_client: Any = None
    script_model: Any = None
    media_model: Any = None
    pexels_client: Any = None
    tts_runner: Optional[Callable] = None
    duration_probe: Optional[Callable] = None
    renderer: Optional[Callable] = None
    now: Any = None
    media_count: int = field(default_factory=lambda: config.PEXELS_CLIP_COUNT)


class PipelineState(TypedDict, total=False):
    """Deljeno stanje kroz graf (nodovi vracaju parcijalne update dict-ove)."""

    run_id: str
    topic: str
    out_dir: str
    deps: Deps

    references: list
    draft: Any  # script.ScriptDraft
    script_score: int
    script_feedback: Optional[str]
    script_attempts: int

    audio_path: str
    audio_duration: float

    keywords: list
    assets: list  # list[media.MediaAsset]
    media_score: int
    media_feedback: Optional[str]
    media_attempts: int

    video_path: str
    metadata: dict


# --- Nodovi ---

def _research_node(state: PipelineState) -> dict:
    deps = state["deps"]
    refs = research.run_research(state["topic"], client=deps.tavily_client)
    return {"references": refs}


def _generate_script_node(state: PipelineState) -> dict:
    deps = state["deps"]
    draft = script.generate_script(
        state["topic"],
        state.get("references", []),
        feedback=state.get("script_feedback"),
        model=deps.script_model,
    )
    return {"draft": draft, "script_attempts": state.get("script_attempts", 0) + 1}


def _eval_script_node(state: PipelineState) -> dict:
    deps = state["deps"]
    ev = script.evaluate_script(state["draft"], model=deps.script_model)
    return {"script_score": ev.score, "script_feedback": ev.feedback}


def _route_script(state: PipelineState) -> str:
    """Agent odluka: prosla ocena ili potroseni pokusaji -> dalje; inace retry."""
    passed = state.get("script_score", 0) >= config.SCRIPT_EVAL_THRESHOLD
    exhausted = state.get("script_attempts", 0) >= config.SCRIPT_MAX_ATTEMPTS
    return "continue" if (passed or exhausted) else "retry"


def _tts_node(state: PipelineState) -> dict:
    deps = state["deps"]
    draft = state["draft"]
    audio_path = Path(state["out_dir"]) / "audio.mp3"
    result = tts.synthesize(
        draft.script,
        audio_path,
        tts_runner=deps.tts_runner,
        duration_probe=deps.duration_probe,
    )
    return {"audio_path": result.audio_path, "audio_duration": result.duration}


def _keywords_node(state: PipelineState) -> dict:
    deps = state["deps"]
    draft = state["draft"]
    kws = media.extract_keywords(draft.title, draft.script, model=deps.media_model)
    return {"keywords": kws}


def _fetch_media_node(state: PipelineState) -> dict:
    deps = state["deps"]
    assets = media.fetch_media(
        state.get("keywords", []),
        count=deps.media_count,
        out_dir=state["out_dir"],
        client=deps.pexels_client,
    )
    return {"assets": assets, "media_attempts": state.get("media_attempts", 0) + 1}


def _eval_media_node(state: PipelineState) -> dict:
    deps = state["deps"]
    draft = state["draft"]
    ev = media.evaluate_media(draft.script, state.get("assets", []), model=deps.media_model)
    return {"media_score": ev.score, "media_feedback": ev.feedback}


def _route_media(state: PipelineState) -> str:
    passed = state.get("media_score", 0) >= config.MEDIA_EVAL_THRESHOLD
    exhausted = state.get("media_attempts", 0) >= config.MEDIA_MAX_ATTEMPTS
    return "continue" if (passed or exhausted) else "retry"


def _refine_keywords_node(state: PipelineState) -> dict:
    """Retry grana za media: iz eval feedback-a generisi bolje keywords."""
    deps = state["deps"]
    draft = state["draft"]
    kws = media.suggest_keywords_from_feedback(
        state.get("media_feedback", ""),
        draft.title,
        draft.script,
        model=deps.media_model,
    )
    # ako model ne vrati nista novo, zadrzi stare da fetch ne pukne
    return {"keywords": kws or state.get("keywords", [])}


def _assemble_node(state: PipelineState) -> dict:
    deps = state["deps"]
    draft = state["draft"]
    out_path = Path(state["out_dir"]) / "final.mp4"
    image_paths = [a.path for a in state.get("assets", [])]
    video_path = assemble.assemble_video(
        state["audio_path"],
        state["audio_duration"],
        image_paths,
        out_path,
        title=draft.title,
        renderer=deps.renderer,
    )
    return {"video_path": video_path}


def _schedule_node(state: PipelineState) -> dict:
    deps = state["deps"]
    draft = state["draft"]
    metadata = schedule.prepare_schedule(
        state["run_id"],
        draft.title,
        state["video_path"],
        state["out_dir"],
        now=deps.now,
    )
    return {"metadata": metadata}


# --- Sastavljanje grafa ---

def build_graph():
    """Sastavi i kompajliraj LangGraph pipeline."""
    g = StateGraph(PipelineState)

    g.add_node("research", _research_node)
    g.add_node("generate_script", _generate_script_node)
    g.add_node("eval_script", _eval_script_node)
    g.add_node("tts", _tts_node)
    g.add_node("extract_keywords", _keywords_node)
    g.add_node("fetch_media", _fetch_media_node)
    g.add_node("eval_media", _eval_media_node)
    g.add_node("refine_keywords", _refine_keywords_node)
    g.add_node("assemble", _assemble_node)
    g.add_node("schedule", _schedule_node)

    g.set_entry_point("research")
    g.add_edge("research", "generate_script")
    g.add_edge("generate_script", "eval_script")
    # Checkpoint 1: skripta (retry -> generate_script, continue -> tts)
    g.add_conditional_edges(
        "eval_script", _route_script,
        {"retry": "generate_script", "continue": "tts"},
    )
    g.add_edge("tts", "extract_keywords")
    g.add_edge("extract_keywords", "fetch_media")
    g.add_edge("fetch_media", "eval_media")
    # Checkpoint 2: vizuali (retry -> refine_keywords -> fetch_media, continue -> assemble)
    g.add_conditional_edges(
        "eval_media", _route_media,
        {"retry": "refine_keywords", "continue": "assemble"},
    )
    g.add_edge("refine_keywords", "fetch_media")
    g.add_edge("assemble", "schedule")
    g.add_edge("schedule", END)

    return g.compile()


def run_pipeline_streaming(
    topic: str,
    run_id: str,
    deps: Deps | None = None,
    on_step: Optional[Callable[[str, dict], None]] = None,
) -> dict:
    """Pokreni pipeline i zovi `on_step(node_name, state)` posle SVAKOG cvora.

    Isti graf i ista logika kao `run_pipeline`, samo se stanje emituje
    inkrementalno preko `graph.stream(..., stream_mode="updates")` umesto da
    se ceka kraj — koristi se za live progress (npr. /dashboard u api.py).

    Args:
        topic: tema videa.
        run_id: jedinstveni id (i ime output foldera).
        deps: injektabilne zavisnosti; None -> pravi klijenti.
        on_step: opciona funkcija (node_name, trenutno_stanje) -> None.
    """
    deps = deps or Deps()
    out_dir = config.OUTPUT_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    graph = build_graph()
    state: dict = {
        "run_id": run_id,
        "topic": topic,
        "out_dir": str(out_dir),
        "deps": deps,
    }
    # recursion_limit da beskonacna petlja (bug) ne visi zauvek
    for update in graph.stream(state, {"recursion_limit": 50}, stream_mode="updates"):
        for node_name, partial in update.items():
            state.update(partial)
            if on_step:
                on_step(node_name, dict(state))
    return state


def run_pipeline(topic: str, run_id: str, deps: Deps | None = None) -> dict:
    """Pokreni ceo pipeline za `topic` i vrati finalno stanje (bez progress callback-a)."""
    return run_pipeline_streaming(topic, run_id, deps=deps)
