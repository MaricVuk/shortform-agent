"""Live status dashboard — vizuelni prikaz jednog pipeline run-a.

Cist HTML/CSS/vanilla JS (bez build koraka ili eksternih CDN-ova), koji
poll-uje `/status/{run_id}` i renderuje 6 faza pipeline-a sa stanjem
(pending/active/done/failed), self-eval attempt/score brojacima za obe
agentske petlje, i finalnim videom kad je gotov.

Odvojeno od `api.py` da endpoint fajl ostane citljiv; `api.py` samo importuje
`render(run_id)`.
"""
from __future__ import annotations

# Redosled i mapiranje LangGraph cvorova (orchestration/graph.py) na "faze"
# koje korisnik vidi — vise cvorova (npr. generate_script + eval_script) se
# spaja u jednu fazu jer je to jedna agentska celina (self-eval petlja).
STAGE_ORDER = ["research", "script", "narration", "visuals", "assembly", "scheduling"]

NODE_TO_STAGE = {
    "research": "research",
    "generate_script": "script",
    "eval_script": "script",
    "tts": "narration",
    "extract_keywords": "visuals",
    "fetch_media": "visuals",
    "eval_media": "visuals",
    "refine_keywords": "visuals",
    "assemble": "assembly",
    "schedule": "scheduling",
}

STAGE_META = {
    "research": ("Research", "Tavily pretrazuje web za temu"),
    "script": ("Skripta + self-eval", "Groq generise, sam ocenjuje hook, retry ako je slab"),
    "narration": ("TTS naracija", "edge-tts pretvara skriptu u audio"),
    "visuals": ("Vizuali + self-eval", "Pexels preuzima slike, Groq ocenjuje relevantnost"),
    "assembly": ("Montaza", "MoviePy spaja audio + vizuale u MP4"),
    "scheduling": ("Mock scheduling", "priprema metadata.json (bez pravog upload API-ja)"),
}

_PAGE = """<!doctype html>
<html lang="sr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Shortform Agent — run __RUN_ID__</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 2rem 1rem 4rem;
    background: #0b0d12; color: #e6e8ee;
    font-family: -apple-system, Segoe UI, Roboto, sans-serif;
    display: flex; justify-content: center;
  }
  .wrap { width: 100%; max-width: 720px; }
  h1 { font-size: 1.25rem; font-weight: 600; margin: 0 0 .25rem; }
  .run-id { color: #8a90a2; font-size: .85rem; margin-bottom: 1.5rem; }
  .badge {
    display: inline-block; padding: .25rem .65rem; border-radius: 999px;
    font-size: .75rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: .03em; margin-bottom: 1.5rem;
  }
  .badge.queued { background: #2a2f3d; color: #9aa3b8; }
  .badge.processing { background: #2b3a55; color: #7fb3ff; }
  .badge.ready_to_publish { background: #1e3d2e; color: #6fe3a0; }
  .badge.failed { background: #3d1e24; color: #ff8a8a; }

  .stage {
    display: flex; gap: .9rem; padding: .85rem 0;
    border-bottom: 1px solid #1c202b;
  }
  .stage:last-child { border-bottom: none; }
  .dot {
    flex: none; width: 1.6rem; height: 1.6rem; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: .85rem; font-weight: 700; margin-top: .1rem;
    background: #1c202b; color: #565d70;
  }
  .stage.done .dot { background: #1e3d2e; color: #6fe3a0; }
  .stage.active .dot { background: #2b3a55; color: #7fb3ff; animation: pulse 1.4s ease-in-out infinite; }
  .stage.failed .dot { background: #3d1e24; color: #ff8a8a; }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: .45; } }

  .stage-title { font-weight: 600; font-size: .95rem; }
  .stage-desc { color: #8a90a2; font-size: .82rem; margin-top: .1rem; }
  .stage-meta { color: #7fb3ff; font-size: .8rem; margin-top: .35rem; }

  .panel {
    margin-top: 1.5rem; padding: 1rem 1.1rem; border-radius: .6rem;
    background: #12151d; border: 1px solid #1c202b;
  }
  .panel.error { border-color: #3d1e24; }
  .panel pre {
    white-space: pre-wrap; word-break: break-word; font-size: .78rem;
    color: #ff9a9a; max-height: 220px; overflow-y: auto; margin: .5rem 0 0;
  }
  video { width: 100%; border-radius: .5rem; margin-top: .5rem; background: #000; }
  .meta-row { font-size: .85rem; color: #b6bccb; margin-top: .35rem; }
  a { color: #7fb3ff; }
</style>
</head>
<body>
<div class="wrap">
  <h1>Shortform Agent</h1>
  <div class="run-id">run_id: <code>__RUN_ID__</code> &middot; tema: <span id="topic">—</span></div>
  <span id="badge" class="badge queued">queued</span>
  <div id="stages"></div>
  <div id="result"></div>
</div>
<script>
const RUN_ID = "__RUN_ID__";
const STAGE_ORDER = __STAGE_ORDER_JSON__;
const NODE_TO_STAGE = __NODE_TO_STAGE_JSON__;
const STAGE_META = __STAGE_META_JSON__;

const stagesEl = document.getElementById("stages");
const badgeEl = document.getElementById("badge");
const resultEl = document.getElementById("result");
const topicEl = document.getElementById("topic");

function renderStages(activeIndex, terminalStatus) {
  stagesEl.innerHTML = STAGE_ORDER.map((key, i) => {
    const [title, desc] = STAGE_META[key];
    let cls = "pending", icon = String(i + 1);
    if (terminalStatus === "ready_to_publish") { cls = "done"; icon = "✓"; }
    else if (terminalStatus === "failed" && i === activeIndex) { cls = "failed"; icon = "✕"; }
    else if (i < activeIndex || (terminalStatus === "failed" && i < activeIndex)) { cls = "done"; icon = "✓"; }
    else if (i === activeIndex) { cls = "active"; icon = String(i + 1); }
    return `<div class="stage ${cls}"><div class="dot">${icon}</div>
      <div><div class="stage-title">${title}</div>
      <div class="stage-desc">${desc}</div>
      <div class="stage-meta" data-stage="${key}"></div></div></div>`;
  }).join("");
}

function setStageMeta(key, text) {
  const el = stagesEl.querySelector(`.stage-meta[data-stage="${key}"]`);
  if (el) el.textContent = text;
}

async function poll() {
  let data;
  try {
    const resp = await fetch(`/status/${RUN_ID}`);
    data = await resp.json();
  } catch (e) {
    return setTimeout(poll, 1500);
  }

  topicEl.textContent = data.topic || "—";
  badgeEl.textContent = data.status;
  badgeEl.className = "badge " + data.status;

  const currentStage = data.current_node ? NODE_TO_STAGE[data.current_node] : null;
  let activeIndex = currentStage ? STAGE_ORDER.indexOf(currentStage) : 0;
  if (data.status === "ready_to_publish") activeIndex = STAGE_ORDER.length;
  renderStages(activeIndex, data.status);

  if (data.script_attempts) {
    let txt = `pokusaj ${data.script_attempts}`;
    if (data.script_score) txt += ` — ocena ${data.script_score}/10`;
    setStageMeta("script", txt);
  }
  if (data.media_attempts) {
    let txt = `pokusaj ${data.media_attempts}`;
    if (data.media_score) txt += ` — ocena ${data.media_score}/10`;
    setStageMeta("visuals", txt);
  }

  if (data.status === "ready_to_publish") {
    const md = data.metadata || {};
    resultEl.innerHTML = `<div class="panel">
      <video controls src="/files/${RUN_ID}/final.mp4"></video>
      <div class="meta-row"><b>Naslov:</b> ${md.title || ""}</div>
      <div class="meta-row"><b>Platforme:</b> ${(md.platforms || []).join(", ")}</div>
      <div class="meta-row"><b>Predlozeno vreme objave:</b> ${md.suggested_publish_time || ""}</div>
      <div class="meta-row"><a href="/files/${RUN_ID}/metadata.json" target="_blank">metadata.json</a></div>
    </div>`;
    return; // stop polling
  }
  if (data.status === "failed") {
    resultEl.innerHTML = `<div class="panel error">
      <div class="meta-row"><b>Greska:</b> ${data.error || "nepoznata greska"}</div>
      ${data.traceback ? `<pre>${data.traceback}</pre>` : ""}
    </div>`;
    return; // stop polling
  }
  setTimeout(poll, 1200);
}

renderStages(0, "queued");
poll();
</script>
</body>
</html>
"""


def render(run_id: str) -> str:
    """Vrati kompletan HTML dashboard za dati `run_id`."""
    import json as _json

    return (
        _PAGE.replace("__RUN_ID__", run_id)
        .replace("__STAGE_ORDER_JSON__", _json.dumps(STAGE_ORDER))
        .replace("__NODE_TO_STAGE_JSON__", _json.dumps(NODE_TO_STAGE))
        .replace("__STAGE_META_JSON__", _json.dumps(STAGE_META))
    )
