"""Testovi FastAPI webhook sloja — pipeline je mock-ovan (bez pravog run-a)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import api
import config


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    api._status_cache.clear()
    return TestClient(api.app)


def test_generate_returns_run_id(client, monkeypatch):
    def fake_run(topic, run_id, on_step=None):
        if on_step:
            on_step("research", {"script_attempts": 0, "media_attempts": 0})
        return {"video_path": "v.mp4", "metadata": {"status": "ready_to_publish"}}

    monkeypatch.setattr(api, "run_pipeline_streaming", fake_run)

    resp = client.post("/generate", json={"topic": "space facts"})
    assert resp.status_code == 200
    body = resp.json()
    assert "run_id" in body
    # TestClient izvrsi background task -> status prelazi u ready_to_publish
    status = client.get(f"/status/{body['run_id']}").json()
    assert status["status"] == "ready_to_publish"
    assert status["video_path"] == "v.mp4"


def test_generate_respects_explicit_run_id(client, monkeypatch):
    monkeypatch.setattr(
        api, "run_pipeline_streaming", lambda t, r, on_step=None: {"metadata": {}}
    )
    resp = client.post("/generate", json={"topic": "x", "run_id": "myrun"})
    assert resp.json()["run_id"] == "myrun"


def test_generate_rejects_invalid_run_id(client):
    resp = client.post("/generate", json={"topic": "x", "run_id": "../evil"})
    assert resp.status_code == 400


def test_failure_is_reported(client, monkeypatch):
    def boom(topic, run_id, on_step=None):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(api, "run_pipeline_streaming", boom)
    run_id = client.post("/generate", json={"topic": "x"}).json()["run_id"]
    status = client.get(f"/status/{run_id}").json()
    assert status["status"] == "failed"
    assert "kaboom" in status["error"]


def test_on_step_callback_writes_live_progress(client):
    # Direktan unit-test callback-a: TestClient izvrsava background task
    # sinhrono, pre nego sto GET /status stigne do njega, pa "mid-flight"
    # stanje ne moze pouzdano da se uhvati kroz pun HTTP round-trip.
    callback = api._on_step("cb_run", "space topic")
    callback("generate_script", {"script_attempts": 1, "script_score": 4})

    status = api._read_status("cb_run")
    assert status["status"] == "processing"
    assert status["current_node"] == "generate_script"
    assert status["script_attempts"] == 1
    assert status["script_score"] == 4


def test_unknown_run_id_404(client):
    assert client.get("/status/does-not-exist").status_code == 404


def test_status_rejects_invalid_run_id(client):
    resp = client.get("/status/bad.id")
    assert resp.status_code == 400


def test_empty_topic_rejected(client):
    assert client.post("/generate", json={"topic": ""}).status_code == 422


def test_root_info(client):
    assert client.get("/").status_code == 200


def test_dashboard_renders_html(client, monkeypatch):
    monkeypatch.setattr(
        api, "run_pipeline_streaming", lambda t, r, on_step=None: {"metadata": {}}
    )
    run_id = client.post("/generate", json={"topic": "x", "run_id": "dash1"}).json()["run_id"]
    resp = client.get(f"/dashboard/{run_id}")
    assert resp.status_code == 200
    assert "dash1" in resp.text
    assert "text/html" in resp.headers["content-type"]


def test_dashboard_rejects_invalid_run_id(client):
    resp = client.get("/dashboard/bad.id")
    assert resp.status_code == 400
