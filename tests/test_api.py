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
    def fake_run(topic, run_id, deps=None):
        return {"video_path": "v.mp4", "metadata": {"status": "ready_to_publish"}}

    monkeypatch.setattr(api, "run_pipeline", fake_run)

    resp = client.post("/generate", json={"topic": "space facts"})
    assert resp.status_code == 200
    body = resp.json()
    assert "run_id" in body
    # TestClient izvrsi background task -> status prelazi u ready_to_publish
    status = client.get(f"/status/{body['run_id']}").json()
    assert status["status"] == "ready_to_publish"
    assert status["video_path"] == "v.mp4"


def test_generate_respects_explicit_run_id(client, monkeypatch):
    monkeypatch.setattr(api, "run_pipeline", lambda t, r, deps=None: {"metadata": {}})
    resp = client.post("/generate", json={"topic": "x", "run_id": "myrun"})
    assert resp.json()["run_id"] == "myrun"


def test_failure_is_reported(client, monkeypatch):
    def boom(topic, run_id, deps=None):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(api, "run_pipeline", boom)
    run_id = client.post("/generate", json={"topic": "x"}).json()["run_id"]
    status = client.get(f"/status/{run_id}").json()
    assert status["status"] == "failed"
    assert "kaboom" in status["error"]


def test_unknown_run_id_404(client):
    assert client.get("/status/does-not-exist").status_code == 404


def test_empty_topic_rejected(client):
    assert client.post("/generate", json={"topic": ""}).status_code == 422


def test_root_info(client):
    assert client.get("/").status_code == 200
