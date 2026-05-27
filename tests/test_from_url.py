"""Tests for the YouTube/URL ingestion flow.

We do NOT hit the network. `downloader.download` is monkeypatched to copy the
local fixture into the job's input path, which exercises the full
download -> analyze -> awaiting_selection -> render -> download pipeline.
"""
import shutil
import time

from fastapi.testclient import TestClient

from app import downloader, jobs, storage
from app.main import app


def _wait_status(client, job_id, target, timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/api/jobs/{job_id}")
        assert r.status_code == 200
        st = r.json()["status"]
        if st == target:
            return r.json()
        if st == "error":
            raise AssertionError(r.json())
        time.sleep(0.5)
    raise AssertionError(f"timeout waiting for {target}")


def test_from_url_runs_full_pipeline(tmp_path, monkeypatch, sample_video):
    monkeypatch.setattr(storage, "JOBS_ROOT", tmp_path)
    jobs.reset()

    # Stub the actual download: copy the fixture to the job's input path.
    def fake_download(url, dst, progress_cb=None):
        assert url == "https://www.youtube.com/watch?v=TEST"
        if progress_cb:
            progress_cb(0.5)
            progress_cb(1.0)
        shutil.copy(sample_video, dst)

    monkeypatch.setattr(downloader, "download", fake_download)

    client = TestClient(app)
    r = client.post("/api/jobs/from-url", json={"url": "https://www.youtube.com/watch?v=TEST"})
    assert r.status_code == 200
    job_id = r.json()["job_id"]
    assert r.json()["status"] == "downloading"

    _wait_status(client, job_id, "awaiting_selection")

    people = client.get(f"/api/jobs/{job_id}/people").json()["people"]
    assert len(people) >= 1

    r = client.post(f"/api/jobs/{job_id}/render", json={"blur_person_ids": [people[0]["id"]]})
    assert r.status_code == 200
    _wait_status(client, job_id, "done")

    r = client.get(f"/api/jobs/{job_id}/download")
    assert r.status_code == 200
    assert r.headers["content-type"] == "video/mp4"
    assert len(r.content) > 0


def test_from_url_rejects_empty_url(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "JOBS_ROOT", tmp_path)
    jobs.reset()
    client = TestClient(app)
    r = client.post("/api/jobs/from-url", json={"url": "   "})
    assert r.status_code == 400


def test_from_url_reports_download_error(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "JOBS_ROOT", tmp_path)
    jobs.reset()

    def boom(url, dst, progress_cb=None):
        raise RuntimeError("video unavailable")

    monkeypatch.setattr(downloader, "download", boom)

    client = TestClient(app)
    job_id = client.post(
        "/api/jobs/from-url", json={"url": "https://youtu.be/bad"}
    ).json()["job_id"]

    deadline = time.time() + 30
    while time.time() < deadline:
        st = client.get(f"/api/jobs/{job_id}").json()
        if st["status"] == "error":
            assert "download failed" in st["error"]
            return
        time.sleep(0.3)
    raise AssertionError("expected error status")
