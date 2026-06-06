import time

from fastapi.testclient import TestClient

from app import jobs, storage
from app.main import app


def _wait_status(client, job_id, target, timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/api/jobs/{job_id}")
        assert r.status_code == 200
        if r.json()["status"] == target:
            return r.json()
        if r.json()["status"] == "error":
            raise AssertionError(r.json())
        time.sleep(0.5)
    raise AssertionError(f"timeout waiting for {target}")


def test_happy_path(tmp_path, monkeypatch, sample_video):
    monkeypatch.setattr(storage, "JOBS_ROOT", tmp_path)
    jobs.reset()
    client = TestClient(app)

    with sample_video.open("rb") as fh:
        r = client.post("/api/jobs", files={"file": ("sample.mp4", fh, "video/mp4")})
    assert r.status_code == 200
    job_id = r.json()["job_id"]
    assert r.json()["status"] == "analyzing"

    _wait_status(client, job_id, "awaiting_selection")

    r = client.get(f"/api/jobs/{job_id}/people")
    assert r.status_code == 200
    people = r.json()["people"]
    assert len(people) >= 1
    for p in people:
        r = client.get(p["thumb_url"])
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("image/")

    r = client.post(f"/api/jobs/{job_id}/render", json={"blur_person_ids": [people[0]["id"]]})
    assert r.status_code == 200

    _wait_status(client, job_id, "done")

    r = client.get(f"/api/jobs/{job_id}/download")
    assert r.status_code == 200
    assert r.headers["content-type"] == "video/mp4"
    assert len(r.content) > 0

    r = client.delete(f"/api/jobs/{job_id}")
    assert r.status_code == 200
    r = client.get(f"/api/jobs/{job_id}")
    assert r.json()["status"] == "unknown"


def test_render_idempotent_when_already_running(tmp_path, monkeypatch, sample_video):
    monkeypatch.setattr(storage, "JOBS_ROOT", tmp_path)
    jobs.reset()
    client = TestClient(app)
    with sample_video.open("rb") as fh:
        job_id = client.post("/api/jobs", files={"file": ("s.mp4", fh, "video/mp4")}).json()["job_id"]
    _wait_status(client, job_id, "awaiting_selection")
    client.post(f"/api/jobs/{job_id}/render", json={"blur_person_ids": []})
    r = client.post(f"/api/jobs/{job_id}/render", json={"blur_person_ids": []})
    assert r.status_code == 200
    _wait_status(client, job_id, "done")


def test_ws_receives_progress(tmp_path, monkeypatch, sample_video):
    monkeypatch.setattr(storage, "JOBS_ROOT", tmp_path)
    jobs.reset()
    client = TestClient(app)
    with sample_video.open("rb") as fh:
        job_id = client.post("/api/jobs", files={"file": ("s.mp4", fh, "video/mp4")}).json()["job_id"]

    with client.websocket_connect(f"/api/jobs/{job_id}/events") as ws:
        phases = set()
        for _ in range(50):
            ev = ws.receive_json()
            phases.add(ev["phase"])
            if ev["phase"] == "awaiting_selection":
                break
        assert "analyzing" in phases
        assert "awaiting_selection" in phases


def test_body_blur_rejected_for_long_video(tmp_path, monkeypatch):
    """Whole-body blur is capped to short clips; a long video must be rejected
    with a helpful 400 (prevents the EdgeTAM out-of-memory crash)."""
    monkeypatch.setattr(storage, "JOBS_ROOT", tmp_path)
    jobs.reset()
    job_id = jobs.create()
    # Fake a long analysis (no real processing needed for the guard).
    storage.write_analysis(job_id, {
        "fps": 30, "duration_sec": 1200, "width": 640, "height": 360,
        "has_audio": False, "people": [{"id": "p1", "thumb": "thumbs/p1.jpg",
        "frame_count": 10, "first_seen_sec": 0.0}], "timeline": [],
    })
    from app import jobs as J
    J._jobs[job_id] = {"status": "awaiting_selection", "progress": 1.0}

    client = TestClient(app)
    r = client.post(f"/api/jobs/{job_id}/render",
                    json={"blur_person_ids": ["p1"], "blur_mode": "body_silhouette"})
    assert r.status_code == 400
    assert "minutes" in r.json()["detail"].lower()

    # face mode on the same long video is allowed
    r = client.post(f"/api/jobs/{job_id}/render",
                    json={"blur_person_ids": ["p1"], "blur_mode": "face"})
    assert r.status_code == 200
