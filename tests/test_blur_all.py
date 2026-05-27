"""Tests for the single-pass 'blur everyone' mode."""
import shutil
import time

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app import downloader, ffmpeg_utils, jobs, storage
from app.main import app
from app.pipeline import blur_all


def test_blur_all_produces_playable_output(sample_video, tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "JOBS_ROOT", tmp_path)
    job_id = storage.new_job()
    shutil.copy(sample_video, storage.input_path(job_id))

    seen = []
    blur_all.run(job_id, progress_cb=lambda p: seen.append(p))

    out = storage.output_path(job_id)
    assert out.exists() and out.stat().st_size > 0
    assert ffmpeg_utils.probe(out)["duration_sec"] > 0
    assert seen and seen[-1] >= 0.99


def test_blur_all_blurs_a_detected_face(sample_video, tmp_path, monkeypatch):
    from app.pipeline import detect
    monkeypatch.setattr(storage, "JOBS_ROOT", tmp_path)
    job_id = storage.new_job()
    shutil.copy(sample_video, storage.input_path(job_id))

    # Find a frame + bbox where a face is detected in the source.
    cap = cv2.VideoCapture(str(storage.input_path(job_id)))
    target_frame, target_bbox = None, None
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        faces = detect.detect_faces(frame)
        if faces:
            target_frame, target_bbox = idx, faces[0]["bbox"]
            break
        idx += 1
    cap.release()
    assert target_frame is not None, "no face in fixture"

    blur_all.run(job_id, progress_cb=lambda p: None)

    def _roi(path, fidx, bbox):
        c = cv2.VideoCapture(str(path))
        c.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ok, fr = c.read()
        c.release()
        assert ok
        x, y, w, h = bbox
        return fr[y:y+h, x:x+w].astype(np.float32)

    before = _roi(storage.input_path(job_id), target_frame, target_bbox).var()
    after = _roi(storage.output_path(job_id), target_frame, target_bbox).var()
    assert after < before * 0.6


def _wait(client, job_id, target, timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = client.get(f"/api/jobs/{job_id}").json()["status"]
        if st == target:
            return
        if st == "error":
            raise AssertionError(client.get(f"/api/jobs/{job_id}").json())
        time.sleep(0.5)
    raise AssertionError(f"timeout waiting for {target}")


def test_api_upload_blur_all_skips_picker(tmp_path, monkeypatch, sample_video):
    monkeypatch.setattr(storage, "JOBS_ROOT", tmp_path)
    jobs.reset()
    client = TestClient(app)
    with sample_video.open("rb") as fh:
        r = client.post(
            "/api/jobs",
            files={"file": ("s.mp4", fh, "video/mp4")},
            data={"blur_all": "true"},
        )
    assert r.status_code == 200
    job_id = r.json()["job_id"]
    assert r.json()["status"] == "blurring"
    _wait(client, job_id, "done")
    r = client.get(f"/api/jobs/{job_id}/download")
    assert r.status_code == 200
    assert len(r.content) > 0


def test_api_from_url_blur_all(tmp_path, monkeypatch, sample_video):
    monkeypatch.setattr(storage, "JOBS_ROOT", tmp_path)
    jobs.reset()

    def fake_download(url, dst, progress_cb=None):
        shutil.copy(sample_video, dst)

    monkeypatch.setattr(downloader, "download", fake_download)

    client = TestClient(app)
    r = client.post("/api/jobs/from-url", json={"url": "https://youtu.be/x", "blur_all": True})
    assert r.status_code == 200
    job_id = r.json()["job_id"]
    _wait(client, job_id, "done")
    assert client.get(f"/api/jobs/{job_id}/download").status_code == 200
