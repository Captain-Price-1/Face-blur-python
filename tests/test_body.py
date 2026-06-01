import shutil
import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import ffmpeg_utils, jobs, storage
from app.main import app
from app.pipeline import analyze, body, render


def _first_frame(video_path):
    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()
    assert ok
    return frame


def test_detect_bodies_on_fixture(sample_video):
    frame = _first_frame(sample_video)
    boxes = body.detect_bodies(frame)
    assert isinstance(boxes, list)
    for (x, y, w, h) in boxes:
        assert w > 0 and h > 0


def test_segment_bodies_returns_masks(sample_video):
    frame = _first_frame(sample_video)
    seg = body.segment_bodies(frame)
    assert isinstance(seg, list)
    if not seg:
        pytest.skip("no person detected in fixture frame")
    for (bbox, mask) in seg:
        assert len(bbox) == 4
        assert mask.shape == frame.shape[:2]
        assert mask.dtype == np.uint8
        assert set(np.unique(mask)).issubset({0, 255})


def _setup_job(sample_video, tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "JOBS_ROOT", tmp_path)
    job_id = storage.new_job()
    shutil.copy(sample_video, storage.input_path(job_id))
    analyze.run(job_id, progress_cb=lambda p: None)
    return job_id


def test_render_body_box_produces_output(sample_video, tmp_path, monkeypatch):
    job_id = _setup_job(sample_video, tmp_path, monkeypatch)
    data = storage.read_analysis(job_id)
    blur_ids = [data["people"][0]["id"]]
    render.run(job_id, blur_ids, progress_cb=lambda p: None, blur_mode="body_box")
    out = storage.output_path(job_id)
    assert out.exists() and out.stat().st_size > 0
    assert ffmpeg_utils.probe(out)["duration_sec"] > 0


def test_render_body_silhouette_produces_output(sample_video, tmp_path, monkeypatch):
    job_id = _setup_job(sample_video, tmp_path, monkeypatch)
    data = storage.read_analysis(job_id)
    blur_ids = [data["people"][0]["id"]]
    render.run(job_id, blur_ids, progress_cb=lambda p: None, blur_mode="body_silhouette")
    out = storage.output_path(job_id)
    assert out.exists() and out.stat().st_size > 0
    assert ffmpeg_utils.probe(out)["duration_sec"] > 0


def _wait_status(client, job_id, target, timeout=180):
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


def test_api_render_accepts_blur_mode(tmp_path, monkeypatch, sample_video):
    monkeypatch.setattr(storage, "JOBS_ROOT", tmp_path)
    jobs.reset()
    client = TestClient(app)

    with sample_video.open("rb") as fh:
        job_id = client.post(
            "/api/jobs", files={"file": ("sample.mp4", fh, "video/mp4")}
        ).json()["job_id"]
    _wait_status(client, job_id, "awaiting_selection")

    pid = client.get(f"/api/jobs/{job_id}/people").json()["people"][0]["id"]

    # bogus mode -> 400
    r = client.post(
        f"/api/jobs/{job_id}/render",
        json={"blur_person_ids": [pid], "blur_mode": "bogus"},
    )
    assert r.status_code == 400

    # valid body_box -> happy path
    r = client.post(
        f"/api/jobs/{job_id}/render",
        json={"blur_person_ids": [pid], "blur_mode": "body_box"},
    )
    assert r.status_code == 200

    _wait_status(client, job_id, "done")
    r = client.get(f"/api/jobs/{job_id}/download")
    assert r.status_code == 200
    assert len(r.content) > 0


# ---- _match_body association (regression for wrong-person / flicker bug) ----
from app.pipeline.render import _match_body


def test_match_body_picks_head_owner_not_overlapping_neighbor():
    # The face: center-x = 150, top = 60.
    face = (140, 60, 20, 24)
    # Body A (the true owner): face sits at A's top-center (center-x = 150).
    A = (110, 55, 80, 250)
    # Body B: a prominent overlapping neighbour that ALSO fully contains the
    # face (old containment score = 1.0) and is listed FIRST — exactly the case
    # the old logic got wrong. The face is off-centre for B and not at B's head,
    # so the fixed head-position logic correctly prefers A.
    B = (60, 50, 150, 260)   # center-x = 135, fully contains the face
    assert _match_body(face, [B, A]) == A


def test_match_body_rejects_body_when_face_in_lower_region():
    # Face down in the torso/legs region of this body -> not its head -> no match.
    body = (0, 0, 100, 300)
    face = (40, 250, 20, 24)   # center-y ~262, well below the head region
    assert _match_body(face, [body]) is None


def test_match_body_temporal_continuity_breaks_near_ties():
    # Two equally-plausible bodies symmetric around the face.
    left = (110, 50, 60, 240)   # center-x = 140
    right = (130, 50, 60, 240)  # center-x = 160
    face = (140, 52, 20, 24)    # center-x = 150 -> equidistant
    # With prev match == left, it should stick to left (and vice-versa).
    assert _match_body(face, [left, right], prev_body=left) == left
    assert _match_body(face, [left, right], prev_body=right) == right


def test_match_body_single_body_still_matches():
    body = (10, 10, 100, 300)
    face = (45, 14, 20, 24)     # center-x = 55 inside, near top
    assert _match_body(face, [body]) == body
