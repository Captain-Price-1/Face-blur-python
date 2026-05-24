import shutil

import cv2
import numpy as np

from app import ffmpeg_utils, storage
from app.pipeline import analyze, render


def _setup_job(sample_video, tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "JOBS_ROOT", tmp_path)
    job_id = storage.new_job()
    shutil.copy(sample_video, storage.input_path(job_id))
    analyze.run(job_id, progress_cb=lambda p: None)
    return job_id


def test_render_produces_playable_output(sample_video, tmp_path, monkeypatch):
    job_id = _setup_job(sample_video, tmp_path, monkeypatch)
    data = storage.read_analysis(job_id)
    blur_ids = [data["people"][0]["id"]]
    render.run(job_id, blur_ids, progress_cb=lambda p: None)
    out = storage.output_path(job_id)
    assert out.exists() and out.stat().st_size > 0
    info = ffmpeg_utils.probe(out)
    assert info["duration_sec"] > 0


def test_render_actually_blurs_selected_face(sample_video, tmp_path, monkeypatch):
    job_id = _setup_job(sample_video, tmp_path, monkeypatch)
    data = storage.read_analysis(job_id)
    blur_ids = [data["people"][0]["id"]]
    render.run(job_id, blur_ids, progress_cb=lambda p: None)

    target_pid = blur_ids[0]
    sample_frame = None
    sample_bbox = None
    for entry in data["timeline"]:
        for face in entry["faces"]:
            if face["person_id"] == target_pid:
                sample_frame = entry["frame"]
                sample_bbox = tuple(face["bbox"])
                break
        if sample_frame is not None:
            break
    assert sample_frame is not None

    def _roi(video_path, frame_idx, bbox):
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        cap.release()
        assert ok
        x, y, w, h = bbox
        return frame[y:y+h, x:x+w].astype(np.float32)

    var_before = _roi(storage.input_path(job_id), sample_frame, sample_bbox).var()
    var_after = _roi(storage.output_path(job_id), sample_frame, sample_bbox).var()
    assert var_after < var_before * 0.5


def test_render_progress_callback(sample_video, tmp_path, monkeypatch):
    job_id = _setup_job(sample_video, tmp_path, monkeypatch)
    seen = []
    render.run(job_id, [], progress_cb=lambda p: seen.append(p))
    assert seen and seen[-1] >= 0.99


def test_render_with_no_audio_track(tmp_path, monkeypatch):
    import subprocess
    monkeypatch.setattr(storage, "JOBS_ROOT", tmp_path)
    job_id = storage.new_job()
    src = storage.input_path(job_id)
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=black:s=160x120:r=5:d=1",
         "-c:v", "libx264", str(src)],
        check=True, capture_output=True,
    )
    storage.write_analysis(job_id, {
        "fps": 5, "duration_sec": 1, "width": 160, "height": 120,
        "has_audio": False, "people": [], "timeline": [],
    })
    render.run(job_id, [], progress_cb=lambda p: None)
    assert storage.output_path(job_id).exists()
