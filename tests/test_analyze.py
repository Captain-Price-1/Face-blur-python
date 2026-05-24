import json
import shutil
from pathlib import Path

from app import storage
from app.pipeline import analyze


def test_analyze_produces_people_and_timeline(sample_video, tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "JOBS_ROOT", tmp_path)
    job_id = storage.new_job()
    shutil.copy(sample_video, storage.input_path(job_id))

    analyze.run(job_id, progress_cb=lambda p: None)

    data = storage.read_analysis(job_id)
    assert data["fps"] > 0
    assert data["duration_sec"] > 0
    # Sample video is short and may have varying people counts; require at least one.
    assert len(data["people"]) >= 1
    for person in data["people"]:
        assert {"id", "thumb", "frame_count", "first_seen_sec"} <= person.keys()
        assert storage.thumb_path(job_id, person["id"]).exists()
    frames = [t["frame"] for t in data["timeline"]]
    assert frames == sorted(frames)


def test_analyze_progress_callback_called(sample_video, tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "JOBS_ROOT", tmp_path)
    job_id = storage.new_job()
    shutil.copy(sample_video, storage.input_path(job_id))

    seen = []
    analyze.run(job_id, progress_cb=lambda p: seen.append(p))

    assert seen, "progress_cb was never called"
    assert all(0.0 <= p <= 1.0 for p in seen)
    assert seen[-1] >= 0.99
