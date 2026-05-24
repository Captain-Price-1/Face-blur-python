import json
from pathlib import Path

import pytest

from app import storage


@pytest.fixture
def tmp_jobs_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "JOBS_ROOT", tmp_path)
    return tmp_path


def test_new_job_creates_unique_dir(tmp_jobs_dir):
    job_id_1 = storage.new_job()
    job_id_2 = storage.new_job()
    assert job_id_1 != job_id_2
    assert (tmp_jobs_dir / job_id_1).is_dir()
    assert (tmp_jobs_dir / job_id_2 / "thumbs").is_dir()


def test_job_dir_returns_path(tmp_jobs_dir):
    job_id = storage.new_job()
    assert storage.job_dir(job_id) == tmp_jobs_dir / job_id


def test_write_and_read_analysis(tmp_jobs_dir):
    job_id = storage.new_job()
    payload = {"fps": 30, "people": [{"id": "p1"}]}
    storage.write_analysis(job_id, payload)
    assert storage.read_analysis(job_id) == payload


def test_input_path_and_output_path(tmp_jobs_dir):
    job_id = storage.new_job()
    assert storage.input_path(job_id).name == "input.mp4"
    assert storage.output_path(job_id).name == "output.mp4"
    assert storage.audio_path(job_id).name == "audio.m4a"


def test_thumb_path(tmp_jobs_dir):
    job_id = storage.new_job()
    p = storage.thumb_path(job_id, "p1")
    assert p.name == "p1.jpg"
    assert p.parent.name == "thumbs"


def test_delete_job_removes_dir(tmp_jobs_dir):
    job_id = storage.new_job()
    storage.delete_job(job_id)
    assert not (tmp_jobs_dir / job_id).exists()


def test_delete_job_unknown_id_is_noop(tmp_jobs_dir):
    storage.delete_job("does-not-exist")  # should not raise
