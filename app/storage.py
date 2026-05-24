"""All disk I/O for jobs. Other modules MUST call through here."""
from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

JOBS_ROOT = Path(__file__).parent.parent / "jobs"


def _ensure_root() -> None:
    JOBS_ROOT.mkdir(parents=True, exist_ok=True)


def new_job() -> str:
    _ensure_root()
    job_id = uuid.uuid4().hex
    (JOBS_ROOT / job_id / "thumbs").mkdir(parents=True)
    return job_id


def job_dir(job_id: str) -> Path:
    return JOBS_ROOT / job_id


def input_path(job_id: str) -> Path:
    return job_dir(job_id) / "input.mp4"


def output_path(job_id: str) -> Path:
    return job_dir(job_id) / "output.mp4"


def audio_path(job_id: str) -> Path:
    return job_dir(job_id) / "audio.m4a"


def video_only_path(job_id: str) -> Path:
    return job_dir(job_id) / "video_only.mp4"


def thumb_path(job_id: str, person_id: str) -> Path:
    return job_dir(job_id) / "thumbs" / f"{person_id}.jpg"


def analysis_path(job_id: str) -> Path:
    return job_dir(job_id) / "analysis.json"


def write_analysis(job_id: str, payload: dict) -> None:
    analysis_path(job_id).write_text(json.dumps(payload))


def read_analysis(job_id: str) -> dict:
    return json.loads(analysis_path(job_id).read_text())


def delete_job(job_id: str) -> None:
    target = job_dir(job_id)
    if target.exists():
        shutil.rmtree(target)
