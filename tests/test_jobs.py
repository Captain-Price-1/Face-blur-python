import asyncio
import shutil

import pytest

from app import jobs, storage


@pytest.fixture
def isolated_jobs(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "JOBS_ROOT", tmp_path)
    jobs.reset()
    yield tmp_path
    jobs.reset()


@pytest.mark.asyncio
async def test_create_and_run_analyze(isolated_jobs, sample_video):
    job_id = jobs.create()
    shutil.copy(sample_video, storage.input_path(job_id))

    jobs.start_analyze(job_id)
    for _ in range(120):
        await asyncio.sleep(0.5)
        state = jobs.get(job_id)
        if state["status"] in ("awaiting_selection", "error"):
            break
    assert jobs.get(job_id)["status"] == "awaiting_selection"


@pytest.mark.asyncio
async def test_render_runs_after_analyze(isolated_jobs, sample_video):
    job_id = jobs.create()
    shutil.copy(sample_video, storage.input_path(job_id))
    jobs.start_analyze(job_id)
    while jobs.get(job_id)["status"] not in ("awaiting_selection", "error"):
        await asyncio.sleep(0.5)
    assert jobs.get(job_id)["status"] == "awaiting_selection"
    data = storage.read_analysis(job_id)
    blur_ids = [p["id"] for p in data["people"][:1]]

    jobs.start_render(job_id, blur_ids)
    for _ in range(120):
        await asyncio.sleep(0.5)
        if jobs.get(job_id)["status"] in ("done", "error"):
            break
    assert jobs.get(job_id)["status"] == "done"
    assert storage.output_path(job_id).exists()


@pytest.mark.asyncio
async def test_double_render_is_noop_while_running(isolated_jobs, sample_video):
    job_id = jobs.create()
    shutil.copy(sample_video, storage.input_path(job_id))
    jobs.start_analyze(job_id)
    while jobs.get(job_id)["status"] != "awaiting_selection":
        await asyncio.sleep(0.5)
    jobs.start_render(job_id, [])
    jobs.start_render(job_id, [])
    while jobs.get(job_id)["status"] != "done":
        await asyncio.sleep(0.5)


@pytest.mark.asyncio
async def test_events_queue_receives_progress(isolated_jobs, sample_video):
    job_id = jobs.create()
    shutil.copy(sample_video, storage.input_path(job_id))
    q = jobs.subscribe(job_id)
    jobs.start_analyze(job_id)
    phases_seen = set()
    while True:
        ev = await asyncio.wait_for(q.get(), timeout=90)
        phases_seen.add(ev["phase"])
        if ev["phase"] in ("awaiting_selection", "error"):
            break
    assert "analyzing" in phases_seen
    assert "awaiting_selection" in phases_seen
