"""In-memory job registry + asyncio worker."""
from __future__ import annotations

import asyncio
import traceback
from typing import Any

from app import storage
from app.pipeline import analyze, render

_jobs: dict[str, dict[str, Any]] = {}
_tasks: dict[str, asyncio.Task] = {}
_subscribers: dict[str, list[asyncio.Queue]] = {}


def reset() -> None:
    """Test helper: wipe in-memory state."""
    _jobs.clear()
    _tasks.clear()
    _subscribers.clear()


def create() -> str:
    job_id = storage.new_job()
    _jobs[job_id] = {"status": "created", "progress": 0.0}
    return job_id


def get(job_id: str) -> dict[str, Any]:
    return _jobs.get(job_id, {"status": "unknown"})


def subscribe(job_id: str) -> asyncio.Queue:
    q: asyncio.Queue = asyncio.Queue()
    _subscribers.setdefault(job_id, []).append(q)
    return q


def _emit(job_id: str, event: dict[str, Any]) -> None:
    for q in _subscribers.get(job_id, []):
        q.put_nowait(event)


def _set(job_id: str, **fields) -> None:
    _jobs[job_id].update(fields)


def start_analyze(job_id: str) -> None:
    if job_id in _tasks and not _tasks[job_id].done():
        return
    _tasks[job_id] = asyncio.create_task(_run_analyze(job_id))


def start_render(job_id: str, blur_person_ids: list[str]) -> None:
    if job_id in _tasks and not _tasks[job_id].done():
        return
    _tasks[job_id] = asyncio.create_task(_run_render(job_id, blur_person_ids))


async def _run_analyze(job_id: str) -> None:
    _set(job_id, status="analyzing", progress=0.0)
    _emit(job_id, {"phase": "analyzing", "progress": 0.0})
    loop = asyncio.get_running_loop()

    def cb(p: float) -> None:
        _set(job_id, progress=p)
        loop.call_soon_threadsafe(_emit, job_id, {"phase": "analyzing", "progress": p})

    try:
        await asyncio.to_thread(analyze.run, job_id, cb)
        _set(job_id, status="awaiting_selection", progress=1.0)
        _emit(job_id, {"phase": "awaiting_selection"})
    except Exception as e:
        traceback.print_exc()
        _set(job_id, status="error", error=str(e))
        _emit(job_id, {"phase": "error", "message": str(e)})


async def _run_render(job_id: str, blur_person_ids: list[str]) -> None:
    _set(job_id, status="rendering", progress=0.0)
    _emit(job_id, {"phase": "rendering", "progress": 0.0})
    loop = asyncio.get_running_loop()

    def cb(p: float) -> None:
        _set(job_id, progress=p)
        loop.call_soon_threadsafe(_emit, job_id, {"phase": "rendering", "progress": p})

    try:
        await asyncio.to_thread(render.run, job_id, blur_person_ids, cb)
        _set(job_id, status="done", progress=1.0)
        _emit(job_id, {"phase": "done", "download_url": f"/api/jobs/{job_id}/download"})
    except Exception as e:
        traceback.print_exc()
        _set(job_id, status="error", error=str(e))
        _emit(job_id, {"phase": "error", "message": str(e)})


def delete(job_id: str) -> None:
    _jobs.pop(job_id, None)
    _tasks.pop(job_id, None)
    _subscribers.pop(job_id, None)
    storage.delete_job(job_id)
