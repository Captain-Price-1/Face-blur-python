"""In-memory job registry + background-thread worker."""
from __future__ import annotations

import asyncio
import threading
import traceback
from typing import Any

from app import downloader, storage
from app.pipeline import analyze, blur_all, render

_jobs: dict[str, dict[str, Any]] = {}
_threads: dict[str, threading.Thread] = {}
_subscribers: dict[str, list[asyncio.Queue]] = {}
_lock = threading.Lock()


def reset() -> None:
    """Test helper: wipe in-memory state."""
    with _lock:
        _jobs.clear()
        _threads.clear()
        _subscribers.clear()


def create() -> str:
    job_id = storage.new_job()
    with _lock:
        _jobs[job_id] = {"status": "created", "progress": 0.0}
    return job_id


def get(job_id: str) -> dict[str, Any]:
    with _lock:
        return dict(_jobs.get(job_id, {"status": "unknown"}))


def subscribe(job_id: str) -> asyncio.Queue:
    q: asyncio.Queue = asyncio.Queue()
    with _lock:
        _subscribers.setdefault(job_id, []).append(q)
    return q


def _emit(job_id: str, event: dict[str, Any]) -> None:
    # Best-effort: put event on queues thread-safely.
    for q in _subscribers.get(job_id, []):
        try:
            loop = q._loop if hasattr(q, "_loop") else None  # CPython internal
            if loop is not None and loop.is_running():
                loop.call_soon_threadsafe(q.put_nowait, event)
            else:
                q.put_nowait(event)
        except Exception:
            pass


def _set(job_id: str, **fields) -> None:
    with _lock:
        _jobs[job_id].update(fields)


def start_analyze(job_id: str) -> None:
    with _lock:
        existing = _threads.get(job_id)
        if existing is not None and existing.is_alive():
            return
        t = threading.Thread(target=_run_analyze, args=(job_id,), daemon=True)
        _threads[job_id] = t
    t.start()


def start_download_and_analyze(job_id: str, url: str) -> None:
    with _lock:
        existing = _threads.get(job_id)
        if existing is not None and existing.is_alive():
            return
        t = threading.Thread(
            target=_run_download_and_analyze, args=(job_id, url), daemon=True
        )
        _threads[job_id] = t
    t.start()


def start_blur_all(job_id: str) -> None:
    with _lock:
        existing = _threads.get(job_id)
        if existing is not None and existing.is_alive():
            return
        t = threading.Thread(target=_run_blur_all, args=(job_id,), daemon=True)
        _threads[job_id] = t
    t.start()


def start_download_and_blur_all(job_id: str, url: str) -> None:
    with _lock:
        existing = _threads.get(job_id)
        if existing is not None and existing.is_alive():
            return
        t = threading.Thread(
            target=_run_download_and_blur_all, args=(job_id, url), daemon=True
        )
        _threads[job_id] = t
    t.start()


def start_render(job_id: str, blur_person_ids: list[str]) -> None:
    with _lock:
        existing = _threads.get(job_id)
        if existing is not None and existing.is_alive():
            return
        t = threading.Thread(target=_run_render, args=(job_id, blur_person_ids), daemon=True)
        _threads[job_id] = t
    t.start()


def _run_download_and_analyze(job_id: str, url: str) -> None:
    _set(job_id, status="downloading", progress=0.0)
    _emit(job_id, {"phase": "downloading", "progress": 0.0})

    def dl_cb(p: float) -> None:
        _set(job_id, progress=p)
        _emit(job_id, {"phase": "downloading", "progress": p})

    try:
        downloader.download(url, storage.input_path(job_id), dl_cb)
    except Exception as e:
        traceback.print_exc()
        _set(job_id, status="error", error=f"download failed: {e}")
        _emit(job_id, {"phase": "error", "message": f"download failed: {e}"})
        return

    _run_analyze(job_id)


def _run_blur_all(job_id: str) -> None:
    _set(job_id, status="blurring", progress=0.0)
    _emit(job_id, {"phase": "blurring", "progress": 0.0})

    def cb(p: float) -> None:
        _set(job_id, progress=p)
        _emit(job_id, {"phase": "blurring", "progress": p})

    try:
        blur_all.run(job_id, cb)
        _set(job_id, status="done", progress=1.0)
        _emit(job_id, {"phase": "done", "download_url": f"/api/jobs/{job_id}/download"})
    except Exception as e:
        traceback.print_exc()
        _set(job_id, status="error", error=str(e))
        _emit(job_id, {"phase": "error", "message": str(e)})


def _run_download_and_blur_all(job_id: str, url: str) -> None:
    _set(job_id, status="downloading", progress=0.0)
    _emit(job_id, {"phase": "downloading", "progress": 0.0})

    def dl_cb(p: float) -> None:
        _set(job_id, progress=p)
        _emit(job_id, {"phase": "downloading", "progress": p})

    try:
        downloader.download(url, storage.input_path(job_id), dl_cb)
    except Exception as e:
        traceback.print_exc()
        _set(job_id, status="error", error=f"download failed: {e}")
        _emit(job_id, {"phase": "error", "message": f"download failed: {e}"})
        return

    _run_blur_all(job_id)


def _run_analyze(job_id: str) -> None:
    _set(job_id, status="analyzing", progress=0.0)
    _emit(job_id, {"phase": "analyzing", "progress": 0.0})

    def cb(p: float) -> None:
        _set(job_id, progress=p)
        _emit(job_id, {"phase": "analyzing", "progress": p})

    try:
        analyze.run(job_id, cb)
        _set(job_id, status="awaiting_selection", progress=1.0)
        _emit(job_id, {"phase": "awaiting_selection"})
    except Exception as e:
        traceback.print_exc()
        _set(job_id, status="error", error=str(e))
        _emit(job_id, {"phase": "error", "message": str(e)})


def _run_render(job_id: str, blur_person_ids: list[str]) -> None:
    _set(job_id, status="rendering", progress=0.0)
    _emit(job_id, {"phase": "rendering", "progress": 0.0})

    def cb(p: float) -> None:
        _set(job_id, progress=p)
        _emit(job_id, {"phase": "rendering", "progress": p})

    try:
        render.run(job_id, blur_person_ids, cb)
        _set(job_id, status="done", progress=1.0)
        _emit(job_id, {"phase": "done", "download_url": f"/api/jobs/{job_id}/download"})
    except Exception as e:
        traceback.print_exc()
        _set(job_id, status="error", error=str(e))
        _emit(job_id, {"phase": "error", "message": str(e)})


def delete(job_id: str) -> None:
    with _lock:
        _jobs.pop(job_id, None)
        _threads.pop(job_id, None)
        _subscribers.pop(job_id, None)
    storage.delete_job(job_id)
