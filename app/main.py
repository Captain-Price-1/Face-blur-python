"""FastAPI app: REST endpoints. WebSocket added in Task 12, static mount in Task 14."""
from __future__ import annotations

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import jobs, storage

app = FastAPI(title="Face Blur App", version="0.1.0")

# Permissive CORS: this is a local single-user tool, and the GitHub Pages
# showcase frontend (a different origin) must be able to call a backend the
# user runs locally. No cookies/credentials are used.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class RenderRequest(BaseModel):
    blur_person_ids: list[str]
    blur_mode: str = "face"


class UrlRequest(BaseModel):
    url: str
    blur_all: bool = False


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


@app.post("/api/jobs")
async def create_job(
    file: UploadFile = File(...),
    blur_all: bool = Form(False),
) -> dict:
    job_id = jobs.create()
    target = storage.input_path(job_id)
    with target.open("wb") as fh:
        while chunk := await file.read(1 << 20):
            fh.write(chunk)
    if blur_all:
        jobs.start_blur_all(job_id)
        return {"job_id": job_id, "status": "blurring"}
    jobs.start_analyze(job_id)
    return {"job_id": job_id, "status": "analyzing"}


@app.post("/api/jobs/from-url")
def create_job_from_url(body: UrlRequest) -> dict:
    url = body.url.strip()
    if not url:
        raise HTTPException(400, "url is required")
    job_id = jobs.create()
    if body.blur_all:
        jobs.start_download_and_blur_all(job_id, url)
    else:
        jobs.start_download_and_analyze(job_id, url)
    return {"job_id": job_id, "status": "downloading"}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    state = jobs.get(job_id)
    return {
        "status": state.get("status", "unknown"),
        "progress": state.get("progress", 0.0),
        "error": state.get("error"),
    }


@app.get("/api/jobs/{job_id}/people")
def get_people(job_id: str) -> dict:
    if jobs.get(job_id).get("status") not in ("awaiting_selection", "rendering", "done"):
        raise HTTPException(409, "analysis not ready")
    data = storage.read_analysis(job_id)
    return {
        "people": [
            {
                "id": p["id"],
                "thumb_url": f"/api/jobs/{job_id}/thumbs/{p['id']}",
                "frame_count": p["frame_count"],
                "first_seen_sec": p["first_seen_sec"],
            }
            for p in data["people"]
        ]
    }


@app.get("/api/jobs/{job_id}/thumbs/{person_id}")
def get_thumb(job_id: str, person_id: str) -> FileResponse:
    path = storage.thumb_path(job_id, person_id)
    if not path.exists():
        raise HTTPException(404, "thumb not found")
    return FileResponse(path, media_type="image/jpeg")


@app.post("/api/jobs/{job_id}/render")
def start_render(job_id: str, body: RenderRequest) -> dict:
    state = jobs.get(job_id)
    if state.get("status") == "unknown":
        raise HTTPException(404, "job not found")
    if body.blur_mode not in {"face", "body_box", "body_silhouette"}:
        raise HTTPException(400, "invalid blur_mode")
    jobs.start_render(job_id, body.blur_person_ids, body.blur_mode)
    return {"status": "rendering"}


@app.get("/api/jobs/{job_id}/download")
def download(job_id: str) -> FileResponse:
    path = storage.output_path(job_id)
    if not path.exists():
        raise HTTPException(404, "output not found")
    return FileResponse(path, media_type="video/mp4", filename="blurred.mp4")


@app.delete("/api/jobs/{job_id}")
def delete(job_id: str) -> dict:
    jobs.delete(job_id)
    return {"ok": True}


@app.websocket("/api/jobs/{job_id}/events")
async def events(ws: WebSocket, job_id: str) -> None:
    await ws.accept()
    queue = jobs.subscribe(job_id)
    state = jobs.get(job_id)
    if state.get("status") and state["status"] != "unknown":
        await ws.send_json({"phase": state["status"], "progress": state.get("progress", 0.0)})
    try:
        while True:
            ev = await queue.get()
            await ws.send_json(ev)
            if ev.get("phase") in ("done", "error"):
                break
    except WebSocketDisconnect:
        pass


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def root() -> RedirectResponse:
    return RedirectResponse(url="/static/index.html")
