# Face Blur App

Local web application: upload a video, pick which detected people to blur, get back the same video with their faces Gaussian-blurred. Audio preserved.

## Stack

- **Backend:** FastAPI (Python 3.11+)
- **Detection:** OpenCV YuNet (ONNX, ~227 KB, bundled in `models/`)
- **Re-identification:** face_recognition (dlib 128-d embeddings) + DBSCAN
- **Frame ops:** OpenCV
- **Encoding/audio mux:** ffmpeg (system binary)
- **Frontend:** Vanilla HTML + JS (no build step)

## Setup (macOS)

```bash
brew install ffmpeg cmake
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`face_recognition` builds dlib from source the first time; expect 2–5 minutes.

## Run

```bash
uvicorn app.main:app --reload
# open http://localhost:8000
```

## Test

```bash
pytest
```

The pipeline tests need `tests/fixtures/sample_5s.mp4` — a 5-second clip with at least one face. One is committed; replace with your own if you want different coverage.

## Project layout

- `app/main.py` — FastAPI routes (REST + WebSocket)
- `app/jobs.py` — In-memory job registry + background worker threads
- `app/pipeline/` — Detection (YuNet), embeddings (dlib), clustering, blur, render
- `app/storage.py` — Disk I/O (single seam for future S3 swap)
- `app/ffmpeg_utils.py` — ffmpeg/ffprobe subprocess wrappers
- `models/` — YuNet ONNX detector
- `static/` — Frontend HTML + JS
- `scripts/quick_blur.py` — One-off CLI that blurs every detected face in a video
- `jobs/` — Per-job runtime directory (gitignored)

## Scope

MVP — single user, single process, local-only. No DB, no auth, no queue.

For the scaling path (mobile client, cloud storage, background workers) see [`docs/superpowers/specs/2026-05-24-face-blur-app-design.md`](docs/superpowers/specs/2026-05-24-face-blur-app-design.md).
