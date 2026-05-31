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

## Use it on your phone (same Wi‑Fi)

The hosted GitHub Pages page is UI-only. To actually process videos on a phone,
run the backend on your computer so the phone can reach it over Wi‑Fi:

```bash
./run.sh
```

This binds to your network and prints a phone URL, e.g. `http://192.168.1.5:8000`.
On your phone's browser (same Wi‑Fi), open that URL — the full app loads and
works directly from your computer. No cloud, no extra config.

Notes:
- Open the `http://<computer-ip>:8000` URL **directly** on the phone. Don't use
  the `github.io` page for processing — an HTTPS page can't call a plain‑http
  computer on your LAN (browser "mixed content" block).
- Your computer must stay on with `./run.sh` running. Works on home Wi‑Fi only.

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
