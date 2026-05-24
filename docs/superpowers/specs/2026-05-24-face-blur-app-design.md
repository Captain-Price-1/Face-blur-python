# Face Blur Web App — Design Spec

**Date:** 2026-05-24
**Status:** Approved for planning
**Scope:** MVP. Local-only single-user web app. Designed to be scaled later (deploy, mobile clients) without rework of the API surface.

---

## 1. Goal

A local web application that lets the user:

1. Upload a video.
2. See thumbnails of each unique person detected in it.
3. Tick which people to blur.
4. Get back an MP4 with those people's faces blurred (Gaussian blur with feathered oval mask), audio preserved.
5. Download the result.

Target: videos up to ~5 minutes / ~1 GB. Single user, single machine for v1.

---

## 2. Architecture

```
┌──────────────────────────────────┐         ┌────────────────────────────────────────┐
│ Browser (vanilla HTML+JS)        │  HTTP   │ FastAPI server                          │
│                                  │  +WS    │                                         │
│  1. Upload page                  │ ──────► │ /api/jobs               (POST upload)   │
│  2. People-picker page           │ ◄────── │ /api/jobs/{id}/people   (GET thumbs)    │
│  3. Processing/progress page     │  WS     │ /api/jobs/{id}/render   (POST selection)│
│  4. Result/download page         │ ──────► │ /api/jobs/{id}/status   (WS progress)   │
│                                  │         │ /api/jobs/{id}/download (GET output)    │
└──────────────────────────────────┘         └─────────────┬──────────────────────────┘
                                                           │
                                                           ▼
                                             ┌──────────────────────────┐
                                             │ Worker (asyncio task)    │
                                             │  • analyze() phase       │
                                             │  • render()  phase       │
                                             └──────────────┬───────────┘
                                                            ▼
                                             ┌──────────────────────────┐
                                             │ Processing pipeline      │
                                             │  ffmpeg → MediaPipe →    │
                                             │  face_recognition →      │
                                             │  cluster → OpenCV blur → │
                                             │  ffmpeg encode + audio   │
                                             └──────────────────────────┘
```

Key decisions:

- **Two-phase job**: *analyze* (detect + cluster faces → return thumbnails) then *render* (apply blur to selected people, encode). Selection sits between phases.
- **Storage**: a `jobs/{job_id}/` directory on disk. Per-job JSON file is source of truth. No database in MVP.
- **Audio**: extracted by `ffmpeg` up front, muxed back in at the end (OpenCV writers strip audio).
- **Worker**: `asyncio.create_task` background coroutine for v1. Swap for Celery/RQ + Redis when scaling, without changing the API.
- **No auth, no DB, no queue, no Docker, no CI.** This is an MVP.

---

## 3. Processing pipeline

### 3a. Analyze phase

```
input.mp4
   │
   ├─► ffmpeg: extract audio.aac
   │
   └─► ffmpeg: probe (fps, duration, resolution)
              │
              ▼
   ┌───────────────────────────────────┐
   │ Frame loop (OpenCV VideoCapture)  │
   │                                   │
   │  Sample every Nth frame           │
   │    │                              │
   │    ▼                              │
   │  MediaPipe FaceDetection          │  ← fast, gives bbox + score
   │    │                              │
   │    ▼ (per face)                   │
   │  Crop + align                     │
   │    │                              │
   │    ▼                              │
   │  face_recognition.encodings()     │  ← 128-d dlib embedding
   │    │                              │
   │    ▼                              │
   │  Append (frame_idx, bbox, embed,  │
   │           thumb_jpg) to track     │
   └───────────────┬───────────────────┘
                   ▼
   ┌───────────────────────────────────┐
   │ Cluster embeddings → people       │
   │  DBSCAN(eps=0.5, metric=cosine)   │
   │  → person_id per detection        │
   └───────────────┬───────────────────┘
                   ▼
   For each person_id:
     • pick best thumbnail (highest score, largest bbox)
     • save thumbs/{person_id}.jpg
     • save bbox-per-frame timeline → analysis.json
```

**Sampling strategy:** detect on every Nth frame (e.g. every 3rd at 30 fps = ~10 detections/sec). Render phase interpolates bboxes for intermediate frames. ~3× speed-up over per-frame detection with no visible quality loss for blur.

**Why these libraries:**

| Choice | Why |
|---|---|
| MediaPipe `FaceDetection` (not FaceMesh) | We only need bboxes; FaceMesh is overkill and slower. |
| face_recognition for embeddings only | dlib's 128-d vectors cluster cleanly with DBSCAN. We do *not* use its built-in detector — MediaPipe is faster. |
| DBSCAN, not k-means | Number of people unknown; DBSCAN handles that and rejects noisy detections as outliers. |
| Sample + interpolate | ~3× speed-up, no visible quality loss for blur. |

**Output of analyze phase:**

- `jobs/{id}/thumbs/{person_id}.jpg`
- `jobs/{id}/analysis.json`:

```json
{
  "fps": 30,
  "duration_sec": 187.3,
  "people": [
    { "id": "p1", "thumb": "thumbs/p1.jpg", "frame_count": 412, "first_seen_sec": 0.3 },
    { "id": "p2", "thumb": "thumbs/p2.jpg", "frame_count": 95,  "first_seen_sec": 12.1 }
  ],
  "timeline": [
    { "frame": 0,  "faces": [{ "person_id": "p1", "bbox": [120, 80, 140, 180] }] },
    { "frame": 3,  "faces": [{ "person_id": "p1", "bbox": [122, 82, 140, 180] }] }
  ]
}
```

### 3b. Render phase

Triggered by `POST /api/jobs/{id}/render` with `{ "blur_person_ids": ["p1", "p3"] }`. Uses cached `analysis.json` — no re-detection.

```
analysis.json + input.mp4 + blur_person_ids
   │
   ▼
┌─────────────────────────────────────────────┐
│ Frame loop (OpenCV VideoCapture, every frame)│
│                                             │
│  For frame_idx in [0..N]:                   │
│    bboxes = interpolate(timeline, frame_idx)│
│    for (person_id, bbox) in bboxes:         │
│       if person_id in blur_person_ids:      │
│          apply_gaussian_blur(frame, bbox)   │
│    writer.write(frame)                      │
│                                             │
│  Emit progress over WebSocket every ~30 fr  │
└────────────────┬────────────────────────────┘
                 ▼
   video_only.mp4
                 │
                 ▼
┌─────────────────────────────────────────────┐
│ ffmpeg mux:                                 │
│   ffmpeg -i video_only.mp4 -i audio.aac     │
│          -c:v copy -c:a copy output.mp4     │
└────────────────┬────────────────────────────┘
                 ▼
            output.mp4
```

**Blur function:**

```python
def apply_gaussian_blur(frame, bbox, expand=1.25):
    x, y, w, h = expand_bbox(bbox, expand)            # pad ~25% to cover hair/jaw
    roi = frame[y:y+h, x:x+w]
    blurred = cv2.GaussianBlur(roi, (0, 0), sigmaX=max(w, h) / 8)
    mask = make_oval_mask(w, h, feather_px=int(min(w, h) * 0.15))
    frame[y:y+h, x:x+w] = blend(roi, blurred, mask)
```

**Interpolation rule:** linear interpolation between sampled-frame bboxes for the same `person_id`. If the gap to the nearest detection is > 0.5 s, do not blur — better a brief un-blur than a blur floating over empty space.

**Encoding:** OpenCV writes `video_only.mp4` with `cv2.VideoWriter_fourcc(*'mp4v')`. ffmpeg then muxes the original audio with `-c:v copy -c:a copy` (no re-encode of the video stream → fast, lossless second step).

**Re-render is cheap.** Changing the selection re-runs only this phase; detection results stay cached.

---

## 4. API surface

REST + one WebSocket. Designed so a future mobile client uses the same endpoints.

| Method | Path | Purpose | Body / Response |
|---|---|---|---|
| `POST` | `/api/jobs` | Upload video, start analyze | multipart `file=<video>` → `{ job_id, status: "analyzing" }` |
| `GET` | `/api/jobs/{id}` | Job status snapshot | `{ status, progress, error? }` |
| `GET` | `/api/jobs/{id}/people` | People found in analyze | `{ people: [{ id, thumb_url, frame_count, first_seen_sec }] }` |
| `GET` | `/api/jobs/{id}/thumbs/{person_id}` | Thumbnail JPG | image/jpeg |
| `POST` | `/api/jobs/{id}/render` | Kick off render | `{ blur_person_ids: ["p1","p3"] }` → `{ status: "rendering" }` |
| `GET` | `/api/jobs/{id}/download` | Final blurred video | video/mp4 |
| `DELETE` | `/api/jobs/{id}` | Clean up | `{ ok: true }` |
| `WS`     | `/api/jobs/{id}/events` | Live progress | streamed JSON events |

**Status values:** `analyzing`, `awaiting_selection`, `rendering`, `done`, `error`.

**WebSocket events:**

```json
{ "phase": "analyzing",  "progress": 0.42, "message": "detected 3 people so far" }
{ "phase": "awaiting_selection" }
{ "phase": "rendering",  "progress": 0.78 }
{ "phase": "done",       "download_url": "/api/jobs/abc/download" }
{ "phase": "error",      "message": "ffmpeg failed: ..." }
```

**Job state machine:**

```
created ─► analyzing ─► awaiting_selection ─► rendering ─► done
              │                                  │
              └──────────── error ◄──────────────┘
```

**Behaviour notes:**

- No auth in v1. Localhost only.
- Job IDs are server-generated UUIDs.
- Files are served by the API, not via a static mount — single seam (`storage.py`) to swap disk for S3 later.
- WebSocket is optional; client can fall back to polling `GET /api/jobs/{id}` every 1 s.
- `POST /render` while rendering is in progress is idempotent: returns current status, doesn't start a second job.

---

## 5. Frontend (vanilla HTML+JS)

Four pages, each its own HTML file. State lives in `?job=<uuid>` URL param — refresh-safe.

```
  ┌──────────────────────┐
  │  1. /                │   Upload
  │  [ choose file ]     │   POST /api/jobs
  │  [   Upload   ]      │   → /people?job=<id>
  └──────────────────────┘
              │
              ▼
  ┌──────────────────────┐
  │ 2. /people?job=<id>  │   Picker
  │  Analyzing… 42%      │   WS progress
  │  Faces found:        │   GET /api/jobs/<id>/people
  │   👤 👤 👤           │   Tap to toggle
  │      [  Next  ]      │   POST /api/jobs/<id>/render
  └──────────────────────┘   → /processing?job=<id>
              │
              ▼
  ┌──────────────────────┐
  │ 3. /processing?job=  │   Render progress
  │  Rendering… 78%      │   WS progress
  │                      │   auto-redirect when done
  └──────────────────────┘
              │
              ▼
  ┌──────────────────────┐
  │ 4. /done?job=<id>    │   Download
  │  ✓ Done              │
  │  <video preview>     │   src = /api/jobs/<id>/download
  │  [ ⬇ Download MP4 ]  │
  │  [  Start over   ]   │   DELETE /api/jobs/<id>
  └──────────────────────┘
```

**UX rules:**

- Default: nothing selected = nothing blurred. User must opt people in.
- Selected-to-blur faces show a visible blurred-tile overlay.
- "Start over" calls `DELETE /api/jobs/{id}` then returns to `/`.

**JS modules (no bundler):**

- `static/js/api.js` — REST wrappers (single place to swap base URL for mobile).
- `static/js/progress.js` — WebSocket with 1 s polling fallback.
- `static/js/pages/{upload,people,processing,done}.js` — one per page.

---

## 6. Project layout

```
face-blur-app/
├── README.md
├── requirements.txt
├── app/
│   ├── __init__.py
│   ├── main.py                     # FastAPI app + all routes
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── analyze.py
│   │   ├── render.py
│   │   └── blur.py
│   ├── storage.py                  # disk I/O — single seam for S3 later
│   └── jobs.py                     # in-memory registry + asyncio worker
├── static/
│   ├── index.html
│   ├── people.html
│   ├── processing.html
│   ├── done.html
│   └── js/
│       ├── api.js
│       └── progress.js
├── jobs/                           # runtime, gitignored
│   └── <uuid>/
│       ├── input.mp4
│       ├── audio.aac
│       ├── analysis.json
│       ├── thumbs/<person_id>.jpg
│       └── output.mp4
└── tests/
    ├── fixtures/sample_5s.mp4
    ├── test_pipeline.py
    └── test_api.py
```

---

## 7. Testing

Two files. No mocks of OpenCV / MediaPipe — too brittle for a video pipeline. Real fixture, real ffmpeg, real outputs.

1. **`test_pipeline.py`** — runs `analyze()` then `render()` on a 5-second 2-person fixture clip. Asserts:
   - Exactly 2 people clustered.
   - `output.mp4` exists, non-empty, playable (ffprobe succeeds).
   - Blurred ROI variance below a threshold (blur actually applied).

2. **`test_api.py`** — FastAPI `TestClient`, walks the happy path: upload → poll until `awaiting_selection` → `GET /people` → `POST /render` → poll until `done` → `GET /download`.

---

## 8. Dependencies

**Python:**

```
fastapi
uvicorn[standard]
python-multipart
opencv-python
mediapipe
face_recognition          # requires dlib (cmake + a few minutes to build on macOS)
numpy
scikit-learn              # DBSCAN
ffmpeg-python             # thin wrapper around subprocess
pytest                    # dev
```

**System:** `ffmpeg` (`brew install ffmpeg`).

---

## 9. Explicitly out of MVP scope

- Authentication, user accounts, multi-user.
- Database, Redis, Celery/RQ.
- Docker, docker-compose, CI/CD.
- Blur style picker (Gaussian only).
- Re-encode options, bitrate/resolution choices.
- Mobile client (React Native / Flutter). The API is *designed* for it but not built.
- Cloud storage (S3). `storage.py` is the seam to add it later.
- Long videos (> 5 min) or batch processing.

These are listed so they don't sneak back in during implementation.

---

## 10. Future scaling path (not in scope, for orientation only)

When/if the MVP graduates:

1. **Mobile client.** Same API endpoints; build React Native / Flutter that talks to the deployed FastAPI.
2. **Background worker.** Replace `asyncio.create_task` with Celery + Redis. Job state moves from in-memory dict to Redis.
3. **Cloud storage.** Swap `storage.py` from disk to S3. No other code changes.
4. **Auth.** Add API key / JWT middleware in FastAPI. Endpoint shapes unchanged.
5. **GPU.** MediaPipe runs CPU fine for short videos; for long ones, switch to a CUDA-accelerated detector.

The MVP's job is to make all five of these *possible* without a rewrite, not to do any of them.
