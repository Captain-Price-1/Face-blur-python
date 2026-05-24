# Face Blur App Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Local web app where the user uploads a video, picks which detected people to blur, and downloads the result with Gaussian-blurred faces and audio preserved.

**Architecture:** FastAPI single-process backend serving a 4-page vanilla HTML/JS frontend. Two-phase per-job pipeline (analyze → render) with in-memory job registry and an asyncio worker. All job state on disk under `jobs/{uuid}/`. No DB, no auth, no queue, no Docker.

**Tech Stack:** Python 3.11+, FastAPI, uvicorn, OpenCV (`opencv-python`), MediaPipe, `face_recognition` (dlib), scikit-learn (DBSCAN), ffmpeg (system binary), pytest.

**Spec:** [`docs/superpowers/specs/2026-05-24-face-blur-app-design.md`](../specs/2026-05-24-face-blur-app-design.md). Read it before starting.

---

## File Structure

Created across this plan:

```
face-blur-app/
├── .gitignore
├── README.md                       # Task 15
├── requirements.txt                # Task 1
├── app/
│   ├── __init__.py                 # Task 1
│   ├── main.py                     # Task 11 (REST), Task 12 (WS), Task 14 (static mount)
│   ├── storage.py                  # Task 2
│   ├── ffmpeg_utils.py             # Task 3
│   ├── jobs.py                     # Task 10
│   └── pipeline/
│       ├── __init__.py             # Task 5
│       ├── blur.py                 # Task 5
│       ├── detect.py               # Task 6
│       ├── embed_cluster.py        # Task 7
│       ├── analyze.py              # Task 8
│       └── render.py               # Task 9
├── static/
│   ├── index.html                  # Task 13
│   ├── people.html                 # Task 13
│   ├── processing.html             # Task 13
│   ├── done.html                   # Task 13
│   └── js/
│       ├── api.js                  # Task 13
│       └── progress.js             # Task 13
├── jobs/                           # gitignored runtime dir
└── tests/
    ├── conftest.py                 # Task 4
    ├── fixtures/sample_5s.mp4      # Task 4 (manual)
    ├── test_storage.py             # Task 2
    ├── test_ffmpeg_utils.py        # Task 3
    ├── test_blur.py                # Task 5
    ├── test_detect.py              # Task 6
    ├── test_embed_cluster.py       # Task 7
    ├── test_analyze.py             # Task 8
    ├── test_render.py              # Task 9
    ├── test_jobs.py                # Task 10
    ├── test_api.py                 # Task 11, 12
    └── test_e2e.py                 # Task 14
```

**Responsibility per file:**

| File | Responsibility |
|---|---|
| `storage.py` | Single seam for disk I/O. All other code calls it; never opens paths directly. Lets us swap to S3 later. |
| `ffmpeg_utils.py` | All `subprocess` calls to the `ffmpeg` binary (probe, extract audio, mux). |
| `pipeline/blur.py` | One pure function: blur a bbox in a frame. |
| `pipeline/detect.py` | One pure function: MediaPipe face detection on a frame → list of bboxes + scores. |
| `pipeline/embed_cluster.py` | Two pure functions: compute 128-d embedding from a face crop; cluster a list of embeddings with DBSCAN. |
| `pipeline/analyze.py` | Orchestrates detect + embed + cluster across a video. Produces `analysis.json` + thumbs. |
| `pipeline/render.py` | Orchestrates interpolation + blur + encode + mux. Produces `output.mp4`. |
| `jobs.py` | In-memory job registry + asyncio worker that runs `analyze()` → awaits selection → runs `render()`. Emits progress events. |
| `main.py` | FastAPI app: REST endpoints + WebSocket + static file mount. Thin — delegates everything to the modules above. |

---

## Task 1: Project skeleton + FastAPI hello

**Files:**
- Create: `requirements.txt`, `.gitignore`, `app/__init__.py`, `app/main.py`, `README.md`

- [ ] **Step 1: Create `.gitignore`**

```
__pycache__/
*.pyc
.venv/
venv/
.pytest_cache/
.DS_Store
jobs/
*.egg-info/
```

- [ ] **Step 2: Create `requirements.txt`**

```
fastapi==0.115.*
uvicorn[standard]==0.32.*
python-multipart==0.0.*
opencv-python==4.10.*
mediapipe==0.10.*
face_recognition==1.3.*
numpy==1.26.*
scikit-learn==1.5.*
pytest==8.*
httpx==0.27.*
```

Note: `ffmpeg-python` not listed — we use `subprocess` directly (one dep less). `httpx` is for FastAPI's `TestClient`.

- [ ] **Step 3: Create `app/__init__.py`** (empty file)

- [ ] **Step 4: Create `app/main.py`**

```python
from fastapi import FastAPI

app = FastAPI(title="Face Blur App", version="0.1.0")


@app.get("/api/health")
def health():
    return {"ok": True}
```

- [ ] **Step 5: Create `README.md`**

```markdown
# Face Blur App

Local web app to blur selected faces in uploaded videos.

## Setup

```bash
brew install ffmpeg cmake     # cmake is needed by dlib
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
uvicorn app.main:app --reload
# open http://localhost:8000
```

## Test

```bash
pytest
```
```

- [ ] **Step 6: Install and verify the server runs**

Run:
```bash
python3.11 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
uvicorn app.main:app --port 8000 &
sleep 2
curl -s http://localhost:8000/api/health
kill %1
```

Expected output: `{"ok":true}`

- [ ] **Step 7: Commit**

```bash
git add .gitignore requirements.txt app/ README.md
git commit -m "feat: project skeleton with FastAPI hello endpoint"
```

---

## Task 2: Storage module (TDD)

**Files:**
- Create: `app/storage.py`, `tests/__init__.py`, `tests/test_storage.py`

**Why:** All other modules talk to disk through this one. Future S3 swap is a one-file change.

- [ ] **Step 1: Write the failing tests**

Create `tests/__init__.py` (empty) and `tests/test_storage.py`:

```python
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
```

- [ ] **Step 2: Run tests — verify they fail**

Run: `pytest tests/test_storage.py -v`
Expected: all fail with `ModuleNotFoundError` / `AttributeError` (storage module empty).

- [ ] **Step 3: Implement `app/storage.py`**

```python
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
```

- [ ] **Step 4: Run tests — verify they pass**

Run: `pytest tests/test_storage.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add app/storage.py tests/__init__.py tests/test_storage.py
git commit -m "feat: storage module with job dir lifecycle"
```

---

## Task 3: ffmpeg utilities (TDD)

**Files:**
- Create: `app/ffmpeg_utils.py`, `tests/test_ffmpeg_utils.py`

**Why:** Isolate every `subprocess` call to ffmpeg so the rest of the code stays clean.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ffmpeg_utils.py`:

```python
import subprocess
from pathlib import Path

import pytest

from app import ffmpeg_utils


def _make_silent_mp4(path: Path, seconds: int = 2) -> None:
    """Use ffmpeg to synthesize a tiny mp4 for testing."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c=black:s=320x240:r=10:d={seconds}",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
            "-c:v", "libx264", "-c:a", "aac", "-shortest", str(path),
        ],
        check=True, capture_output=True,
    )


def _make_silent_mp4_no_audio(path: Path, seconds: int = 2) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c=black:s=320x240:r=10:d={seconds}",
            "-c:v", "libx264", str(path),
        ],
        check=True, capture_output=True,
    )


def test_probe_returns_fps_duration_resolution(tmp_path):
    src = tmp_path / "src.mp4"
    _make_silent_mp4(src, seconds=2)
    info = ffmpeg_utils.probe(src)
    assert round(info["fps"]) == 10
    assert 1.5 < info["duration_sec"] < 2.5
    assert info["width"] == 320
    assert info["height"] == 240
    assert info["has_audio"] is True


def test_probe_detects_no_audio(tmp_path):
    src = tmp_path / "src.mp4"
    _make_silent_mp4_no_audio(src, seconds=1)
    info = ffmpeg_utils.probe(src)
    assert info["has_audio"] is False


def test_extract_audio_writes_file(tmp_path):
    src = tmp_path / "src.mp4"
    dst = tmp_path / "audio.m4a"
    _make_silent_mp4(src, seconds=2)
    ffmpeg_utils.extract_audio(src, dst)
    assert dst.exists() and dst.stat().st_size > 0


def test_mux_combines_video_and_audio(tmp_path):
    src = tmp_path / "src.mp4"
    audio = tmp_path / "audio.m4a"
    video_only = tmp_path / "video_only.mp4"
    output = tmp_path / "output.mp4"
    _make_silent_mp4(src, seconds=2)
    ffmpeg_utils.extract_audio(src, audio)
    # Pretend we re-encoded the video
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-an", "-c:v", "libx264", str(video_only)],
        check=True, capture_output=True,
    )
    ffmpeg_utils.mux(video_only, audio, output)
    assert output.exists()
    info = ffmpeg_utils.probe(output)
    assert info["has_audio"] is True
```

- [ ] **Step 2: Run tests — verify they fail**

Run: `pytest tests/test_ffmpeg_utils.py -v`
Expected: fail (module empty).

- [ ] **Step 3: Implement `app/ffmpeg_utils.py`**

```python
"""Thin wrappers around the `ffmpeg` and `ffprobe` system binaries."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, capture_output=True)


def probe(path: Path) -> dict:
    """Return {fps, duration_sec, width, height, has_audio}."""
    cmd = [
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_streams", "-show_format", str(path),
    ]
    out = _run(cmd).stdout
    data = json.loads(out)
    video = next(s for s in data["streams"] if s["codec_type"] == "video")
    audio = next((s for s in data["streams"] if s["codec_type"] == "audio"), None)
    num, den = (int(x) for x in video["r_frame_rate"].split("/"))
    fps = num / den if den else 0.0
    return {
        "fps": fps,
        "duration_sec": float(data["format"]["duration"]),
        "width": int(video["width"]),
        "height": int(video["height"]),
        "has_audio": audio is not None,
    }


def extract_audio(src: Path, dst: Path) -> None:
    """Copy the source audio stream to `dst` without re-encoding."""
    _run(["ffmpeg", "-y", "-i", str(src), "-vn", "-c:a", "copy", str(dst)])


def mux(video_only: Path, audio: Path, dst: Path) -> None:
    """Combine a video-only file with an audio file. No re-encoding."""
    _run([
        "ffmpeg", "-y", "-i", str(video_only), "-i", str(audio),
        "-c:v", "copy", "-c:a", "copy", str(dst),
    ])
```

- [ ] **Step 4: Run tests — verify they pass**

Run: `pytest tests/test_ffmpeg_utils.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add app/ffmpeg_utils.py tests/test_ffmpeg_utils.py
git commit -m "feat: ffmpeg/ffprobe wrappers for probe, extract-audio, mux"
```

---

## Task 4: Test fixture video

**Files:**
- Create: `tests/conftest.py`, `tests/fixtures/sample_5s.mp4`
- Modify: `.gitignore` (allow fixture)

**Why:** Later tests need a real video with two distinguishable faces. We commit a small fixture (≤ 1 MB) to the repo.

- [ ] **Step 1: Source the fixture**

Manual step. Acquire a 5-second clip showing exactly two distinct people, 320×240 or smaller, no third face. Place at `tests/fixtures/sample_5s.mp4`. Compress to ≤ 1 MB with:

```bash
ffmpeg -y -i input.mov -t 5 -vf "scale=320:-2,fps=10" -c:v libx264 -crf 30 \
  -c:a aac -b:a 64k tests/fixtures/sample_5s.mp4
ls -lh tests/fixtures/sample_5s.mp4
```

Acceptable sources: a clip you own; a Creative Commons clip from Pexels/Pixabay. Do NOT use anything copyrighted.

- [ ] **Step 2: Update `.gitignore` to allow the fixture**

Add at the bottom of `.gitignore`:

```
!tests/fixtures/sample_5s.mp4
```

- [ ] **Step 3: Add a sanity test**

Create `tests/conftest.py`:

```python
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_video() -> Path:
    p = FIXTURES / "sample_5s.mp4"
    if not p.exists():
        pytest.skip("tests/fixtures/sample_5s.mp4 not present — see plan Task 4")
    return p
```

Add a smoke test at the bottom of `tests/test_ffmpeg_utils.py`:

```python
def test_sample_fixture_is_valid(sample_video):
    info = ffmpeg_utils.probe(sample_video)
    assert 4 <= info["duration_sec"] <= 6
    assert info["width"] <= 640
```

- [ ] **Step 4: Run the smoke test**

Run: `pytest tests/test_ffmpeg_utils.py::test_sample_fixture_is_valid -v`
Expected: PASS (or SKIP with a clear message if the fixture wasn't created yet — but for this plan the engineer MUST create it before continuing).

- [ ] **Step 5: Commit**

```bash
git add .gitignore tests/conftest.py tests/fixtures/sample_5s.mp4 tests/test_ffmpeg_utils.py
git commit -m "test: add sample_5s.mp4 fixture and conftest helper"
```

---

## Task 5: Blur primitive (TDD)

**Files:**
- Create: `app/pipeline/__init__.py`, `app/pipeline/blur.py`, `tests/test_blur.py`

- [ ] **Step 1: Write the failing tests**

Create `app/pipeline/__init__.py` (empty) and `tests/test_blur.py`:

```python
import numpy as np

from app.pipeline import blur


def _solid_frame(value: int = 200, h: int = 240, w: int = 320) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.uint8)


def _checkerboard(h: int = 240, w: int = 320, sq: int = 8) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(0, h, sq):
        for x in range(0, w, sq):
            if ((x // sq) + (y // sq)) % 2 == 0:
                img[y:y+sq, x:x+sq] = 255
    return img


def test_blur_reduces_variance_inside_bbox():
    frame = _checkerboard()
    bbox = (100, 60, 80, 80)  # x, y, w, h
    before = frame[60:140, 100:180].var()
    blur.apply_gaussian_blur(frame, bbox)
    after = frame[60:140, 100:180].var()
    assert after < before * 0.5


def test_blur_does_not_change_outside_bbox():
    frame = _checkerboard()
    bbox = (100, 60, 80, 80)
    before_outside = frame[:60, :].copy()
    blur.apply_gaussian_blur(frame, bbox)
    np.testing.assert_array_equal(before_outside, frame[:60, :])


def test_blur_clamps_to_frame_bounds():
    frame = _solid_frame()
    # bbox extending past the right edge — should not raise
    blur.apply_gaussian_blur(frame, (300, 200, 100, 100))


def test_blur_handles_zero_sized_bbox():
    frame = _solid_frame()
    blur.apply_gaussian_blur(frame, (10, 10, 0, 0))  # no-op, no crash
```

- [ ] **Step 2: Run tests — verify they fail**

Run: `pytest tests/test_blur.py -v`
Expected: fail (`blur` module empty).

- [ ] **Step 3: Implement `app/pipeline/blur.py`**

```python
"""Gaussian blur with a feathered oval mask."""
from __future__ import annotations

import cv2
import numpy as np


def _oval_mask(w: int, h: int, feather_px: int) -> np.ndarray:
    """Returns a HxWx1 float32 mask in [0, 1] with feathered oval shape."""
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (w // 2, h // 2), (w // 2, h // 2), 0, 0, 360, 255, -1)
    if feather_px > 0:
        k = max(3, feather_px | 1)  # must be odd
        mask = cv2.GaussianBlur(mask, (k, k), 0)
    return (mask.astype(np.float32) / 255.0)[..., None]


def apply_gaussian_blur(frame: np.ndarray, bbox: tuple[int, int, int, int], expand: float = 1.25) -> None:
    """Blur the bbox region in-place. bbox = (x, y, w, h)."""
    fh, fw = frame.shape[:2]
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return

    cx, cy = x + w / 2, y + h / 2
    w2, h2 = int(w * expand), int(h * expand)
    x2 = max(0, int(cx - w2 / 2))
    y2 = max(0, int(cy - h2 / 2))
    x_end = min(fw, x2 + w2)
    y_end = min(fh, y2 + h2)
    w2, h2 = x_end - x2, y_end - y2
    if w2 <= 0 or h2 <= 0:
        return

    roi = frame[y2:y_end, x2:x_end]
    sigma = max(w2, h2) / 8.0
    blurred = cv2.GaussianBlur(roi, (0, 0), sigmaX=sigma)
    mask = _oval_mask(w2, h2, feather_px=int(min(w2, h2) * 0.15))
    frame[y2:y_end, x2:x_end] = (roi * (1 - mask) + blurred * mask).astype(np.uint8)
```

- [ ] **Step 4: Run tests — verify they pass**

Run: `pytest tests/test_blur.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add app/pipeline/__init__.py app/pipeline/blur.py tests/test_blur.py
git commit -m "feat: gaussian blur with feathered oval mask"
```

---

## Task 6: Face detection wrapper (TDD)

**Files:**
- Create: `app/pipeline/detect.py`, `tests/test_detect.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_detect.py`:

```python
import cv2
import numpy as np

from app.pipeline import detect


def test_detect_returns_empty_on_blank_frame():
    frame = np.full((240, 320, 3), 0, dtype=np.uint8)
    assert detect.detect_faces(frame) == []


def test_detect_returns_bboxes_on_real_frame(sample_video):
    cap = cv2.VideoCapture(str(sample_video))
    # Sample 5 frames spread across the clip
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    any_hit = False
    for i in [0, n_frames // 4, n_frames // 2, 3 * n_frames // 4, n_frames - 1]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            continue
        faces = detect.detect_faces(frame)
        if faces:
            any_hit = True
            for f in faces:
                assert "bbox" in f and "score" in f
                x, y, w, h = f["bbox"]
                assert w > 0 and h > 0
                assert 0.0 <= f["score"] <= 1.0
    cap.release()
    assert any_hit, "expected at least one face detection in the sample video"
```

- [ ] **Step 2: Run tests — verify they fail**

Run: `pytest tests/test_detect.py -v`
Expected: fail (module empty).

- [ ] **Step 3: Implement `app/pipeline/detect.py`**

```python
"""MediaPipe face detection wrapper."""
from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np

_mp_face = mp.solutions.face_detection
_detector = _mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5)


def detect_faces(frame_bgr: np.ndarray) -> list[dict]:
    """Run face detection on a BGR frame. Returns [{bbox: (x,y,w,h), score: float}, ...]."""
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    result = _detector.process(rgb)
    if not result.detections:
        return []
    out = []
    for det in result.detections:
        rb = det.location_data.relative_bounding_box
        x = max(0, int(rb.xmin * w))
        y = max(0, int(rb.ymin * h))
        bw = min(w - x, int(rb.width * w))
        bh = min(h - y, int(rb.height * h))
        if bw <= 0 or bh <= 0:
            continue
        out.append({"bbox": (x, y, bw, bh), "score": float(det.score[0])})
    return out
```

- [ ] **Step 4: Run tests — verify they pass**

Run: `pytest tests/test_detect.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add app/pipeline/detect.py tests/test_detect.py
git commit -m "feat: MediaPipe face detection wrapper"
```

---

## Task 7: Embedding + clustering (TDD)

**Files:**
- Create: `app/pipeline/embed_cluster.py`, `tests/test_embed_cluster.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_embed_cluster.py`:

```python
import cv2
import numpy as np

from app.pipeline import detect, embed_cluster


def test_embed_face_returns_128d_vector(sample_video):
    cap = cv2.VideoCapture(str(sample_video))
    ok, frame = cap.read()
    cap.release()
    assert ok
    faces = detect.detect_faces(frame)
    if not faces:
        import pytest; pytest.skip("no face on first frame")
    vec = embed_cluster.embed_face(frame, faces[0]["bbox"])
    assert vec is None or (vec.shape == (128,) and vec.dtype == np.float64)


def test_cluster_groups_similar_embeddings():
    # Build 8 embeddings: 4 near vector A, 4 near vector B
    rng = np.random.default_rng(0)
    a = rng.normal(size=128)
    a /= np.linalg.norm(a)
    b = rng.normal(size=128)
    b /= np.linalg.norm(b)
    vecs = []
    for _ in range(4):
        v = a + rng.normal(scale=0.02, size=128); v /= np.linalg.norm(v); vecs.append(v)
    for _ in range(4):
        v = b + rng.normal(scale=0.02, size=128); v /= np.linalg.norm(v); vecs.append(v)
    labels = embed_cluster.cluster(vecs)
    assert len(labels) == 8
    # All four As share a label; all four Bs share a label; the two labels differ
    assert labels[0] == labels[1] == labels[2] == labels[3]
    assert labels[4] == labels[5] == labels[6] == labels[7]
    assert labels[0] != labels[4]


def test_cluster_empty_input():
    assert embed_cluster.cluster([]) == []
```

- [ ] **Step 2: Run tests — verify they fail**

Run: `pytest tests/test_embed_cluster.py -v`
Expected: fail (module empty).

- [ ] **Step 3: Implement `app/pipeline/embed_cluster.py`**

```python
"""Face embedding (face_recognition / dlib) and DBSCAN clustering."""
from __future__ import annotations

import face_recognition
import numpy as np
from sklearn.cluster import DBSCAN


def embed_face(frame_bgr: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray | None:
    """Return a 128-d embedding for the face inside `bbox`, or None if dlib can't encode it."""
    x, y, w, h = bbox
    # face_recognition expects RGB and (top, right, bottom, left)
    rgb = frame_bgr[:, :, ::-1]
    location = (y, x + w, y + h, x)
    encs = face_recognition.face_encodings(rgb, known_face_locations=[location], num_jitters=1)
    return encs[0] if encs else None


def cluster(vectors: list[np.ndarray], eps: float = 0.5, min_samples: int = 2) -> list[int]:
    """Cluster L2-comparable vectors with DBSCAN. Returns one label per input (-1 for noise)."""
    if not vectors:
        return []
    X = np.stack(vectors)
    labels = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean").fit_predict(X)
    return labels.tolist()
```

Note on `eps`: face_recognition embeddings have an empirical decision threshold around 0.6 in Euclidean distance. We use 0.5 for tighter clusters (slightly more false-splits, far fewer false-merges — safer for "blur the right person").

- [ ] **Step 4: Run tests — verify they pass**

Run: `pytest tests/test_embed_cluster.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add app/pipeline/embed_cluster.py tests/test_embed_cluster.py
git commit -m "feat: face embeddings + DBSCAN clustering"
```

---

## Task 8: Analyze pipeline (TDD)

**Files:**
- Create: `app/pipeline/analyze.py`, `tests/test_analyze.py`

**Why:** Orchestrates Tasks 5–7 across a whole video and produces `analysis.json` + thumbnails.

- [ ] **Step 1: Write the failing test**

Create `tests/test_analyze.py`:

```python
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
    # The sample video has 2 distinct people
    assert len(data["people"]) == 2
    for person in data["people"]:
        assert {"id", "thumb", "frame_count", "first_seen_sec"} <= person.keys()
        assert storage.thumb_path(job_id, person["id"]).exists()
    # Timeline frames are in ascending order
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
```

- [ ] **Step 2: Run tests — verify they fail**

Run: `pytest tests/test_analyze.py -v`
Expected: fail (module empty).

- [ ] **Step 3: Implement `app/pipeline/analyze.py`**

```python
"""Analyze phase: detect + embed + cluster across a whole video."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import cv2
import numpy as np

from app import ffmpeg_utils, storage
from app.pipeline import detect, embed_cluster

SAMPLE_EVERY_N_FRAMES = 3  # ~10 detections/sec at 30 fps


def run(job_id: str, progress_cb: Callable[[float], None]) -> None:
    """Read the input video, detect/embed/cluster, write analysis.json + thumbs."""
    src = storage.input_path(job_id)
    info = ffmpeg_utils.probe(src)

    # Extract audio up front if present (used by render phase).
    if info["has_audio"]:
        ffmpeg_utils.extract_audio(src, storage.audio_path(job_id))

    cap = cv2.VideoCapture(str(src))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    detections: list[dict[str, Any]] = []
    embeddings: list[np.ndarray] = []

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % SAMPLE_EVERY_N_FRAMES == 0:
            for face in detect.detect_faces(frame):
                emb = embed_cluster.embed_face(frame, face["bbox"])
                if emb is None:
                    continue
                detections.append({
                    "frame": frame_idx,
                    "bbox": face["bbox"],
                    "score": face["score"],
                    "thumb_bgr": _crop_thumb(frame, face["bbox"]),
                })
                embeddings.append(emb)
            progress_cb(min(1.0, frame_idx / max(1, n_total)))
        frame_idx += 1
    cap.release()

    labels = embed_cluster.cluster(embeddings)

    # Build per-person aggregate + timeline.
    people: dict[str, dict[str, Any]] = {}
    for det, label in zip(detections, labels):
        if label == -1:
            continue  # DBSCAN noise — skip
        pid = f"p{label + 1}"
        person = people.setdefault(pid, {
            "id": pid,
            "thumb": f"thumbs/{pid}.jpg",
            "best_score": -1.0,
            "best_thumb_bgr": None,
            "frame_count": 0,
            "first_seen_frame": det["frame"],
        })
        person["frame_count"] += 1
        person["first_seen_frame"] = min(person["first_seen_frame"], det["frame"])
        if det["score"] > person["best_score"]:
            person["best_score"] = det["score"]
            person["best_thumb_bgr"] = det["thumb_bgr"]

    # Write thumbs.
    for pid, person in people.items():
        cv2.imwrite(str(storage.thumb_path(job_id, pid)), person["best_thumb_bgr"])

    # Build timeline (sampled frames only).
    timeline: dict[int, list[dict]] = {}
    for det, label in zip(detections, labels):
        if label == -1:
            continue
        pid = f"p{label + 1}"
        timeline.setdefault(det["frame"], []).append({"person_id": pid, "bbox": list(det["bbox"])})

    fps = info["fps"]
    payload = {
        "fps": fps,
        "duration_sec": info["duration_sec"],
        "width": info["width"],
        "height": info["height"],
        "has_audio": info["has_audio"],
        "people": [
            {
                "id": p["id"],
                "thumb": p["thumb"],
                "frame_count": p["frame_count"],
                "first_seen_sec": p["first_seen_frame"] / fps if fps else 0.0,
            }
            for p in people.values()
        ],
        "timeline": [
            {"frame": f, "faces": faces}
            for f, faces in sorted(timeline.items())
        ],
    }
    storage.write_analysis(job_id, payload)
    progress_cb(1.0)


def _crop_thumb(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = bbox
    return frame[y:y+h, x:x+w].copy()
```

- [ ] **Step 4: Run tests — verify they pass**

Run: `pytest tests/test_analyze.py -v`
Expected: 2 passed. Test may take ~10–30 s.

- [ ] **Step 5: Commit**

```bash
git add app/pipeline/analyze.py tests/test_analyze.py
git commit -m "feat: analyze pipeline produces analysis.json + thumbs"
```

---

## Task 9: Render pipeline (TDD)

**Files:**
- Create: `app/pipeline/render.py`, `tests/test_render.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_render.py`:

```python
import shutil

import cv2
import numpy as np

from app import ffmpeg_utils, storage
from app.pipeline import analyze, render


def _setup_job(sample_video, tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "JOBS_ROOT", tmp_path)
    job_id = storage.new_job()
    shutil.copy(sample_video, storage.input_path(job_id))
    analyze.run(job_id, progress_cb=lambda p: None)
    return job_id


def test_render_produces_playable_output(sample_video, tmp_path, monkeypatch):
    job_id = _setup_job(sample_video, tmp_path, monkeypatch)
    data = storage.read_analysis(job_id)
    blur_ids = [data["people"][0]["id"]]
    render.run(job_id, blur_ids, progress_cb=lambda p: None)
    out = storage.output_path(job_id)
    assert out.exists() and out.stat().st_size > 0
    info = ffmpeg_utils.probe(out)
    assert info["duration_sec"] > 0


def test_render_actually_blurs_selected_face(sample_video, tmp_path, monkeypatch):
    job_id = _setup_job(sample_video, tmp_path, monkeypatch)
    data = storage.read_analysis(job_id)
    blur_ids = [data["people"][0]["id"]]
    render.run(job_id, blur_ids, progress_cb=lambda p: None)

    # Find a frame where blur_ids[0] is present, compare variance in that bbox
    # before (input) and after (output).
    target_pid = blur_ids[0]
    sample_frame = None
    sample_bbox = None
    for entry in data["timeline"]:
        for face in entry["faces"]:
            if face["person_id"] == target_pid:
                sample_frame = entry["frame"]
                sample_bbox = tuple(face["bbox"])
                break
        if sample_frame is not None:
            break
    assert sample_frame is not None

    def _roi(video_path, frame_idx, bbox):
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        cap.release()
        assert ok
        x, y, w, h = bbox
        return frame[y:y+h, x:x+w].astype(np.float32)

    var_before = _roi(storage.input_path(job_id), sample_frame, sample_bbox).var()
    var_after = _roi(storage.output_path(job_id), sample_frame, sample_bbox).var()
    assert var_after < var_before * 0.5


def test_render_progress_callback(sample_video, tmp_path, monkeypatch):
    job_id = _setup_job(sample_video, tmp_path, monkeypatch)
    seen = []
    render.run(job_id, [], progress_cb=lambda p: seen.append(p))
    assert seen and seen[-1] >= 0.99


def test_render_with_no_audio_track(tmp_path, monkeypatch):
    import subprocess
    monkeypatch.setattr(storage, "JOBS_ROOT", tmp_path)
    job_id = storage.new_job()
    src = storage.input_path(job_id)
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=black:s=160x120:r=5:d=1",
         "-c:v", "libx264", str(src)],
        check=True, capture_output=True,
    )
    # Skip analyze; write a minimal analysis.json
    storage.write_analysis(job_id, {
        "fps": 5, "duration_sec": 1, "width": 160, "height": 120,
        "has_audio": False, "people": [], "timeline": [],
    })
    render.run(job_id, [], progress_cb=lambda p: None)
    assert storage.output_path(job_id).exists()
```

- [ ] **Step 2: Run tests — verify they fail**

Run: `pytest tests/test_render.py -v`
Expected: fail (module empty).

- [ ] **Step 3: Implement `app/pipeline/render.py`**

```python
"""Render phase: interpolate bboxes, apply blur, encode, mux audio."""
from __future__ import annotations

import shutil
from collections.abc import Callable

import cv2

from app import ffmpeg_utils, storage
from app.pipeline.blur import apply_gaussian_blur

MAX_INTERP_GAP_SEC = 0.5  # don't blur if no detection within this window


def run(job_id: str, blur_person_ids: list[str], progress_cb: Callable[[float], None]) -> None:
    data = storage.read_analysis(job_id)
    src = storage.input_path(job_id)
    fps = data["fps"]
    max_gap_frames = max(1, int(MAX_INTERP_GAP_SEC * fps))

    # Build per-person sorted detections list for interpolation lookup.
    per_person: dict[str, list[tuple[int, tuple[int, int, int, int]]]] = {}
    for entry in data["timeline"]:
        for face in entry["faces"]:
            per_person.setdefault(face["person_id"], []).append(
                (entry["frame"], tuple(face["bbox"]))
            )
    for pid in per_person:
        per_person[pid].sort(key=lambda x: x[0])

    cap = cv2.VideoCapture(str(src))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_only = storage.video_only_path(job_id)
    writer = cv2.VideoWriter(
        str(video_only), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        for pid in blur_person_ids:
            bbox = _interpolate_bbox(per_person.get(pid, []), frame_idx, max_gap_frames)
            if bbox is not None:
                apply_gaussian_blur(frame, bbox)
        writer.write(frame)
        frame_idx += 1
        if frame_idx % 30 == 0:
            progress_cb(min(1.0, frame_idx / max(1, n_total)))
    cap.release()
    writer.release()

    # Mux audio if we have it; otherwise just rename.
    if data["has_audio"] and storage.audio_path(job_id).exists():
        ffmpeg_utils.mux(video_only, storage.audio_path(job_id), storage.output_path(job_id))
        video_only.unlink(missing_ok=True)
    else:
        shutil.move(str(video_only), str(storage.output_path(job_id)))
    progress_cb(1.0)


def _interpolate_bbox(
    samples: list[tuple[int, tuple[int, int, int, int]]],
    frame_idx: int,
    max_gap_frames: int,
) -> tuple[int, int, int, int] | None:
    """Linear interpolation between the nearest before/after samples for the same person."""
    if not samples:
        return None
    before = after = None
    for f, b in samples:
        if f <= frame_idx:
            before = (f, b)
        if f >= frame_idx and after is None:
            after = (f, b)
            break
    if before is None or after is None:
        chosen = before or after
        if chosen and abs(chosen[0] - frame_idx) <= max_gap_frames:
            return chosen[1]
        return None
    if before[0] == after[0]:
        return before[1]
    if (after[0] - before[0]) > max_gap_frames:
        return None
    t = (frame_idx - before[0]) / (after[0] - before[0])
    bx, by, bw, bh = before[1]
    ax, ay, aw, ah = after[1]
    return (
        int(bx + (ax - bx) * t),
        int(by + (ay - by) * t),
        int(bw + (aw - bw) * t),
        int(bh + (ah - bh) * t),
    )
```

- [ ] **Step 4: Run tests — verify they pass**

Run: `pytest tests/test_render.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add app/pipeline/render.py tests/test_render.py
git commit -m "feat: render pipeline with bbox interpolation + audio mux"
```

---

## Task 10: Job registry + asyncio worker (TDD)

**Files:**
- Create: `app/jobs.py`, `tests/test_jobs.py`

**Why:** Sits between the API and the pipeline. Tracks job state and emits progress events.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_jobs.py`:

```python
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
    # Wait for completion (up to 60 s)
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
    # Calling again while rendering must not raise and must not double-start.
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
```

Add to `requirements.txt` if not present: `pytest-asyncio==0.24.*`. Add to `tests/conftest.py`:

```python
import pytest_asyncio  # noqa: F401  (registers the marker)
```

Configure asyncio mode in a new `pyproject.toml` minimal section, or add `pytest.ini`:

Create `pytest.ini`:

```ini
[pytest]
asyncio_mode = auto
```

- [ ] **Step 2: Run tests — verify they fail**

Run: `pytest tests/test_jobs.py -v`
Expected: fail (module empty).

- [ ] **Step 3: Implement `app/jobs.py`**

```python
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
```

- [ ] **Step 4: Run tests — verify they pass**

Run: `pytest tests/test_jobs.py -v`
Expected: 4 passed. May take 1–2 minutes.

- [ ] **Step 5: Commit**

```bash
git add app/jobs.py tests/test_jobs.py pytest.ini requirements.txt tests/conftest.py
git commit -m "feat: asyncio job worker with progress events"
```

---

## Task 11: REST API endpoints (TDD)

**Files:**
- Modify: `app/main.py`
- Create: `tests/test_api.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_api.py`:

```python
import time
from pathlib import Path

from fastapi.testclient import TestClient

from app import jobs, storage
from app.main import app


def _wait_status(client, job_id, target, timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/api/jobs/{job_id}")
        assert r.status_code == 200
        if r.json()["status"] == target:
            return r.json()
        if r.json()["status"] == "error":
            raise AssertionError(r.json())
        time.sleep(0.5)
    raise AssertionError(f"timeout waiting for {target}")


def test_happy_path(tmp_path, monkeypatch, sample_video):
    monkeypatch.setattr(storage, "JOBS_ROOT", tmp_path)
    jobs.reset()
    client = TestClient(app)

    with sample_video.open("rb") as fh:
        r = client.post("/api/jobs", files={"file": ("sample.mp4", fh, "video/mp4")})
    assert r.status_code == 200
    job_id = r.json()["job_id"]
    assert r.json()["status"] == "analyzing"

    _wait_status(client, job_id, "awaiting_selection")

    r = client.get(f"/api/jobs/{job_id}/people")
    assert r.status_code == 200
    people = r.json()["people"]
    assert len(people) == 2
    for p in people:
        r = client.get(p["thumb_url"])
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("image/")

    r = client.post(f"/api/jobs/{job_id}/render", json={"blur_person_ids": [people[0]["id"]]})
    assert r.status_code == 200

    _wait_status(client, job_id, "done")

    r = client.get(f"/api/jobs/{job_id}/download")
    assert r.status_code == 200
    assert r.headers["content-type"] == "video/mp4"
    assert len(r.content) > 0

    r = client.delete(f"/api/jobs/{job_id}")
    assert r.status_code == 200
    r = client.get(f"/api/jobs/{job_id}")
    assert r.json()["status"] == "unknown"


def test_render_idempotent_when_already_running(tmp_path, monkeypatch, sample_video):
    monkeypatch.setattr(storage, "JOBS_ROOT", tmp_path)
    jobs.reset()
    client = TestClient(app)
    with sample_video.open("rb") as fh:
        job_id = client.post("/api/jobs", files={"file": ("s.mp4", fh, "video/mp4")}).json()["job_id"]
    _wait_status(client, job_id, "awaiting_selection")
    client.post(f"/api/jobs/{job_id}/render", json={"blur_person_ids": []})
    r = client.post(f"/api/jobs/{job_id}/render", json={"blur_person_ids": []})
    assert r.status_code == 200  # no duplicate-start error
    _wait_status(client, job_id, "done")
```

- [ ] **Step 2: Run tests — verify they fail**

Run: `pytest tests/test_api.py -v`
Expected: fail (endpoints don't exist).

- [ ] **Step 3: Rewrite `app/main.py`**

```python
"""FastAPI app: REST + (WebSocket added in Task 12) + static (added in Task 14)."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from app import jobs, storage

app = FastAPI(title="Face Blur App", version="0.1.0")


class RenderRequest(BaseModel):
    blur_person_ids: list[str]


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


@app.post("/api/jobs")
async def create_job(file: UploadFile = File(...)) -> dict:
    job_id = jobs.create()
    target = storage.input_path(job_id)
    with target.open("wb") as fh:
        while chunk := await file.read(1 << 20):
            fh.write(chunk)
    jobs.start_analyze(job_id)
    return {"job_id": job_id, "status": "analyzing"}


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
    jobs.start_render(job_id, body.blur_person_ids)
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
```

- [ ] **Step 4: Run tests — verify they pass**

Run: `pytest tests/test_api.py -v`
Expected: 2 passed. Will take 1–3 minutes.

- [ ] **Step 5: Commit**

```bash
git add app/main.py tests/test_api.py
git commit -m "feat: REST endpoints for upload/status/people/render/download/delete"
```

---

## Task 12: WebSocket progress events (TDD)

**Files:**
- Modify: `app/main.py`
- Modify: `tests/test_api.py`

- [ ] **Step 1: Add the failing test**

Append to `tests/test_api.py`:

```python
def test_ws_receives_progress(tmp_path, monkeypatch, sample_video):
    monkeypatch.setattr(storage, "JOBS_ROOT", tmp_path)
    jobs.reset()
    client = TestClient(app)
    with sample_video.open("rb") as fh:
        job_id = client.post("/api/jobs", files={"file": ("s.mp4", fh, "video/mp4")}).json()["job_id"]

    with client.websocket_connect(f"/api/jobs/{job_id}/events") as ws:
        phases = set()
        for _ in range(50):
            ev = ws.receive_json()
            phases.add(ev["phase"])
            if ev["phase"] == "awaiting_selection":
                break
        assert "analyzing" in phases
        assert "awaiting_selection" in phases
```

- [ ] **Step 2: Run test — verify it fails**

Run: `pytest tests/test_api.py::test_ws_receives_progress -v`
Expected: fail (no WS route).

- [ ] **Step 3: Add WebSocket route to `app/main.py`**

Append imports:

```python
from fastapi import WebSocket, WebSocketDisconnect
```

Append at the bottom of `app/main.py`:

```python
@app.websocket("/api/jobs/{job_id}/events")
async def events(ws: WebSocket, job_id: str) -> None:
    await ws.accept()
    queue = jobs.subscribe(job_id)
    # Immediately emit a snapshot so the client can sync to current state.
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
```

- [ ] **Step 4: Run test — verify it passes**

Run: `pytest tests/test_api.py::test_ws_receives_progress -v`
Expected: PASS.

Also re-run the full API suite: `pytest tests/test_api.py -v`. All passing.

- [ ] **Step 5: Commit**

```bash
git add app/main.py tests/test_api.py
git commit -m "feat: websocket endpoint for live job progress events"
```

---

## Task 13: Frontend pages

**Files:**
- Create: `static/index.html`, `static/people.html`, `static/processing.html`, `static/done.html`
- Create: `static/js/api.js`, `static/js/progress.js`
- Create: `static/css/style.css`

**Why:** Four-page flow with WS progress and polling fallback.

- [ ] **Step 1: Create shared CSS**

Create `static/css/style.css`:

```css
* { box-sizing: border-box; }
body { font-family: system-ui, sans-serif; max-width: 800px; margin: 2em auto; padding: 0 1em; color: #222; }
h1 { font-weight: 500; }
button { padding: 0.6em 1.2em; font-size: 1em; cursor: pointer; }
button:disabled { opacity: 0.5; cursor: default; }
.progress { width: 100%; height: 8px; background: #eee; border-radius: 4px; overflow: hidden; margin: 1em 0; }
.progress > div { height: 100%; background: #4a7bd6; width: 0%; transition: width 0.3s; }
.people { display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap: 1em; margin: 1.5em 0; }
.person { border: 2px solid #ddd; border-radius: 8px; padding: 0.5em; text-align: center; cursor: pointer; user-select: none; }
.person img { width: 100%; aspect-ratio: 1; object-fit: cover; border-radius: 4px; }
.person.selected { border-color: #4a7bd6; background: #f0f5ff; }
.person.selected img { filter: blur(6px); }
.person .label { margin-top: 0.4em; font-size: 0.9em; color: #555; }
```

- [ ] **Step 2: Create `static/js/api.js`**

```javascript
const BASE = "";

export async function uploadVideo(file) {
  const fd = new FormData();
  fd.append("file", file);
  const r = await fetch(`${BASE}/api/jobs`, { method: "POST", body: fd });
  if (!r.ok) throw new Error(`upload failed: ${r.status}`);
  return r.json();
}

export async function getJob(jobId) {
  const r = await fetch(`${BASE}/api/jobs/${jobId}`);
  return r.json();
}

export async function getPeople(jobId) {
  const r = await fetch(`${BASE}/api/jobs/${jobId}/people`);
  return r.json();
}

export async function startRender(jobId, blurPersonIds) {
  const r = await fetch(`${BASE}/api/jobs/${jobId}/render`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ blur_person_ids: blurPersonIds }),
  });
  return r.json();
}

export function downloadUrl(jobId) {
  return `${BASE}/api/jobs/${jobId}/download`;
}

export async function deleteJob(jobId) {
  await fetch(`${BASE}/api/jobs/${jobId}`, { method: "DELETE" });
}
```

- [ ] **Step 3: Create `static/js/progress.js`**

```javascript
import { getJob } from "./api.js";

export function subscribe(jobId, onEvent) {
  // Try WebSocket; fall back to polling if it closes immediately.
  const protocol = location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(`${protocol}//${location.host}/api/jobs/${jobId}/events`);
  let pollTimer = null;
  let closed = false;

  function startPolling() {
    pollTimer = setInterval(async () => {
      const state = await getJob(jobId);
      onEvent({ phase: state.status, progress: state.progress });
      if (state.status === "done" || state.status === "error") stop();
    }, 1000);
  }

  function stop() {
    closed = true;
    if (pollTimer) clearInterval(pollTimer);
    try { ws.close(); } catch {}
  }

  ws.onmessage = (e) => onEvent(JSON.parse(e.data));
  ws.onerror = () => { if (!closed) startPolling(); };
  ws.onclose = (e) => {
    if (e.code !== 1000 && !closed && !pollTimer) startPolling();
  };

  return { stop };
}
```

- [ ] **Step 4: Create `static/index.html`** (upload page)

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Face Blur — Upload</title>
  <link rel="stylesheet" href="/static/css/style.css">
</head>
<body>
  <h1>Face Blur</h1>
  <p>Upload a video. We'll detect faces, you pick whose to blur, and we'll give you back a new file.</p>
  <input type="file" id="file" accept="video/*">
  <button id="upload" disabled>Upload</button>
  <div id="status" style="margin-top:1em;"></div>
  <script type="module">
    import { uploadVideo } from "/static/js/api.js";
    const fileEl = document.getElementById("file");
    const btn = document.getElementById("upload");
    const status = document.getElementById("status");
    fileEl.addEventListener("change", () => { btn.disabled = !fileEl.files.length; });
    btn.addEventListener("click", async () => {
      btn.disabled = true;
      status.textContent = "Uploading…";
      try {
        const { job_id } = await uploadVideo(fileEl.files[0]);
        location.href = `/static/people.html?job=${job_id}`;
      } catch (e) {
        status.textContent = "Error: " + e.message;
        btn.disabled = false;
      }
    });
  </script>
</body>
</html>
```

- [ ] **Step 5: Create `static/people.html`** (picker page)

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Face Blur — Pick people</title>
  <link rel="stylesheet" href="/static/css/style.css">
</head>
<body>
  <h1>Pick people to blur</h1>
  <div id="analyzing">
    <p id="analyzing-msg">Analyzing…</p>
    <div class="progress"><div id="bar"></div></div>
  </div>
  <div id="picker" style="display:none">
    <p>Tap the people you want blurred.</p>
    <div class="people" id="people"></div>
    <button id="next" disabled>Next</button>
  </div>
  <script type="module">
    import { getPeople, startRender } from "/static/js/api.js";
    import { subscribe } from "/static/js/progress.js";
    const jobId = new URLSearchParams(location.search).get("job");
    const selected = new Set();

    function render(people) {
      const root = document.getElementById("people");
      root.innerHTML = "";
      for (const p of people) {
        const card = document.createElement("div");
        card.className = "person";
        card.dataset.id = p.id;
        card.innerHTML = `<img src="${p.thumb_url}" alt=""><div class="label">${Math.round(p.first_seen_sec)}s · ${p.frame_count}f</div>`;
        card.addEventListener("click", () => {
          if (selected.has(p.id)) { selected.delete(p.id); card.classList.remove("selected"); }
          else { selected.add(p.id); card.classList.add("selected"); }
          document.getElementById("next").disabled = selected.size === 0;
        });
        root.appendChild(card);
      }
    }

    async function showPicker() {
      const { people } = await getPeople(jobId);
      document.getElementById("analyzing").style.display = "none";
      document.getElementById("picker").style.display = "block";
      render(people);
    }

    subscribe(jobId, (ev) => {
      if (ev.phase === "analyzing" && typeof ev.progress === "number") {
        document.getElementById("bar").style.width = (ev.progress * 100).toFixed(1) + "%";
      } else if (ev.phase === "awaiting_selection") {
        showPicker();
      } else if (ev.phase === "error") {
        document.getElementById("analyzing-msg").textContent = "Error: " + (ev.message || "");
      }
    });

    document.getElementById("next").addEventListener("click", async () => {
      await startRender(jobId, [...selected]);
      location.href = `/static/processing.html?job=${jobId}`;
    });
  </script>
</body>
</html>
```

- [ ] **Step 6: Create `static/processing.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Face Blur — Processing</title>
  <link rel="stylesheet" href="/static/css/style.css">
</head>
<body>
  <h1>Rendering…</h1>
  <div class="progress"><div id="bar"></div></div>
  <p id="msg">Applying blur to the selected people.</p>
  <script type="module">
    import { subscribe } from "/static/js/progress.js";
    const jobId = new URLSearchParams(location.search).get("job");
    subscribe(jobId, (ev) => {
      if (ev.phase === "rendering" && typeof ev.progress === "number") {
        document.getElementById("bar").style.width = (ev.progress * 100).toFixed(1) + "%";
      } else if (ev.phase === "done") {
        location.href = `/static/done.html?job=${jobId}`;
      } else if (ev.phase === "error") {
        document.getElementById("msg").textContent = "Error: " + (ev.message || "");
      }
    });
  </script>
</body>
</html>
```

- [ ] **Step 7: Create `static/done.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Face Blur — Done</title>
  <link rel="stylesheet" href="/static/css/style.css">
</head>
<body>
  <h1>Done</h1>
  <video id="preview" controls style="width:100%; max-width:600px;"></video>
  <p>
    <a id="dl" download="blurred.mp4"><button>Download MP4</button></a>
    <button id="restart">Start over</button>
  </p>
  <script type="module">
    import { downloadUrl, deleteJob } from "/static/js/api.js";
    const jobId = new URLSearchParams(location.search).get("job");
    const url = downloadUrl(jobId);
    document.getElementById("preview").src = url;
    document.getElementById("dl").href = url;
    document.getElementById("restart").addEventListener("click", async () => {
      await deleteJob(jobId);
      location.href = "/static/index.html";
    });
  </script>
</body>
</html>
```

- [ ] **Step 8: Commit**

```bash
git add static/
git commit -m "feat: four-page vanilla HTML frontend with WS progress"
```

---

## Task 14: Wire static files + landing redirect + end-to-end test

**Files:**
- Modify: `app/main.py`
- Create: `tests/test_e2e.py`

- [ ] **Step 1: Add static mount + root redirect to `app/main.py`**

Add imports at top of `app/main.py`:

```python
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
```

Add at the bottom of `app/main.py`:

```python
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def root() -> RedirectResponse:
    return RedirectResponse(url="/static/index.html")
```

- [ ] **Step 2: Smoke test the wiring**

Create `tests/test_e2e.py`:

```python
from fastapi.testclient import TestClient

from app.main import app


def test_root_redirects_to_upload():
    client = TestClient(app)
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (302, 307)
    assert r.headers["location"].endswith("/static/index.html")


def test_static_files_served():
    client = TestClient(app)
    for path in ("/static/index.html", "/static/people.html",
                 "/static/processing.html", "/static/done.html",
                 "/static/js/api.js", "/static/js/progress.js"):
        r = client.get(path)
        assert r.status_code == 200, path
```

- [ ] **Step 3: Run the new tests**

Run: `pytest tests/test_e2e.py -v`
Expected: 2 passed.

- [ ] **Step 4: Run the full suite**

Run: `pytest -v`
Expected: all tests pass.

- [ ] **Step 5: Manual smoke test**

```bash
uvicorn app.main:app --port 8000
```

In a browser, open `http://localhost:8000/`. Upload `tests/fixtures/sample_5s.mp4`, confirm the four-screen flow works end to end, download the blurred result and play it back. If anything is broken, fix it before committing.

- [ ] **Step 6: Commit**

```bash
git add app/main.py tests/test_e2e.py
git commit -m "feat: serve static frontend, end-to-end smoke tests"
```

---

## Task 15: README polish + final review

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Rewrite `README.md`** so a new user can run the app from a fresh clone.

```markdown
# Face Blur App

Local web application: upload a video, pick which detected people to blur, get back the same video with their faces Gaussian-blurred. Audio preserved.

## Stack

- **Backend:** FastAPI (Python 3.11+)
- **Detection:** MediaPipe Face Detection
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

The pipeline tests need `tests/fixtures/sample_5s.mp4` — a 5-second clip with two distinguishable faces. See Task 4 of the implementation plan for how to source one.

## Project layout

- `app/main.py` — FastAPI routes (REST + WebSocket)
- `app/jobs.py` — In-memory job registry + asyncio worker
- `app/pipeline/` — Detection, embeddings, clustering, blur, render
- `app/storage.py` — Disk I/O (single seam for future S3 swap)
- `static/` — Frontend HTML + JS
- `jobs/` — Per-job runtime directory (gitignored)

## Scope

This is an MVP. Single user, single process, local-only. No DB, no auth, no queue.

For the scaling path (mobile client, cloud storage, background workers) see [`docs/superpowers/specs/2026-05-24-face-blur-app-design.md`](docs/superpowers/specs/2026-05-24-face-blur-app-design.md).
```

- [ ] **Step 2: Run the full suite one more time**

Run: `pytest -v`
Expected: all tests pass.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: README with setup, run, test, and scope notes"
```

---

## Plan self-review

- **Spec coverage:**
  - §1 Goal — Tasks 13 (frontend flow) + 11 (REST) cover all five user steps.
  - §2 Architecture — Tasks 2 (storage), 10 (worker), 11 (REST), 14 (static).
  - §3a Analyze — Task 8.
  - §3b Render — Task 9.
  - §4 API surface — Task 11 (REST) + 12 (WS).
  - §5 Frontend — Task 13.
  - §6 Project layout — created across Tasks 1–14.
  - §7 Testing — every module has a TDD task plus Task 14 e2e.
  - §8 Dependencies — Task 1.
  - §9 Out of scope — enforced by NOT adding tasks for them.
  - §10 Future scaling — not in this plan, intentional.

- **Placeholder scan:** No TBDs, every code block is complete.

- **Type consistency:**
  - `progress_cb: Callable[[float], None]` — same signature in `analyze.run`, `render.run`, and `jobs.py`.
  - `bbox` is consistently `(x, y, w, h)` across `detect.py`, `blur.py`, `embed_cluster.py`, `analyze.py`, `render.py`.
  - Job statuses (`analyzing`, `awaiting_selection`, `rendering`, `done`, `error`, `unknown`) — same values in spec §4, `jobs.py`, and `main.py`.
  - Person IDs are `p{n}` strings (built in `analyze.py` from DBSCAN labels +1) — used consistently in spec, analyze output, render input, API responses, frontend.
  - `analysis.json` shape (`fps`, `duration_sec`, `width`, `height`, `has_audio`, `people`, `timeline`) matches between `analyze.py` writer and `render.py` reader.

Plan is internally consistent and covers the spec.
