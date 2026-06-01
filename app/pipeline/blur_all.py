"""Single-pass "blur everyone" mode.

Skips identification (embeddings/clustering) and the people-picker entirely:
decode each frame once, detect faces, blur them, write — in one video pass.
This is the fast path for "just blur all faces".

Why one pass is faster than analyze+render: the picker flow decodes the video
twice (once to find/identify faces, once to blur the chosen ones). When we
blur everyone we don't need identity, so we fold detection and blurring into a
single decode+encode.

Detection runs every DETECT_EVERY frames (downsampled for speed). Between
detections, the most recent boxes are *held forward* for a few frames so a
brief detection miss doesn't un-blur a face. We never project boxes backward,
so a face is never blurred before it actually appears.
"""
from __future__ import annotations

import queue
import subprocess
import threading
from collections.abc import Callable

import cv2

from app import storage
from app.pipeline import detect
from app.pipeline.blur import apply_gaussian_blur

DETECT_EVERY = 5          # run detection every Nth frame
HOLD_FRAMES = 8           # keep a detected box alive this many frames after last seen
DETECT_MAX_DIM = 640      # downsample bigger frames before detection
PREFETCH_QUEUE_SIZE = 8


def _scale_factor(width: int, height: int) -> float:
    longer = max(width, height)
    return DETECT_MAX_DIM / longer if longer > DETECT_MAX_DIM else 1.0


def run(job_id: str, progress_cb: Callable[[float], None]) -> None:
    src = storage.input_path(job_id)
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open input video: {src}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    scale = _scale_factor(w, h)

    # Does the source have an audio stream? (Cheap probe via OpenCV is not
    # possible, so use ffprobe through ffmpeg_utils.)
    from app import ffmpeg_utils
    has_audio = ffmpeg_utils.probe(src)["has_audio"]

    output_path = storage.output_path(job_id)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{w}x{h}", "-r", str(fps),
        "-i", "-",
    ]
    if has_audio:
        cmd += ["-i", str(src), "-map", "0:v:0", "-map", "1:a:0", "-c:a", "aac", "-b:a", "192k"]
    cmd += [
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
        "-movflags", "+faststart",  # moov at front -> instant browser preview
        "-shortest", str(output_path),
    ]
    ffmpeg = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    frame_q: queue.Queue = queue.Queue(maxsize=PREFETCH_QUEUE_SIZE)
    SENTINEL = object()

    def decoder() -> None:
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_q.put(frame)
        finally:
            frame_q.put(SENTINEL)

    threading.Thread(target=decoder, daemon=True).start()

    # active boxes: list of [bbox, ttl]
    active: list[list] = []
    frame_idx = 0
    try:
        while True:
            item = frame_q.get()
            if item is SENTINEL:
                break
            frame = item

            if frame_idx % DETECT_EVERY == 0:
                if scale < 1.0:
                    small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                    inv = 1.0 / scale
                    boxes = [
                        (int(f["bbox"][0] * inv), int(f["bbox"][1] * inv),
                         int(f["bbox"][2] * inv), int(f["bbox"][3] * inv))
                        for f in detect.detect_faces(small)
                    ]
                else:
                    boxes = [f["bbox"] for f in detect.detect_faces(frame)]
                active = [[b, HOLD_FRAMES] for b in boxes]

            for entry in active:
                apply_gaussian_blur(frame, entry[0])
                entry[1] -= 1
            active = [e for e in active if e[1] > 0]

            try:
                ffmpeg.stdin.write(frame.tobytes())
            except BrokenPipeError:
                break
            frame_idx += 1
            if frame_idx % 30 == 0:
                progress_cb(min(1.0, frame_idx / max(1, n_total)))
    finally:
        cap.release()
        try:
            ffmpeg.stdin.close()
        except BrokenPipeError:
            pass
        rc = ffmpeg.wait()
        if rc != 0:
            err = ffmpeg.stderr.read().decode("utf-8", "replace") if ffmpeg.stderr else ""
            raise RuntimeError(f"ffmpeg exited {rc}: {err}")

    progress_cb(1.0)
