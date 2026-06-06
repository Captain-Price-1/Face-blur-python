"""Render phase: blur selected people, encode + mux via a single ffmpeg pipe.

- "face": blur each selected person's interpolated face box (cheap, 1 pass).
- "body_box" / "body_silhouette": blur the WHOLE body using EdgeTAM, a
  promptable video-object-segmentation model (SAM2 family). Each selected
  person is converted into a SEED POINT on the frame their face is clearest
  (a torso point derived from the face box); EdgeTAM segments that exact
  person and propagates the silhouette mask across the whole clip via a
  learned memory bank. This tracks the *selected* person — and only that
  person — through crossings and occlusions, which the previous
  detect+track+bind pipeline could not do reliably.
    body_silhouette -> blur the mask (true body shape)
    body_box        -> blur the mask's bounding box (rectangle)
  Detection runs on a downscaled, frame-strided copy for speed; masks are held
  across the stride and upscaled to full resolution at blur time.
"""
from __future__ import annotations

import queue
import subprocess
import threading
from collections.abc import Callable

import cv2
import numpy as np

from app import storage
from app.pipeline.blur import apply_gaussian_blur, apply_gaussian_blur_mask

MAX_INTERP_GAP_SEC = 0.5
PREFETCH_QUEUE_SIZE = 8

VALID_MODES = ("face", "body_box", "body_silhouette")

EDGETAM_WIDTH = 384     # width frames are downscaled to for segmentation
EDGETAM_STRIDE = 5      # run segmentation on every Nth frame; hold masks between


def _pick_device() -> str:
    """CPU. MPS (Apple GPU) runs EdgeTAM ~3x faster but produces WRONG masks
    here — it silently drops one of several tracked objects (a known SAM2/MPS
    op-coverage issue). Correctness wins, so we use CPU. On a machine with a
    real (CUDA) GPU this would be 'cuda' and both fast and correct."""
    return "cpu"


def run(
    job_id: str,
    blur_person_ids: list[str],
    progress_cb: Callable[[float], None],
    blur_mode: str = "face",
) -> None:
    if blur_mode not in VALID_MODES:
        raise ValueError(f"invalid blur_mode: {blur_mode!r}")

    data = storage.read_analysis(job_id)
    src = storage.input_path(job_id)
    fps = data["fps"]
    max_gap_frames = max(1, int(MAX_INTERP_GAP_SEC * fps))

    per_person: dict[str, list[tuple[int, tuple[int, int, int, int]]]] = {}
    for entry in data["timeline"]:
        for face in entry["faces"]:
            per_person.setdefault(face["person_id"], []).append(
                (entry["frame"], tuple(face["bbox"]))
            )
    for pid in per_person:
        per_person[pid].sort(key=lambda x: x[0])

    n_total = int(data.get("frame_count") or 0)
    if n_total <= 0:
        cap0 = cv2.VideoCapture(str(src))
        n_total = int(cap0.get(cv2.CAP_PROP_FRAME_COUNT))
        cap0.release()

    is_body = blur_mode in ("body_box", "body_silhouette")
    # masks_by_sample: sampled-frame-index -> downscaled uint8 mask; plus the
    # stride and the downscaled size so the render pass can map+upscale.
    masks_by_sample: dict[int, np.ndarray] = {}
    down_wh = (EDGETAM_WIDTH, 0)
    if is_body:
        masks_by_sample, down_wh = _edgetam_masks(
            src, blur_person_ids, per_person, n_total,
            lambda p: progress_cb(0.7 * p),
        )

    cap = cv2.VideoCapture(str(src))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = storage.output_path(job_id)
    has_audio = data["has_audio"] and storage.audio_path(job_id).exists()
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{w}x{h}", "-r", str(fps),
        "-i", "-",
    ]
    if has_audio:
        cmd += ["-i", str(storage.audio_path(job_id)),
                "-map", "0:v:0", "-map", "1:a:0", "-c:a", "copy"]
    cmd += [
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
        "-movflags", "+faststart",
        "-shortest",
        str(output_path),
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

    render_base = 0.7 if is_body else 0.0
    render_span = 0.3 if is_body else 1.0
    max_sample = max(masks_by_sample) if masks_by_sample else 0

    frame_idx = 0
    try:
        while True:
            item = frame_q.get()
            if item is SENTINEL:
                break
            frame = item

            if blur_mode == "face":
                for pid in blur_person_ids:
                    bbox = _interpolate_bbox(per_person.get(pid, []), frame_idx, max_gap_frames)
                    if bbox is not None:
                        apply_gaussian_blur(frame, bbox)
            else:
                sample = min(frame_idx // EDGETAM_STRIDE, max_sample)
                small_mask = masks_by_sample.get(sample)
                if small_mask is not None and small_mask.any():
                    full_mask = cv2.resize(small_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    if blur_mode == "body_silhouette":
                        apply_gaussian_blur_mask(frame, full_mask * 255)
                    else:  # body_box
                        ys, xs = np.where(full_mask > 0)
                        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
                        apply_gaussian_blur(frame, (int(x1), int(y1), int(x2 - x1), int(y2 - y1)), expand=1.05)

            try:
                ffmpeg.stdin.write(frame.tobytes())
            except BrokenPipeError:
                break
            frame_idx += 1
            if frame_idx % 30 == 0:
                progress_cb(min(1.0, render_base + render_span * frame_idx / max(1, n_total)))
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


# --------------------------------------------------------------------------- #
# EdgeTAM body masks
# --------------------------------------------------------------------------- #

def _seed_points(face_bbox, scale: float, width: int, height: int):
    """Two positive torso points (below the face) in downscaled coords. A point
    on the torso makes SAM segment the whole body; a point on the face alone
    tends to grab only the head."""
    fx, fy, fw, fh = face_bbox
    cx = (fx + fw / 2.0) * scale
    pts = []
    for k in (1.6, 2.6):
        x = int(min(max(cx, 0), width - 1))
        y = int(min(max((fy + k * fh) * scale, 0), height - 1))
        pts.append((x, y))
    return pts


def _edgetam_masks(
    src,
    blur_person_ids: list[str],
    per_person: dict[str, list[tuple[int, tuple[int, int, int, int]]]],
    n_total: int,
    progress_cb: Callable[[float], None],
):
    """Read a downscaled, frame-strided copy of the video, seed EdgeTAM from
    each selected person's clearest face, and return {sampled_idx: mask}."""
    from app.pipeline import body

    cap = cv2.VideoCapture(str(src))
    ofw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ofh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = EDGETAM_WIDTH / ofw
    dh = int(ofh * scale)
    frames: list = []
    fidx = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if fidx % EDGETAM_STRIDE == 0:
            small = cv2.resize(fr, (EDGETAM_WIDTH, dh), interpolation=cv2.INTER_AREA)
            frames.append(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        fidx += 1
    cap.release()
    progress_cb(0.1)
    if not frames:
        return {}, (EDGETAM_WIDTH, dh)

    # One seed per selected person, at their FIRST appearance — forward
    # propagation then covers their whole time on screen. (Seeding at the
    # *clearest* face instead would leave the frames before it un-blurred,
    # since we propagate forward only for speed.)
    seeds = []
    for i, pid in enumerate(blur_person_ids, start=1):
        samples = per_person.get(pid, [])
        if not samples:
            continue
        seed_frame, fbox = samples[0]
        sample_idx = min(seed_frame // EDGETAM_STRIDE, len(frames) - 1)
        seeds.append((i, sample_idx, _seed_points(fbox, scale, EDGETAM_WIDTH, dh)))
    if not seeds:
        return {}, (EDGETAM_WIDTH, dh)

    device = _pick_device()
    try:
        masks = body.track_masks(frames, seeds, device=device)
    except Exception:
        if device != "cpu":   # MPS can be flaky; fall back to CPU
            from app.pipeline import body as _b
            _b._edgetam_model = None
            _b._edgetam_proc = None
            masks = body.track_masks(frames, seeds, device="cpu")
        else:
            raise
    progress_cb(1.0)
    return masks, (EDGETAM_WIDTH, dh)


def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _match_body(face_bbox, body_bboxes, prev_body=None):
    """Pick the body a face belongs to by head-position fit. Retained for the
    analyze/test path; the render body modes now use EdgeTAM instead."""
    fx, fy, fw, fh = face_bbox
    fcx = fx + fw / 2.0
    fcy = fy + fh / 2.0
    best, best_score = None, None
    for b in body_bboxes:
        bx, by, bw, bh = b
        if bw <= 0 or bh <= 0:
            continue
        if not (bx <= fcx <= bx + bw):
            continue
        if fcy < by - fh or fcy > by + 0.55 * bh:
            continue
        top_gap = abs(fy - by) / bh
        horiz_off = abs(fcx - (bx + bw / 2.0)) / bw
        score = top_gap + horiz_off
        if prev_body is not None:
            score += (1.0 - _iou(b, prev_body)) * 0.6
        if best_score is None or score < best_score:
            best_score, best = score, b
    return best


def _interpolate_bbox(
    samples: list[tuple[int, tuple[int, int, int, int]]],
    frame_idx: int,
    max_gap_frames: int,
) -> tuple[int, int, int, int] | None:
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
