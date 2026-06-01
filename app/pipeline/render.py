"""Render phase: interpolate bboxes, blur, encode + mux via single ffmpeg pipe.

Performance notes:
- Output is encoded via a single ffmpeg subprocess that reads raw BGR frames
  from stdin and muxes the original audio in one pass. This avoids the
  intermediate `video_only.mp4` write/read cycle.
- A pre-decode worker thread reads the next frame while the main thread
  blurs and pipes the current frame. Hides ~30-40% of the decode latency.

Blur modes:
- "face": original behavior — blur each selected person's interpolated face box.
- "body_box": sparse pre-pass runs YOLO person detection on the sampled frames,
  associates each selected face to a body box via containment, then the main
  pass interpolates and blurs those body rectangles (fast, YOLO calls sparse).
- "body_silhouette": main pass runs per-frame YOLO segmentation and blurs the
  matched person's silhouette mask (slower, best looking).
"""
from __future__ import annotations

import queue
import subprocess
import threading
from collections.abc import Callable

import cv2

from app import storage
from app.pipeline.blur import apply_gaussian_blur, apply_gaussian_blur_mask

MAX_INTERP_GAP_SEC = 0.5
PREFETCH_QUEUE_SIZE = 8  # bounded so we don't run out of RAM on huge videos

VALID_MODES = ("face", "body_box", "body_silhouette")

# Body modes: run the (expensive) YOLO model only every Nth frame in a single
# sequential pass, and HOLD the matched body box / mask forward for the
# in-between frames. Sequential decode + sparse inference is far faster than
# seeking to sampled frames in a separate pre-pass. A held entry expires after
# HOLD_FRAMES so a person who leaves the frame stops being blurred shortly
# after, while a brief detection miss never un-blurs them.
BODY_DETECT_EVERY = 5
SIL_DETECT_EVERY = 3
HOLD_FRAMES_BOX = 12
HOLD_FRAMES_SIL = 6


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

    # Per-person FACE samples, as today.
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

    # Build ffmpeg command. One process, raw frames on stdin, optionally
    # audio on a second input, h264 + aac out.
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
                "-map", "0:v:0", "-map", "1:a:0",
                "-c:a", "copy"]
    cmd += [
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
        # +faststart moves the moov index to the front so the browser can
        # preview/stream the file immediately instead of waiting for the whole
        # (potentially 100+ MB) download — important for long videos.
        "-movflags", "+faststart",
        "-shortest",
        str(output_path),
    ]
    ffmpeg = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    # Pre-decode worker. Reads frames into a bounded queue so the main loop
    # never blocks on disk I/O.
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

    decoder_thread = threading.Thread(target=decoder, daemon=True)
    decoder_thread.start()

    # Hold-forward state for body modes: pid -> [bbox|mask, ttl].
    held_boxes: dict[str, list] = {}
    held_masks: dict[str, list] = {}

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
            elif blur_mode == "body_box":
                _blur_body_box(
                    frame, frame_idx, blur_person_ids, per_person, max_gap_frames, held_boxes
                )
            else:  # body_silhouette
                _blur_silhouettes(
                    frame, frame_idx, blur_person_ids, per_person, max_gap_frames, held_masks
                )

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
    """Pick the body a face belongs to.

    Plain containment is ambiguous when people overlap: a small face can fall
    inside several body boxes at once. A person's face actually sits at the
    TOP-CENTER of THEIR body, so we score each candidate by head-position fit
    (face near the body's top, horizontally centered) instead. A temporal bias
    toward the body matched on the previous detection tick prevents the blur
    from flickering between adjacent people. Returns None if no body plausibly
    owns the face (lower score is better).
    """
    fx, fy, fw, fh = face_bbox
    fcx = fx + fw / 2.0
    fcy = fy + fh / 2.0
    best, best_score = None, None
    for b in body_bboxes:
        bx, by, bw, bh = b
        if bw <= 0 or bh <= 0:
            continue
        # Face center must be horizontally inside the body and in its UPPER
        # region — a head is never in the lower half of its own body. This
        # rejects neighbouring bodies that merely overlap the face.
        if not (bx <= fcx <= bx + bw):
            continue
        if fcy < by - fh or fcy > by + 0.55 * bh:
            continue
        top_gap = abs(fy - by) / bh            # face top vs body top
        horiz_off = abs(fcx - (bx + bw / 2.0)) / bw  # face vs body center-x
        score = top_gap + horiz_off
        if prev_body is not None:              # temporal continuity
            score += (1.0 - _iou(b, prev_body)) * 0.6
        if best_score is None or score < best_score:
            best_score, best = score, b
    return best


def _blur_body_box(
    frame,
    frame_idx: int,
    blur_person_ids: list[str],
    per_person: dict[str, list[tuple[int, tuple[int, int, int, int]]]],
    max_gap_frames: int,
    held: dict[str, list],
) -> None:
    """Single-pass body-box blur. Runs YOLO person detection every
    BODY_DETECT_EVERY frames, matches each selected person's interpolated face
    box to a body box, and holds the result forward for the in-between frames."""
    from app.pipeline import body

    if frame_idx % BODY_DETECT_EVERY == 0:
        bodies = body.detect_bodies(frame)
        if bodies:
            for pid in blur_person_ids:
                face_bbox = _interpolate_bbox(per_person.get(pid, []), frame_idx, max_gap_frames)
                if face_bbox is None:
                    continue
                prev = held[pid][0] if pid in held else None
                matched = _match_body(face_bbox, bodies, prev)
                if matched is not None:
                    held[pid] = [matched, HOLD_FRAMES_BOX]

    for pid in list(held):
        bbox, ttl = held[pid]
        # YOLO body boxes are already tight around the whole person, so use a
        # small expand (not the wide face default that pads for hair/jaw).
        apply_gaussian_blur(frame, bbox, expand=1.1)
        if ttl <= 1:
            del held[pid]
        else:
            held[pid][1] = ttl - 1


def _blur_silhouettes(
    frame,
    frame_idx: int,
    blur_person_ids: list[str],
    per_person: dict[str, list[tuple[int, tuple[int, int, int, int]]]],
    max_gap_frames: int,
    held: dict[str, list],
) -> None:
    """Single-pass silhouette blur. Runs YOLO segmentation every
    SIL_DETECT_EVERY frames, matches each selected person's interpolated face
    box to a body mask, and holds the mask forward for the in-between frames."""
    from app.pipeline import body

    if frame_idx % SIL_DETECT_EVERY == 0:
        seg = body.segment_bodies(frame)
        if seg:
            seg_boxes = [b for b, _ in seg]
            for pid in blur_person_ids:
                face_bbox = _interpolate_bbox(per_person.get(pid, []), frame_idx, max_gap_frames)
                if face_bbox is None:
                    continue
                prev = held[pid][1] if pid in held else None
                matched = _match_body(face_bbox, seg_boxes, prev)
                if matched is None:
                    continue
                for b, mask in seg:
                    if b == matched:
                        held[pid] = [mask, matched, HOLD_FRAMES_SIL]
                        break

    for pid in list(held):
        mask, bbox, ttl = held[pid]
        apply_gaussian_blur_mask(frame, mask)
        if ttl <= 1:
            del held[pid]
        else:
            held[pid][2] = ttl - 1


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
