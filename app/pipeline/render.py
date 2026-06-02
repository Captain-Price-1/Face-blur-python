"""Render phase: blur selected people, encode + mux via a single ffmpeg pipe.

Performance notes:
- Output is encoded via a single ffmpeg subprocess that reads raw BGR frames
  from stdin and muxes the original audio in one pass (+faststart so the
  browser can preview/stream immediately).
- A pre-decode worker thread reads the next frame while the main thread blurs
  and pipes the current one.

Blur modes:
- "face": blur each selected person's interpolated face box (cheap, 1 pass).
- "body_box" / "body_silhouette": blur the whole body. These first build a
  face-INDEPENDENT body track per selected person (pre-pass), then blur from it
  every frame. The face is only used to identify WHICH body is the selected
  person at the frames where it's visible; the body is then followed across
  frames by IoU — forward AND backward — so coverage:
    * starts as soon as the body appears (even slightly before the face is
      first detected), removing the start-up lag, and
    * never drops out when the face turns away / is occluded, as long as the
      body itself is still detected.
  This trades extra compute (a detection pre-pass) for accuracy, by design.
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

SIL_DETECT_EVERY = 2     # segmentation cadence in the render pass (silhouette)
BRIDGE_GAP_FRAMES = 8    # max gap (frames) to linearly bridge in a body track
SCENE_CUT_DIFF = 30.0    # mean |gray diff| (64x36 thumbs) above which a frame is a shot cut


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

    # Per-person FACE samples from analysis.json.
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

    # Body modes: build face-independent body tracks first (pre-pass = first
    # half of progress). Face mode needs no pre-pass.
    body_tracks: dict[str, dict[int, tuple[int, int, int, int]]] = {}
    is_body = blur_mode in ("body_box", "body_silhouette")
    if is_body:
        body_tracks = _build_body_tracks(
            src, blur_person_ids, per_person, max_gap_frames, n_total,
            lambda p: progress_cb(0.5 * p),
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

    render_base = 0.5 if is_body else 0.0
    render_span = 0.5 if is_body else 1.0
    held_masks: dict[str, object] = {}  # pid -> last silhouette mask (silhouette)

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
                for pid in blur_person_ids:
                    tb = body_tracks.get(pid, {}).get(frame_idx)
                    if tb is not None:
                        apply_gaussian_blur(frame, tb, expand=1.05)
            else:  # body_silhouette
                _blur_silhouettes_tracked(
                    frame, frame_idx, blur_person_ids, body_tracks, held_masks
                )

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
# Body tracking (pre-pass)
# --------------------------------------------------------------------------- #

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
    """Pick the body a face belongs to. A person's face sits at the TOP-CENTER
    of THEIR body, so score by head-position fit (not plain containment, which
    is ambiguous when people overlap). Optional temporal bias toward the body
    matched previously. Returns None if no body plausibly owns the face."""
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


def _build_body_tracks(
    src,
    blur_person_ids: list[str],
    per_person: dict[str, list[tuple[int, tuple[int, int, int, int]]]],
    max_gap_frames: int,
    n_total: int,
    progress_cb: Callable[[float], None],
) -> dict[str, dict[int, tuple[int, int, int, int]]]:
    """ByteTrack pre-pass (the standard MOT approach, via ultralytics).

    1. Track every person through the video with STABLE track IDs — Kalman
       motion prediction + Hungarian assignment + a track buffer keep the same
       human on the same ID through crossings and brief occlusions.
    2. Detect SCENE CUTS (cheap frame-difference) and RESET the tracker at
       each one. Trackers predict motion across frames; across a hard cut that
       prediction is meaningless and lets an ID bleed onto a different person
       in the new shot. Per-shot IDs make that impossible.
    3. Bind each selected face to track IDs by MAJORITY VOTE across the frames
       where that face is visible — with face interpolation CLAMPED to the
       same shot, so a face can never vote for a body in a different shot
       (stale boxes held across a cut were observed voting for whoever stood
       at that screen position in the next scene).
    4. Emit a dense per-frame {frame: body_box} for each selected person,
       bridging only small tracker dropouts.
    """
    from app.pipeline import body

    model = body.new_tracking_model()
    cap = cv2.VideoCapture(str(src))
    frame_tracks: list[list[tuple[int, tuple[int, int, int, int]]]] = []
    shot_of: list[int] = []          # frame index -> shot index
    shot = 0
    prev_small: np.ndarray | None = None
    fidx = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        small = cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), (64, 36)).astype(np.int16)
        if prev_small is not None and float(np.abs(small - prev_small).mean()) > SCENE_CUT_DIFF:
            shot += 1
            model = body.new_tracking_model()  # hard reset: track IDs are per-shot
        prev_small = small
        shot_of.append(shot)
        # Namespace track IDs by shot: the tracker's ID counter restarts on
        # reset, so a raw ID is only unique within its shot.
        frame_tracks.append(
            [((shot, tid), b) for tid, b in body.track_bodies(model, fr)]
        )
        if fidx % 30 == 0:
            progress_cb(min(1.0, fidx / max(1, n_total)))
        fidx += 1
    cap.release()
    progress_cb(1.0)
    n = len(frame_tracks)

    tracks: dict[str, dict[int, tuple[int, int, int, int]]] = {}
    for pid in blur_person_ids:
        samples = per_person.get(pid, [])
        if not samples:
            continue
        # Face samples grouped by shot, so interpolation can never reach
        # across a cut.
        samples_by_shot: dict[int, list] = {}
        for f, b in samples:
            if 0 <= f < n:
                samples_by_shot.setdefault(shot_of[f], []).append((f, b))

        # Vote: at every frame where this person's face is visible (within the
        # same shot), head-match the face to the tracked boxes.
        votes: dict[int, int] = {}
        for f in range(n):
            if not frame_tracks[f]:
                continue
            shot_samples = samples_by_shot.get(shot_of[f])
            if not shot_samples:
                continue
            face = _interpolate_bbox(shot_samples, f, max_gap_frames)
            if face is None:
                continue
            boxes = [b for _, b in frame_tracks[f]]
            mb = _match_body(face, boxes)
            if mb is None:
                continue
            for tid, b in frame_tracks[f]:
                if b == mb:
                    votes[tid] = votes.get(tid, 0) + 1
                    break
        if not votes:
            continue
        total = sum(votes.values())
        threshold = max(2, int(0.15 * total))
        strong = {tid for tid, v in votes.items() if v >= threshold}
        if not strong:
            strong = {max(votes, key=votes.get)}

        dense: dict[int, tuple[int, int, int, int]] = {}
        for f in range(n):
            cands = [(tid, b) for tid, b in frame_tracks[f] if tid in strong]
            if cands:
                # If two selected IDs ever co-occur (rare ID-handoff overlap),
                # keep the more-voted one.
                cands.sort(key=lambda tb: -votes.get(tb[0], 0))
                dense[f] = cands[0][1]

        # Bridge small tracker dropouts by interpolation; leave large gaps
        # (genuine absence / scene change) unfilled.
        sf = sorted(dense)
        for i in range(len(sf) - 1):
            f, f2 = sf[i], sf[i + 1]
            if 1 < (f2 - f) <= BRIDGE_GAP_FRAMES:
                a, b = dense[f], dense[f2]
                for g in range(f + 1, f2):
                    t = (g - f) / (f2 - f)
                    dense[g] = tuple(int(a[k] + (b[k] - a[k]) * t) for k in range(4))
        tracks[pid] = dense

    return tracks


MASK_MATCH_IOU = 0.5   # a person's OWN seg mask overlaps their tracked box at
                       # ~0.7-0.95; an overlapping NEIGHBOUR scores ~0.3. Strict
                       # matching stops a neighbour's mask being blurred when the
                       # target's own mask is missing from a seg result.
MASK_STALE_IOU = 0.3   # reuse a held mask only while it still overlaps the track


def _blur_silhouettes_tracked(frame, frame_idx, blur_person_ids, body_tracks, held_masks):
    """Blur the silhouette of each tracked body.

    Segmentation runs every SIL_DETECT_EVERY frames; a mask is accepted for a
    person only when it STRICTLY matches their (face-independent) tracked box
    (IoU >= MASK_MATCH_IOU), and is reused between segmentation frames while it
    still overlaps the moving track. Whenever no trustworthy own-mask exists
    (the seg model occasionally misses a person entirely), the tracked BOX is
    blurred instead — guaranteeing the right person stays covered rather than
    borrowing an overlapping neighbour's mask."""
    from app.pipeline import body

    seg = body.segment_bodies(frame, conf=0.3) if frame_idx % SIL_DETECT_EVERY == 0 else None

    for pid in blur_person_ids:
        tb = body_tracks.get(pid, {}).get(frame_idx)
        if tb is None:
            held_masks.pop(pid, None)
            continue
        if seg:
            best, best_iou = None, MASK_MATCH_IOU
            for bbox, mask in seg:
                s = _iou(tb, bbox)
                if s > best_iou:
                    best_iou, best = s, (mask, bbox)
            if best is not None:
                held_masks[pid] = best
        entry = held_masks.get(pid)
        if entry is not None and _iou(tb, entry[1]) >= MASK_STALE_IOU:
            apply_gaussian_blur_mask(frame, entry[0])
        else:
            held_masks.pop(pid, None)
            apply_gaussian_blur(frame, tb, expand=1.05)  # right person, box coverage


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
