"""Render phase: interpolate bboxes, apply blur, encode, mux audio."""
from __future__ import annotations

import shutil
from collections.abc import Callable

import cv2

from app import ffmpeg_utils, storage
from app.pipeline.blur import apply_gaussian_blur

MAX_INTERP_GAP_SEC = 0.5


def run(job_id: str, blur_person_ids: list[str], progress_cb: Callable[[float], None]) -> None:
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
