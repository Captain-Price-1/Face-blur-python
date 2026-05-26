"""Analyze phase: detect → temporal-track → embed-per-track → cluster.

Key design:
- Tracking by IoU association across sampled frames produces stable identities
  within a continuous appearance. Eliminates per-frame flicker / ID swap.
- ONE face embedding per track (from the highest-scoring detection). Cuts the
  embedding cost from O(detections) to O(tracks) — typically a 10–100× speedup.
- Cross-track identity via DBSCAN on track embeddings. Handles the case where
  a person leaves and re-enters frame.
- Bbox positions are EMA-smoothed before being written to analysis.json so the
  render phase produces stable, non-jittery blurs.
- For detection speed: source frames are downsampled to DETECT_MAX_DIM if
  larger, then bboxes are scaled back to original coordinates.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from app import ffmpeg_utils, storage
from app.pipeline import detect, embed_cluster

SAMPLE_EVERY_N_FRAMES = 5     # 1 = every frame; higher = faster, sparser bbox samples
DETECT_MAX_DIM = 640          # downsample frames bigger than this for detection
IOU_THRESHOLD = 0.2           # min IoU to associate a detection with an existing track
MAX_TRACK_GAP_SAMPLES = 3     # how many sampled frames a track can miss before closing
EMA_ALPHA = 0.5               # bbox smoothing strength (0 = no smoothing, 1 = full)


@dataclass
class _Track:
    samples: list[tuple[int, tuple[int, int, int, int]]] = field(default_factory=list)
    best_score: float = -1.0
    best_thumb_bgr: np.ndarray | None = None
    best_frame_bgr: np.ndarray | None = None
    best_bbox: tuple[int, int, int, int] | None = None
    last_seen_frame: int = -1


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _ema_smooth(samples: list[tuple[int, tuple[int, int, int, int]]]) -> list[tuple[int, tuple[int, int, int, int]]]:
    """Two-pass exponential moving average smoothing on bboxes per coordinate."""
    if len(samples) < 2:
        return samples
    arr = np.array([list(s[1]) for s in samples], dtype=np.float32)  # (N, 4)
    # forward
    fwd = arr.copy()
    for i in range(1, len(fwd)):
        fwd[i] = EMA_ALPHA * fwd[i - 1] + (1 - EMA_ALPHA) * arr[i]
    # backward
    bwd = arr.copy()
    for i in range(len(bwd) - 2, -1, -1):
        bwd[i] = EMA_ALPHA * bwd[i + 1] + (1 - EMA_ALPHA) * arr[i]
    smoothed = ((fwd + bwd) / 2).round().astype(int)
    return [(samples[i][0], tuple(smoothed[i].tolist())) for i in range(len(samples))]


def _scale_factor(width: int, height: int) -> float:
    longer = max(width, height)
    if longer <= DETECT_MAX_DIM:
        return 1.0
    return DETECT_MAX_DIM / longer


def run(job_id: str, progress_cb: Callable[[float], None]) -> None:
    src = storage.input_path(job_id)
    info = ffmpeg_utils.probe(src)

    if info["has_audio"]:
        ffmpeg_utils.extract_audio(src, storage.audio_path(job_id))

    cap = cv2.VideoCapture(str(src))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    scale = _scale_factor(info["width"], info["height"])

    tracks: list[_Track] = []

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % SAMPLE_EVERY_N_FRAMES == 0:
            # Run detection on a (possibly) downsampled frame for speed.
            if scale < 1.0:
                small = cv2.resize(
                    frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
                )
                small_faces = detect.detect_faces(small)
                inv = 1.0 / scale
                faces = [
                    {
                        "bbox": (
                            int(f["bbox"][0] * inv),
                            int(f["bbox"][1] * inv),
                            int(f["bbox"][2] * inv),
                            int(f["bbox"][3] * inv),
                        ),
                        "score": f["score"],
                    }
                    for f in small_faces
                ]
            else:
                faces = detect.detect_faces(frame)

            # Greedy IoU association: each detection goes to its best matching open track.
            taken: set[int] = set()
            for face in faces:
                bbox = face["bbox"]
                best_idx = -1
                best_iou = IOU_THRESHOLD
                for ti, tr in enumerate(tracks):
                    if ti in taken:
                        continue
                    if frame_idx - tr.last_seen_frame > MAX_TRACK_GAP_SAMPLES * SAMPLE_EVERY_N_FRAMES:
                        continue
                    score = _iou(bbox, tr.samples[-1][1])
                    if score > best_iou:
                        best_iou = score
                        best_idx = ti

                if best_idx >= 0:
                    taken.add(best_idx)
                    tr = tracks[best_idx]
                else:
                    tr = _Track()
                    tracks.append(tr)

                tr.samples.append((frame_idx, bbox))
                tr.last_seen_frame = frame_idx
                if face["score"] > tr.best_score:
                    tr.best_score = face["score"]
                    tr.best_thumb_bgr = _crop(frame, bbox)
                    tr.best_frame_bgr = frame.copy()
                    tr.best_bbox = bbox

            progress_cb(min(0.85, frame_idx / max(1, n_total) * 0.85))
        frame_idx += 1
    cap.release()

    progress_cb(0.85)

    # Embed one face per track. Skip tracks where embedding can't be computed.
    track_embeddings: list[np.ndarray] = []
    embeddable_tracks: list[_Track] = []
    for tr in tracks:
        if tr.best_frame_bgr is None or tr.best_bbox is None:
            continue
        emb = embed_cluster.embed_face(tr.best_frame_bgr, tr.best_bbox)
        if emb is None:
            continue
        embeddable_tracks.append(tr)
        track_embeddings.append(emb)

    progress_cb(0.92)

    # Cluster tracks across the whole video. min_samples=1 means even a single
    # unique track gets its own cluster label (DBSCAN normally labels singletons
    # as noise = -1; we don't want to drop unique people).
    cluster_labels: list[int] = (
        embed_cluster.cluster(track_embeddings, eps=0.5, min_samples=1)
        if track_embeddings
        else []
    )

    # Map each (clustered) track to a person_id. Tracks without an embedding
    # get their own unique person_id — they exist visually and need blurring.
    person_id_for_track: dict[int, str] = {}  # id(track) -> "p1"/"p2"/...
    cluster_to_pid: dict[int, str] = {}
    next_pid = 1

    for tr, label in zip(embeddable_tracks, cluster_labels):
        if label == -1:
            pid = f"p{next_pid}"
            next_pid += 1
        else:
            if label not in cluster_to_pid:
                cluster_to_pid[label] = f"p{next_pid}"
                next_pid += 1
            pid = cluster_to_pid[label]
        person_id_for_track[id(tr)] = pid

    for tr in tracks:
        if id(tr) not in person_id_for_track and tr.best_frame_bgr is not None:
            person_id_for_track[id(tr)] = f"p{next_pid}"
            next_pid += 1

    # Aggregate per-person: thumbnail (highest score across all tracks of that
    # person), frame count, first appearance.
    people: dict[str, dict[str, Any]] = {}
    for tr in tracks:
        pid = person_id_for_track.get(id(tr))
        if pid is None:
            continue
        person = people.setdefault(pid, {
            "id": pid,
            "thumb": f"thumbs/{pid}.jpg",
            "best_score": -1.0,
            "best_thumb_bgr": None,
            "frame_count": 0,
            "first_seen_frame": tr.samples[0][0],
        })
        person["frame_count"] += len(tr.samples)
        person["first_seen_frame"] = min(person["first_seen_frame"], tr.samples[0][0])
        if tr.best_score > person["best_score"]:
            person["best_score"] = tr.best_score
            person["best_thumb_bgr"] = tr.best_thumb_bgr

    for pid, person in people.items():
        if person["best_thumb_bgr"] is not None:
            cv2.imwrite(str(storage.thumb_path(job_id, pid)), person["best_thumb_bgr"])

    progress_cb(0.97)

    # Build smoothed timeline. Merge bboxes by frame.
    timeline_by_frame: dict[int, list[dict]] = {}
    for tr in tracks:
        pid = person_id_for_track.get(id(tr))
        if pid is None:
            continue
        smoothed = _ema_smooth(tr.samples)
        for f_idx, bbox in smoothed:
            timeline_by_frame.setdefault(f_idx, []).append(
                {"person_id": pid, "bbox": list(bbox)}
            )

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
            for f, faces in sorted(timeline_by_frame.items())
        ],
    }
    storage.write_analysis(job_id, payload)
    progress_cb(1.0)


def _crop(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = bbox
    h_f, w_f = frame.shape[:2]
    x = max(0, x)
    y = max(0, y)
    w = min(w_f - x, w)
    h = min(h_f - y, h)
    return frame[y:y+h, x:x+w].copy() if w > 0 and h > 0 else np.zeros((1, 1, 3), dtype=np.uint8)
