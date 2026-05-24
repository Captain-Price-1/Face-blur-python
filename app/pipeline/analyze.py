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

    people: dict[str, dict[str, Any]] = {}
    for det, label in zip(detections, labels):
        if label == -1:
            continue
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

    for pid, person in people.items():
        cv2.imwrite(str(storage.thumb_path(job_id, pid)), person["best_thumb_bgr"])

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
