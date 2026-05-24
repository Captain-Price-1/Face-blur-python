"""YuNet (OpenCV) face detection wrapper.

YuNet ships with OpenCV >= 4.5.4 as `cv2.FaceDetectorYN`. Tiny model (~227 KB),
fast on CPU, multi-scale, high recall. We instantiate one detector per frame
size — YuNet requires the input size up front.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "face_detection_yunet_2023mar.onnx"
SCORE_THRESHOLD = 0.6
NMS_THRESHOLD = 0.3
TOP_K = 5000

# Cache: one detector per (width, height). YuNet ties the input size to the
# detector instance, so we keep a tiny LRU-ish dict by frame shape.
_detectors: dict[tuple[int, int], cv2.FaceDetectorYN] = {}


def _detector_for(width: int, height: int) -> cv2.FaceDetectorYN:
    key = (width, height)
    if key not in _detectors:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"YuNet model not found at {MODEL_PATH}")
        _detectors[key] = cv2.FaceDetectorYN.create(
            str(MODEL_PATH), "", (width, height), SCORE_THRESHOLD, NMS_THRESHOLD, TOP_K
        )
    return _detectors[key]


def detect_faces(frame_bgr: np.ndarray) -> list[dict]:
    """Detect faces in a BGR frame. Returns list of {bbox: (x,y,w,h), score: float}."""
    if frame_bgr.size == 0:
        return []
    h, w = frame_bgr.shape[:2]
    detector = _detector_for(w, h)
    _, faces = detector.detect(frame_bgr)
    if faces is None:
        return []
    out: list[dict] = []
    for face in faces:
        x, y, fw, fh = (int(v) for v in face[:4])
        x = max(0, x)
        y = max(0, y)
        fw = min(w - x, fw)
        fh = min(h - y, fh)
        if fw <= 0 or fh <= 0:
            continue
        score = float(face[-1])
        score = max(0.0, min(1.0, score))
        out.append({"bbox": (x, y, fw, fh), "score": score})
    return out
