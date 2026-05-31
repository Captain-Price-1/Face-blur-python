"""YOLO11n person detection + segmentation for full-body blur."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

_DETECT_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "yolo11n.pt"
_SEG_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "yolo11n-seg.pt"
CONF = 0.4

_detector = None
_segmenter = None


def _get_detector():
    global _detector
    if _detector is None:
        from ultralytics import YOLO
        _detector = YOLO(str(_DETECT_PATH))
    return _detector


def _get_segmenter():
    global _segmenter
    if _segmenter is None:
        from ultralytics import YOLO
        _segmenter = YOLO(str(_SEG_PATH))
    return _segmenter


def detect_bodies(frame_bgr: np.ndarray, conf: float = CONF) -> list[tuple[int, int, int, int]]:
    """Person bounding boxes as (x, y, w, h)."""
    res = _get_detector()(frame_bgr, classes=[0], conf=conf, verbose=False)[0]
    out = []
    if res.boxes is None:
        return out
    for b in res.boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = (int(v) for v in b)
        if x2 > x1 and y2 > y1:
            out.append((x1, y1, x2 - x1, y2 - y1))
    return out


def segment_bodies(frame_bgr: np.ndarray, conf: float = CONF) -> list[tuple[tuple[int, int, int, int], np.ndarray]]:
    """List of (bbox, mask). mask is full-frame HxW uint8 in {0,255}."""
    h, w = frame_bgr.shape[:2]
    res = _get_segmenter()(frame_bgr, classes=[0], conf=conf, verbose=False)[0]
    out = []
    if res.masks is None or res.boxes is None:
        return out
    boxes = res.boxes.xyxy.cpu().numpy()
    masks = res.masks.data.cpu().numpy()  # (N, mh, mw) floats 0..1 at model res
    for box, m in zip(boxes, masks):
        mask = (m > 0.5).astype(np.uint8) * 255
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        x1, y1, x2, y2 = (int(v) for v in box)
        if x2 > x1 and y2 > y1:
            out.append(((x1, y1, x2 - x1, y2 - y1), mask))
    return out
