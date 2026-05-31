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
    sigma = max(15.0, max(w2, h2) / 4.0)
    blurred = cv2.GaussianBlur(roi, (0, 0), sigmaX=sigma)
    mask = _oval_mask(w2, h2, feather_px=int(min(w2, h2) * 0.15))
    frame[y2:y_end, x2:x_end] = (roi * (1 - mask) + blurred * mask).astype(np.uint8)


def apply_gaussian_blur_mask(frame: np.ndarray, mask: np.ndarray) -> None:
    """Blur the frame wherever `mask` (HxW uint8) is non-zero, in place.
    Feathers the mask edge so the silhouette blends."""
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return
    y1, y2, x1, x2 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    roi = frame[y1:y2, x1:x2]
    sigma = max(15.0, max(roi.shape[0], roi.shape[1]) / 6.0)
    blurred = cv2.GaussianBlur(roi, (0, 0), sigmaX=sigma)
    m = mask[y1:y2, x1:x2].astype(np.float32) / 255.0
    # feather
    k = max(3, (int(min(roi.shape[0], roi.shape[1]) * 0.05)) | 1)
    m = cv2.GaussianBlur(m, (k, k), 0)[..., None]
    frame[y1:y2, x1:x2] = (roi * (1 - m) + blurred * m).astype(np.uint8)
