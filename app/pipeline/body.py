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


_edgetam_model = None
_edgetam_proc = None


def _get_edgetam(device: str = "cpu"):
    global _edgetam_model, _edgetam_proc
    if _edgetam_model is None:
        from transformers import EdgeTamVideoModel, Sam2VideoProcessor
        _edgetam_model = (
            EdgeTamVideoModel.from_pretrained("yonigozlan/EdgeTAM-hf").to(device).eval()
        )
        _edgetam_proc = Sam2VideoProcessor.from_pretrained("yonigozlan/EdgeTAM-hf")
    return _edgetam_model, _edgetam_proc


def track_masks(
    frames_rgb: list,
    seeds: list[tuple[int, int, tuple[int, int]]],
    device: str = "cpu",
) -> dict[int, np.ndarray]:
    """Promptable video segmentation (EdgeTAM / SAM2-style).

    Each seed is a (obj_id, frame_idx, (x, y)) positive point prompt placed on
    the person to track. EdgeTAM segments the object under each point and
    propagates its silhouette mask across the whole clip using a learned memory
    bank — this is what keeps the blur locked on the *selected* person and off
    everyone else, through crossings and occlusions.

    `frames_rgb` are the (already downscaled) RGB frames fed to the model.
    Returns {frame_idx: union_mask} where union_mask is a HxW uint8 {0,1} of all
    requested objects merged.
    """
    import torch

    if not frames_rgb or not seeds:
        return {}
    model, proc = _get_edgetam(device)
    h, w = frames_rgb[0].shape[:2]

    # Each seed is (obj_id, seed_frame, [(x,y), ...]) with seed_frame = the
    # person's FIRST appearance. We process each object in its own session and
    # propagate FORWARD ONLY from its seed: the person is present from their
    # seed onward, so a forward pass covers their whole appearance at half the
    # cost of a bidirectional pass. (A single shared session with objects
    # seeded at *different* frames is rejected by this EdgeTAM build, so we
    # session-per-object — fine for the handful of people a user selects.)
    masks: dict[int, np.ndarray] = {}
    for obj_id, seed_frame, pts in seeds:
        pts = list(pts)
        session = proc.init_video_session(video=frames_rgb, inference_device=device)
        proc.add_inputs_to_inference_session(
            inference_session=session,
            frame_idx=int(seed_frame),
            obj_ids=int(obj_id),
            input_points=[[[list(p) for p in pts]]],
            input_labels=[[[1] * len(pts)]],
        )
        with torch.inference_mode():
            for out in model.propagate_in_video_iterator(
                session, start_frame_idx=int(seed_frame)
            ):
                pm = proc.post_process_masks([out.pred_masks], [(h, w)], binarize=True)[0]
                m = (pm.cpu().numpy().reshape(-1, h, w) > 0.5).any(axis=0).astype(np.uint8)
                prev = masks.get(out.frame_idx)
                masks[out.frame_idx] = m if prev is None else (prev | m)
    return masks


def new_tracking_model():
    """A FRESH detection model instance for ByteTrack tracking.

    `model.track(persist=True)` carries tracker state (Kalman filters, ID
    counters) inside the model instance, so each render job must get its own
    instance rather than the shared singleton — otherwise track IDs and motion
    state would leak between videos.
    """
    from ultralytics import YOLO
    return YOLO(str(_DETECT_PATH))


def track_bodies(model, frame_bgr: np.ndarray, conf: float = CONF) -> list[tuple[int, tuple[int, int, int, int]]]:
    """One ByteTrack tracking step on a frame: [(track_id, (x, y, w, h)), ...].

    Track IDs are STABLE across frames — Kalman motion prediction, Hungarian
    assignment and a track buffer keep the same person on the same ID through
    crossings and brief occlusions (the standard MOT machinery that naive
    IoU-carry lacks). Call with consecutive frames of one video only.
    """
    res = model.track(
        frame_bgr,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[0],
        conf=conf,
        verbose=False,
    )[0]
    out: list[tuple[int, tuple[int, int, int, int]]] = []
    if res.boxes is None or res.boxes.id is None:
        return out
    ids = res.boxes.id.int().cpu().tolist()
    for tid, b in zip(ids, res.boxes.xyxy.cpu().numpy()):
        x1, y1, x2, y2 = (int(v) for v in b)
        if x2 > x1 and y2 > y1:
            out.append((int(tid), (x1, y1, x2 - x1, y2 - y1)))
    return out


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
