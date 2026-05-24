"""Face embedding (face_recognition / dlib) and DBSCAN clustering."""
from __future__ import annotations

import face_recognition
import numpy as np
from sklearn.cluster import DBSCAN


def embed_face(frame_bgr: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray | None:
    """Return a 128-d embedding for the face inside `bbox`, or None if dlib can't encode it."""
    x, y, w, h = bbox
    # face_recognition expects RGB and (top, right, bottom, left).
    # np.ascontiguousarray is required: the [::-1] slice produces a non-C-contiguous
    # view, which causes dlib 20.x pybind11 bindings to reject the array.
    rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
    location = (y, x + w, y + h, x)
    encs = face_recognition.face_encodings(rgb, known_face_locations=[location], num_jitters=1)
    return encs[0] if encs else None


def cluster(vectors: list[np.ndarray], eps: float = 0.5, min_samples: int = 2) -> list[int]:
    """Cluster L2-comparable vectors with DBSCAN. Returns one label per input (-1 for noise)."""
    if not vectors:
        return []
    X = np.stack(vectors)
    labels = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean").fit_predict(X)
    return labels.tolist()
