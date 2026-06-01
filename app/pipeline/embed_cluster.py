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


def assign_people(
    embeddings: list[np.ndarray],
    frame_sets: list[set[int]],
    eps: float = 0.5,
) -> list[int]:
    """Assign each track to a person, respecting a hard cannot-link constraint.

    Two tracks that share ANY frame are visible at the same time, so they are
    necessarily different people and must never receive the same person label —
    no matter how similar their face embeddings look. (Plain DBSCAN ignores
    this and happily merges several simultaneous people into one cluster, which
    is the bug this replaces.)

    Greedy: for each track in order, attach it to the nearest existing person
    that (a) is within `eps` euclidean distance of the track embedding and
    (b) does not co-occur in time with it; otherwise start a new person.
    Returns one integer person-label per input track (0-based, dense).
    """
    people: list[dict] = []  # {"sum": vec, "n": int, "frames": set}
    labels: list[int] = []
    for emb, frames in zip(embeddings, frame_sets):
        fset = set(frames)
        best_i, best_d = -1, eps
        for i, p in enumerate(people):
            if p["frames"] & fset:          # co-occurs -> cannot be same person
                continue
            d = float(np.linalg.norm(p["sum"] / p["n"] - emb))
            if d < best_d:
                best_d, best_i = d, i
        if best_i >= 0:
            p = people[best_i]
            p["sum"] = p["sum"] + emb
            p["n"] += 1
            p["frames"] |= fset
            labels.append(best_i)
        else:
            people.append({"sum": np.asarray(emb, dtype=np.float64).copy(), "n": 1, "frames": fset})
            labels.append(len(people) - 1)
    return labels
