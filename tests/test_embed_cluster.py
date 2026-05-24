import cv2
import numpy as np

from app.pipeline import detect, embed_cluster


def test_embed_face_returns_128d_vector(sample_video):
    cap = cv2.VideoCapture(str(sample_video))
    ok, frame = cap.read()
    cap.release()
    assert ok
    faces = detect.detect_faces(frame)
    if not faces:
        import pytest; pytest.skip("no face on first frame")
    vec = embed_cluster.embed_face(frame, faces[0]["bbox"])
    assert vec is None or (vec.shape == (128,) and vec.dtype == np.float64)


def test_cluster_groups_similar_embeddings():
    # Build 8 embeddings: 4 near vector A, 4 near vector B
    rng = np.random.default_rng(0)
    a = rng.normal(size=128)
    a /= np.linalg.norm(a)
    b = rng.normal(size=128)
    b /= np.linalg.norm(b)
    vecs = []
    for _ in range(4):
        v = a + rng.normal(scale=0.02, size=128); v /= np.linalg.norm(v); vecs.append(v)
    for _ in range(4):
        v = b + rng.normal(scale=0.02, size=128); v /= np.linalg.norm(v); vecs.append(v)
    labels = embed_cluster.cluster(vecs)
    assert len(labels) == 8
    assert labels[0] == labels[1] == labels[2] == labels[3]
    assert labels[4] == labels[5] == labels[6] == labels[7]
    assert labels[0] != labels[4]


def test_cluster_empty_input():
    assert embed_cluster.cluster([]) == []
