import cv2
import numpy as np

from app.pipeline import detect


def test_detect_returns_empty_on_blank_frame():
    frame = np.full((240, 320, 3), 0, dtype=np.uint8)
    assert detect.detect_faces(frame) == []


def test_detect_returns_bboxes_on_real_frame(sample_video):
    cap = cv2.VideoCapture(str(sample_video))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    any_hit = False
    for i in [0, n_frames // 4, n_frames // 2, 3 * n_frames // 4, n_frames - 1]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            continue
        faces = detect.detect_faces(frame)
        if faces:
            any_hit = True
            for f in faces:
                assert "bbox" in f and "score" in f
                x, y, w, h = f["bbox"]
                assert w > 0 and h > 0
                assert 0.0 <= f["score"] <= 1.0
    cap.release()
    assert any_hit, "expected at least one face detection in the sample video"
