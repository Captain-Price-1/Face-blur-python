import numpy as np

from app.pipeline import blur


def _solid_frame(value: int = 200, h: int = 240, w: int = 320) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.uint8)


def _checkerboard(h: int = 240, w: int = 320, sq: int = 8) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(0, h, sq):
        for x in range(0, w, sq):
            if ((x // sq) + (y // sq)) % 2 == 0:
                img[y:y+sq, x:x+sq] = 255
    return img


def test_blur_reduces_variance_inside_bbox():
    frame = _checkerboard()
    bbox = (100, 60, 80, 80)  # x, y, w, h
    before = frame[60:140, 100:180].var()
    blur.apply_gaussian_blur(frame, bbox)
    after = frame[60:140, 100:180].var()
    assert after < before * 0.5


def test_blur_does_not_change_outside_bbox():
    frame = _checkerboard()
    bbox = (100, 60, 80, 80)
    before_outside = frame[:30, :].copy()
    blur.apply_gaussian_blur(frame, bbox)
    np.testing.assert_array_equal(before_outside, frame[:30, :])


def test_blur_clamps_to_frame_bounds():
    frame = _solid_frame()
    # bbox extending past the right edge — should not raise
    blur.apply_gaussian_blur(frame, (300, 200, 100, 100))


def test_blur_handles_zero_sized_bbox():
    frame = _solid_frame()
    blur.apply_gaussian_blur(frame, (10, 10, 0, 0))  # no-op, no crash
