"""Per-frame face blur using OpenCV's YuNet detector.

Usage:
    python scripts/quick_blur.py INPUT.mp4 OUTPUT.mp4

YuNet is a tiny (~227 KB), fast, multi-scale face detector bundled with
OpenCV. Detects on every frame, no persistence/extrapolation — a frame is
blurred only when the detector actually sees a face in it.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2

from app.pipeline.blur import apply_gaussian_blur

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "face_detection_yunet_2023mar.onnx"
SCORE_THRESHOLD = 0.6   # detection confidence; lower = more recall, more false positives
NMS_THRESHOLD = 0.3
TOP_K = 5000


def has_audio(path: Path) -> bool:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=index", "-of", "csv=p=0", str(path)],
        check=True, capture_output=True, text=True,
    )
    return bool(r.stdout.strip())


def main(src: Path, dst: Path) -> None:
    if not src.exists():
        sys.exit(f"input not found: {src}")
    if not MODEL_PATH.exists():
        sys.exit(f"YuNet model not found at {MODEL_PATH}")

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        sys.exit(f"cannot open: {src}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    detector = cv2.FaceDetectorYN.create(
        str(MODEL_PATH), "", (w, h), SCORE_THRESHOLD, NMS_THRESHOLD, TOP_K
    )

    tmp = Path(tempfile.mkdtemp())
    video_only = tmp / "video_only.mp4"
    writer = cv2.VideoWriter(
        str(video_only), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )

    frame_idx = 0
    frames_with_face = 0
    total_blurs = 0
    print(f"Detecting + blurring {total} frames at {w}x{h}…")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        _, faces = detector.detect(frame)
        if faces is not None:
            frames_with_face += 1
            for face in faces:
                x, y, fw, fh = (int(v) for v in face[:4])
                if fw > 0 and fh > 0:
                    apply_gaussian_blur(frame, (x, y, fw, fh))
                    total_blurs += 1
        writer.write(frame)
        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"  {frame_idx}/{total} frames; {frames_with_face} with face; {total_blurs} blurs")
    cap.release()
    writer.release()
    print(f"\nDone: {frames_with_face}/{total} frames had a detected face; "
          f"{total_blurs} total blur applications.")

    if has_audio(src):
        print("Muxing original audio…")
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(video_only), "-i", str(src),
             "-c:v", "copy", "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0",
             "-shortest", str(dst)],
            check=True, capture_output=True,
        )
    else:
        shutil.move(str(video_only), str(dst))
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"Output: {dst}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python scripts/quick_blur.py INPUT.mp4 OUTPUT.mp4")
    main(Path(sys.argv[1]).expanduser(), Path(sys.argv[2]).expanduser())
