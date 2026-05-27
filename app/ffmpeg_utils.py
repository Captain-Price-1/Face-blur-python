"""Thin wrappers around the `ffmpeg` and `ffprobe` system binaries."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", "replace")[-2000:].strip()
        raise RuntimeError(f"ffmpeg failed (exit {proc.returncode}): {stderr}")
    return proc


def probe(path: Path) -> dict:
    """Return {fps, duration_sec, width, height, has_audio}."""
    cmd = [
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_streams", "-show_format", str(path),
    ]
    out = _run(cmd).stdout
    data = json.loads(out)
    video = next(s for s in data["streams"] if s["codec_type"] == "video")
    audio = next((s for s in data["streams"] if s["codec_type"] == "audio"), None)
    num, den = (int(x) for x in video["r_frame_rate"].split("/"))
    fps = num / den if den else 0.0
    return {
        "fps": fps,
        "duration_sec": float(data["format"]["duration"]),
        "width": int(video["width"]),
        "height": int(video["height"]),
        "has_audio": audio is not None,
    }


def extract_audio(src: Path, dst: Path) -> None:
    """Extract the source audio to AAC/m4a.

    Transcodes (not stream-copy) so any source codec — including Opus, which
    YouTube commonly delivers and which cannot live in an MP4/m4a container —
    is converted to AAC, which always can.
    """
    _run(["ffmpeg", "-y", "-i", str(src), "-vn", "-c:a", "aac", "-b:a", "192k", str(dst)])


def mux(video_only: Path, audio: Path, dst: Path) -> None:
    """Combine a video-only file with an audio file. No re-encoding."""
    _run([
        "ffmpeg", "-y", "-i", str(video_only), "-i", str(audio),
        "-c:v", "copy", "-c:a", "copy", str(dst),
    ])
