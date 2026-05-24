import subprocess
from pathlib import Path

import pytest

from app import ffmpeg_utils


def _make_silent_mp4(path: Path, seconds: int = 2) -> None:
    """Use ffmpeg to synthesize a tiny mp4 for testing."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c=black:s=320x240:r=10:d={seconds}",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
            "-c:v", "libx264", "-c:a", "aac", "-shortest", str(path),
        ],
        check=True, capture_output=True,
    )


def _make_silent_mp4_no_audio(path: Path, seconds: int = 2) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c=black:s=320x240:r=10:d={seconds}",
            "-c:v", "libx264", str(path),
        ],
        check=True, capture_output=True,
    )


def test_probe_returns_fps_duration_resolution(tmp_path):
    src = tmp_path / "src.mp4"
    _make_silent_mp4(src, seconds=2)
    info = ffmpeg_utils.probe(src)
    assert round(info["fps"]) == 10
    assert 1.5 < info["duration_sec"] < 2.5
    assert info["width"] == 320
    assert info["height"] == 240
    assert info["has_audio"] is True


def test_probe_detects_no_audio(tmp_path):
    src = tmp_path / "src.mp4"
    _make_silent_mp4_no_audio(src, seconds=1)
    info = ffmpeg_utils.probe(src)
    assert info["has_audio"] is False


def test_extract_audio_writes_file(tmp_path):
    src = tmp_path / "src.mp4"
    dst = tmp_path / "audio.m4a"
    _make_silent_mp4(src, seconds=2)
    ffmpeg_utils.extract_audio(src, dst)
    assert dst.exists() and dst.stat().st_size > 0


def test_mux_combines_video_and_audio(tmp_path):
    src = tmp_path / "src.mp4"
    audio = tmp_path / "audio.m4a"
    video_only = tmp_path / "video_only.mp4"
    output = tmp_path / "output.mp4"
    _make_silent_mp4(src, seconds=2)
    ffmpeg_utils.extract_audio(src, audio)
    # Pretend we re-encoded the video
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-an", "-c:v", "libx264", str(video_only)],
        check=True, capture_output=True,
    )
    ffmpeg_utils.mux(video_only, audio, output)
    assert output.exists()
    info = ffmpeg_utils.probe(output)
    assert info["has_audio"] is True
