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


def test_extract_audio_transcodes_opus_source(tmp_path):
    """Regression: YouTube delivers Opus, which can't be stream-copied into m4a.
    extract_audio must transcode to AAC instead of copying."""
    src = tmp_path / "src.mkv"  # mkv holds opus + h264 (like a YouTube DASH merge)
    r = subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=black:s=160x120:r=5:d=1",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
            "-c:v", "libx264", "-c:a", "libopus", "-shortest", str(src),
        ],
        capture_output=True,
    )
    if r.returncode != 0:
        pytest.skip("ffmpeg build lacks libopus encoder")
    dst = tmp_path / "audio.m4a"
    ffmpeg_utils.extract_audio(src, dst)  # would raise with -c:a copy
    assert dst.exists() and dst.stat().st_size > 0
    assert ffmpeg_utils.probe(_with_dummy_video(dst, tmp_path))["has_audio"] is True


def _with_dummy_video(audio_m4a: Path, tmp_path: Path) -> Path:
    """probe() needs a video stream; wrap the audio with a 1-frame black video."""
    out = tmp_path / "probe_wrap.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=black:s=16x16:r=1:d=1",
         "-i", str(audio_m4a), "-c:v", "libx264", "-c:a", "copy", "-shortest", str(out)],
        check=True, capture_output=True,
    )
    return out


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


def test_sample_fixture_is_valid(sample_video):
    info = ffmpeg_utils.probe(sample_video)
    assert 4 <= info["duration_sec"] <= 6
    assert info["width"] <= 640
