"""Download videos from YouTube (and any other yt-dlp supported site).

Note on Terms of Service: downloading YouTube content can violate YouTube's
ToS. Intended for local processing of videos you own or have rights to.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import yt_dlp

MAX_HEIGHT = 720


def download(
    url: str,
    dst: Path,
    progress_cb: Callable[[float], None] | None = None,
) -> None:
    """Download `url` and produce an mp4 at exactly `dst`.

    Selects the best stream up to MAX_HEIGHT and merges video+audio to mp4
    (requires ffmpeg, which the project already depends on).
    """
    def hook(d: dict) -> None:
        if progress_cb and d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            if total:
                progress_cb(min(1.0, d.get("downloaded_bytes", 0) / total))

    out_base = dst.with_suffix("")  # yt-dlp appends the real container ext
    ydl_opts = {
        "format": f"bestvideo[height<={MAX_HEIGHT}]+bestaudio/best[height<={MAX_HEIGHT}]/best",
        "outtmpl": f"{out_base}.%(ext)s",
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "noplaylist": True,
        "progress_hooks": [hook],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    if dst.exists():
        return
    # Fallback: yt-dlp produced a different extension; locate and rename.
    candidates = sorted(dst.parent.glob(f"{out_base.name}.*"))
    candidates = [c for c in candidates if c != dst]
    if not candidates:
        raise RuntimeError(f"download produced no output file for {url}")
    candidates[0].rename(dst)
