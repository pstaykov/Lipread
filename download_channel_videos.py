#!/usr/bin/env python3
"""Download the first N videos from a YouTube channel or uploads page."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable, List, Optional
import shutil
import subprocess

import yt_dlp


YOUTUBE_WATCH_URL = "https://www.youtube.com/watch?v={video_id}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a fixed number of videos from a YouTube channel into a folder."
    )
    parser.add_argument(
        "source_url",
        help="YouTube channel, user, handle, or uploads/playlists URL.",
    )
    parser.add_argument(
        "count",
        type=int,
        help="Number of videos to download.",
    )
    parser.add_argument(
        "output_dir",
        help="Folder where the videos will be saved.",
    )
    parser.add_argument(
        "--oldest-first",
        action="store_true",
        help="Download the oldest videos first instead of the newest ones first.",
    )
    parser.add_argument(
        "--use-cropper",
        action="store_true",
        help="Run a face-cropping script on each downloaded video if available.",
    )
    parser.add_argument(
        "--cropper",
        help="Path to a camera_preview.py cropper script to run on each video.",
        default=None,
    )
    parser.add_argument(
        "--center-crop-ratio",
        type=float,
        default=0.6,
        help=(
            "Before face tracking, keep only the center portion of each frame "
            "(0-1]. Use 1.0 to disable center precropping."
        ),
    )
    parser.add_argument(
        "--download-audio",
        action="store_true",
        help="Also download the audio track for each video.",
    )
    parser.add_argument(
        "--audio-format",
        choices=("m4a", "mp3"),
        default="m4a",
        help="Audio format to save (requires ffmpeg for mp3 conversion).",
    )
    return parser.parse_args()


def resolve_video_url(entry: dict) -> Optional[str]:
    if entry.get("webpage_url"):
        return entry["webpage_url"]
    if entry.get("url") and entry.get("url", "").startswith("http"):
        return entry["url"]

    video_id = entry.get("id") or entry.get("url")
    if not video_id:
        return None
    return YOUTUBE_WATCH_URL.format(video_id=video_id)


def extract_entries(source_url: str) -> List[dict]:
    extract_options = {
        "quiet": True,
        "skip_download": True,
        "extract_flat": True,
        "noplaylist": False,
    }
    with yt_dlp.YoutubeDL(extract_options) as ydl:
        info = ydl.extract_info(source_url, download=False)
    # If `info` is a playlist or channel, it will have an `entries` list.
    # For single-video URLs `info` is the video dict itself, so handle that.
    entries = info.get("entries") if isinstance(info, dict) else None
    if not entries:
        # treat the info dict itself as a single entry if it looks like one
        if isinstance(info, dict) and (info.get("id") or info.get("webpage_url")):
            return [info]
        return []

    return [entry for entry in entries if entry]


def select_entries(entries: List[dict], count: int, oldest_first: bool) -> List[dict]:
    filtered = list(entries)
    if oldest_first:
        filtered = list(reversed(filtered))
    return filtered[:count]


def download_videos(video_urls: Iterable[str], output_dir: Path):
    """Yield downloaded file paths for each video URL.

    The function yields the resulting downloaded `Path` for each provided URL so
    the caller can optionally run the cropper and move the final file where
    desired.
    """
    output_template = str(output_dir / "%(title).200s [%(id)s].%(ext)s")
    # If ffmpeg is available we can safely request separate video+audio
    # and let yt-dlp merge them. If ffmpeg is missing, request a single
    # file format to avoid yt-dlp aborting due to merge requirements.
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        fmt = "bestvideo[height<=720]+bestaudio/best[height<=720]/best"
    else:
        fmt = "best[height<=720]/best"

    download_options = {
        "outtmpl": output_template,
        "format": fmt,
        "quiet": False,
        "noplaylist": True,
    }

    fragment_pattern = re.compile(r"\.f\d+\.")

    def resolve_downloaded_path(info: dict) -> Optional[Path]:
        requested_downloads = info.get("requested_downloads") if isinstance(info, dict) else None
        if requested_downloads:
            for requested in requested_downloads:
                filepath = requested.get("filepath") if isinstance(requested, dict) else None
                if filepath:
                    candidate = Path(filepath).expanduser().resolve()
                    if candidate.exists() and not fragment_pattern.search(candidate.name) and not candidate.name.endswith(".part"):
                        return candidate

        video_id = info.get("id") if isinstance(info, dict) else None
        if not video_id:
            return None

        matching_files = []
        for candidate in output_dir.iterdir():
            if video_id not in candidate.name:
                continue
            if fragment_pattern.search(candidate.name) or candidate.name.endswith(".part"):
                continue
            if candidate.is_file():
                matching_files.append(candidate)

        if not matching_files:
            return None

        return max(matching_files, key=lambda item: item.stat().st_mtime)

    for video_url in video_urls:
        if not video_url:
            continue
        print(f"Downloading: {video_url}")

        with yt_dlp.YoutubeDL(download_options) as ydl:
            info = ydl.extract_info(video_url, download=True)

        downloaded_path = resolve_downloaded_path(info if isinstance(info, dict) else {})
        if downloaded_path is None:
            print("Could not determine downloaded filename for", video_url, file=sys.stderr)
            continue
        yield video_url, downloaded_path


def tracked_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_tracked.mp4")


def main() -> int:
    args = parse_args()

    if args.count <= 0:
        print("count must be greater than 0", file=sys.stderr)
        return 2
    if args.center_crop_ratio <= 0 or args.center_crop_ratio > 1.0:
        print("--center-crop-ratio must be in the range (0, 1].", file=sys.stderr)
        return 2

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        entries = extract_entries(args.source_url)
    except yt_dlp.utils.DownloadError as exc:
        print(f"Failed to read channel videos: {exc}", file=sys.stderr)
        return 1

    if not entries:
        print("No videos were found for the provided URL.", file=sys.stderr)
        return 1

    selected = select_entries(entries, args.count, args.oldest_first)
    video_urls = [resolve_video_url(entry) for entry in selected]
    video_urls = [url for url in video_urls if url]

    if not video_urls:
        print("No downloadable video URLs were resolved.", file=sys.stderr)
        return 1

    should_run_cropper = args.use_cropper or args.cropper is not None or args.center_crop_ratio < 1.0

    # determine cropper script path if requested
    cropper_path: Optional[Path] = None
    cropper_python = Path(__file__).resolve().parent / ".venv" / "Scripts" / "python.exe"
    if args.cropper:
        cropper_path = Path(args.cropper).expanduser().resolve()
        if not cropper_path.is_file():
            print(f"Cropper script not found: {cropper_path}", file=sys.stderr)
            cropper_path = None
    elif should_run_cropper:
        possible = Path(__file__).resolve().parent / "Lipread-main" / "cameraTracker" / "cameraTracker" / "camera_preview.py"
        if possible.is_file():
            cropper_path = possible
        else:
            print("No cropper found at default path; continuing without cropper.", file=sys.stderr)

    print(f"Downloading {len(video_urls)} video(s) into {output_dir}")

    for video_url, downloaded_path in download_videos(video_urls, output_dir):
        try:
            if cropper_path is not None:
                python_executable = cropper_python if cropper_python.is_file() else Path(sys.executable)
                if args.center_crop_ratio < 1.0:
                    print(
                        f"Cropping face track with center pre-crop ({args.center_crop_ratio:.2f}): {downloaded_path.name}"
                    )
                else:
                    print(f"Cropping face track: {downloaded_path.name}")
                proc = subprocess.run(
                    [
                        str(python_executable),
                        str(cropper_path),
                        "--prefer-gpu",
                        "--center-crop-ratio",
                        str(args.center_crop_ratio),
                        str(downloaded_path),
                    ]
                )
                if proc.returncode != 0:
                    print(f"Cropper failed for {downloaded_path}", file=sys.stderr)
                    target = output_dir / downloaded_path.name
                    shutil.move(str(downloaded_path), str(target))
                    continue

                out_path = tracked_output_path(downloaded_path)
                if out_path.is_file():
                    target = output_dir / out_path.name
                    shutil.move(str(out_path), str(target))
                    if downloaded_path.exists():
                        downloaded_path.unlink(missing_ok=True)
                    # optionally download audio for the output file's URL as well
                    if args.download_audio:
                        download_audio(video_url, output_dir, args.audio_format)
                    continue

                target = output_dir / downloaded_path.name
                shutil.move(str(downloaded_path), str(target))
                if args.download_audio:
                    download_audio(video_url, output_dir, args.audio_format)
            else:
                target = output_dir / downloaded_path.name
                shutil.move(str(downloaded_path), str(target))
                if args.download_audio:
                    download_audio(video_url, output_dir, args.audio_format)
        except Exception as exc:
            print(f"Error handling downloaded file {downloaded_path}: {exc}", file=sys.stderr)
    return 0


def download_audio(video_url: str, output_dir: Path, audio_format: str = "m4a") -> None:
    """Download the audio track for a given video URL into output_dir.

    If `audio_format` is "mp3", this attempts to convert using ffmpeg. If
    ffmpeg is unavailable the function falls back to downloading `m4a`.
    """
    audio_template = str(output_dir / "%(title).200s [%(id)s].%(ext)s")
    if audio_format == "mp3":
        # prefer best audio, then convert to mp3 with ffmpeg
        ydl_opts = {
            "outtmpl": audio_template,
            "format": "bestaudio/best",
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192",
                }
            ],
            "quiet": False,
            "noplaylist": True,
        }

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            print("ffmpeg not found; falling back to m4a audio download.", file=sys.stderr)
            audio_format = "m4a"

    if audio_format == "m4a":
        ydl_opts = {
            "outtmpl": audio_template,
            "format": "bestaudio[ext=m4a]/bestaudio/best",
            "quiet": False,
            "noplaylist": True,
        }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
    except yt_dlp.utils.DownloadError as exc:
        print(f"Failed to download audio for {video_url}: {exc}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
