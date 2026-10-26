"""Environment check for a mining machine: imports, ffmpeg with x264, alignment model, Mediathek access."""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cv2  # noqa: E402,F401
import mediapipe as mp  # noqa: E402

import mine_clips as m  # noqa: E402
from mine_bundestag import ffmpeg_exe  # noqa: E402

assert hasattr(mp, "solutions"), "mediapipe without mp.solutions (needs 0.10.21)"
enc = subprocess.run([ffmpeg_exe(), "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
assert "libx264" in enc, "ffmpeg without libx264"
print("ffmpeg ok:", ffmpeg_exe())
m.aligner()
print("alignment model ok")
mp4, mp3 = m.media_urls("7630185")
print("Mediathek ok:", m.probe(mp4))
print("check ok")
