#!/usr/bin/env bash
# One-time setup on an Ubuntu machine without GPU: bash dataset/vm/setup.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

sudo apt-get update -qq
sudo apt-get install -y -qq curl ca-certificates libgl1 libglib2.0-0 >/dev/null
if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

# same versions as the main PC (mediapipe 0.10.21 still has mp.solutions; it needs numpy < 2)
uv venv -q -p 3.11 .venv-mine
PY=.venv-mine/bin/python
uv pip install -q -p $PY numpy==1.26.4 mediapipe==0.10.21 imageio-ffmpeg==0.6.0 protobuf==4.25.9
uv pip install -q -p $PY torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cpu

# catalog part + vocabulary for this machine (bundestag_data/ is gitignored)
mkdir -p bundestag_data/words bundestag_data/clips
gunzip -c dataset/vm/catalog_1of2.jsonl.gz > bundestag_data/catalog.jsonl
cp dataset/vm/vocab.csv bundestag_data/words/vocab.csv

# check imports, ffmpeg, the alignment model (downloaded once) and the Mediathek connection
cd dataset
../.venv-mine/bin/python - <<'PY'
import mediapipe, cv2, imageio_ffmpeg, subprocess
import mine_clips as m
print("ffmpeg", imageio_ffmpeg.get_ffmpeg_exe())
assert "libx264" in subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-encoders"],
                                   capture_output=True, text=True).stdout
m.aligner()
print("MMS_FA loaded")
mp4, mp3 = m.media_urls("7630185")
print("Mediathek:", m.probe(mp4))
PY
echo "setup ok -> start with: bash dataset/vm/run.sh"
