# GNet lip-reading — demo

Compares three checkpoints (video-only, audio-only, multimodal) side by side,
all trained on the same full **500-class GLips vocabulary**, on 16 real
held-out test clips — plus an optional live webcam recording feature. This
folder is fully self-contained: models, code, and sample clips all ship
together, nothing else needs to be downloaded.

This is a demo, not a pitch: the sample is a fixed-seed random draw across 16
distinct classes (not cherry-picked), and one of the 16 clips (`haushalt`) is
misclassified by all three models. See `predictions.json` for the raw output.

## Quick start

**Windows**: double-click `install.bat` (or run `.\install.ps1` in
PowerShell).

**macOS / Linux**: `./install.sh`

That's it — it creates an isolated Python environment, installs everything
needed, starts the server, and opens the demo in your browser automatically
(`http://localhost:8000`). First run takes a few minutes (downloading
PyTorch etc.); after that, re-running it starts in seconds. On Windows the
server keeps running in the background after the installer window closes —
stop it with `Stop-Process -Id <pid>` (the PID is printed at the end) or just
run the installer again, which replaces it. On macOS/Linux, press **Ctrl+C**
in the terminal to stop it.

Requires Python 3.9+ on the machine (get it from
[python.org](https://www.python.org/downloads/) if needed — on Windows, tick
"Add python.exe to PATH" in the installer) — everything else is installed
automatically into a local `venv/` folder inside this directory, so it won't
touch or conflict with anything else on the machine.

If the optional live-recording extras (`mediapipe`/`opencv`/`flask`) fail to
install — this happens on some newer Python versions, since `mediapipe`'s pin
doesn't always have a wheel for them — the script falls back automatically to
the static comparison demo (the 16 test clips + noise conditions), just
without the "Selbst aufnehmen" webcam section.

### Manual start (if you'd rather not use install.sh)

```
python -m venv venv
venv/bin/activate          # or venv\Scripts\activate on Windows
pip install -r requirements.txt
python server.py           # or `python serve.py` for no live recording
```
Then open `http://localhost:8000`. (Must be served over HTTP, not opened as a
`file://` path — the page fetches `predictions.json`.)

## What's here

```
install.sh / install.ps1 / install.bat   One-command setup + launch (see above).
requirements.txt      Full dependency list, for manual installs.
index.html / app.js / style.css   The demo page.
server.py             Flask server: static site + live-recording API.
serve.py              Static-only fallback server, no recording feature.
live_infer.py         Mouth-ROI crop + inference for a freshly recorded clip.
predictions.json      Precomputed predictions for the 16 shipped test clips.
clips/                The 16 test clips, each in 5 audio conditions.
babble_pool.npz       Small sample of real speech, for babble-noise mixing.
models/
  video_only/    GNet, visual-only.            test:  34.2% top-1 / 53.9% top-5
  audio_only/    Whisper-base probe, audio-only. val:  61.2% top-1 / 76.9% top-5
  multimodal/    GNet + Whisper, fine-tuned end-to-end cross-attention.
                                                 test:  72.0% top-1
```

All three numbers above are on the identical 500-class vocabulary and class
ordering (`classes.json` is byte-identical across the three folders), so their
predictions are directly comparable clip-for-clip. `video_only` and
`multimodal` report held-out **test**-split accuracy; `audio_only` is reported
on **val** (the val/test gap for the sibling probe was under half a point).

Each `models/<name>/` folder contains only the code needed to actually run
that checkpoint here (flattened out of the main repo's package structure) —
the full training scripts and provenance details live in the main project
repo, not in this handoff.

## Using the demo

Each clip shows the mouth-crop video with its original audio muxed back in,
the true word, and each model's top-1 (✓/✗ + confidence) and top-5 breakdown,
including whether the true word is anywhere in the top-5 — audio-only in
particular often ranks the right word 2nd–5th even when its top-1 guess is
wrong. Aggregate accuracy over just these 16 clips is shown at the bottom,
separately from the full-test-set numbers quoted at the top.

**Selbst aufnehmen (live recording)**: pick a target word, speak it into the
webcam, and the page crops the mouth region with the same MediaPipe FaceMesh
pipeline the training data went through before running all three models. It
shows the cropped clip back so you can see what the models actually saw.

**Audio condition selector**: a dropdown switches `audio_only` and
`multimodal` between clean audio and additive white/babble noise at 10dB/5dB
SNR. Playback switches too — each clip ships five muxed variants with the
actual noised audio baked in, so you hear what the models heard.
`video_only` is audio-independent, so its prediction stays the same across
every condition.

## Caveats

- `mediapipe` must be `0.10.21` — newer `1.0.x` releases dropped the legacy
  `solutions.face_mesh` API this pipeline uses, and that exact pin isn't
  always available as a wheel for newer Python versions. The installer
  scripts handle this automatically (falling back to the static comparison
  demo, no "Selbst aufnehmen" section, rather than failing outright).
- Live-recording crop quality (and therefore video-only accuracy) depends on
  face detection succeeding every frame. Good lighting and a front-facing
  webcam angle help; `server.py` reports the detection rate and warns below
  50%.
- Single-seed models; GLips's stock split is not speaker-disjoint, so these
  are within-corpus rather than speaker-independent numbers.
- The multimodal model's audio branch is Whisper `base`, pretrained mostly on
  English and used frozen/unadapted to German.
- `video_only`'s 34.2% is a full-vocabulary *visual-only* floor — it's the
  weakest of the three by design (that's the point of measuring what fusion
  adds in `multimodal`), not a sign the video pipeline is broken.
