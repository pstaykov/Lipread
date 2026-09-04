# GNet lip-reading — handoff

The best checkpoint of each modality (video-only, audio-only, multimodal), all
trained on the same full **500-class GLips vocabulary**, plus a small local demo
that runs the three checkpoints side by side on 16 real held-out test clips.

This is a demo, not a pitch: the sample is a fixed-seed random draw across 16
distinct classes (not cherry-picked), and one of the 16 clips (`haushalt`) is
misclassified by all three models. See `demo/predictions.json` for the raw output.

## What's here

```
models/
  video_only/    GNet, visual-only.            test:  34.2% top-1 / 53.9% top-5
  audio_only/    Whisper-base probe, audio-only. val:  61.2% top-1 / 76.9% top-5
  multimodal/    GNet + Whisper, fine-tuned end-to-end cross-attention.
                                                 test:  72.0% top-1
demo/            Local site comparing all three on 16 test clips, + live webcam recording.
  server.py        Flask server: static site + live-recording API (needs flask, mediapipe).
  serve.py          Static-only fallback server, no recording feature.
  live_infer.py       Mouth-ROI crop + inference for a freshly recorded clip.
  build_demo_data.py   Dev-only: regenerates predictions.json + demo/clips/*.mp4.
requirements.txt
```

All three numbers above are on the identical 500-class vocabulary and class
ordering (`classes.json` is byte-identical across the three folders), so their
predictions are directly comparable clip-for-clip. `video_only` and `multimodal`
report held-out **test**-split accuracy (matching the paper); `audio_only` is
reported on **val** because it was trained fresh for this handoff (see below) and
hasn't had a separate test-split pass run — the val/test gap for the sibling
probe in the paper was under half a point, so this should track closely.

## Provenance

- `video_only` and `multimodal` are exactly the checkpoints reported in
  `paper/gnet.tex` (500-class rows of Table `tab:main`/`tab:mm`), copied from
  `lipreading/Transformer_based/checkpoints_500/best_model.pth` and
  `multimodal/fusion/models/checkpoints_500/best_model.pth` in the main repo.
- `audio_only` did **not** exist as a full 500-class model before this handoff — the
  paper's Whisper probe was trained excluding 'hier'/'soll', which were dropped for
  an unrelated *visual* mouth-ROI preprocessing gap that never affected audio. Since
  that gap doesn't apply to audio, `train_audio_probe_500.py` reruns the identical
  recipe over all 500 classes so every model in this handoff shares one vocabulary.
  Result: 61.2% val top-1 — matching the earlier probe's 61.4%, confirming the two
  extra classes cost nothing.

Each `models/<name>/` folder contains the exact `model.py` + `train.py` (plus
`dataset.py`/`train_loop.py`/`dataset15.py` where the training script needs them)
that produced its checkpoint, flattened out of the main repo's package structure.
They're reference copies for inspecting how each checkpoint was made, not
turnkey-runnable here: the ~250k-clip GLips corpus and (for audio/multimodal) a
16 GB prebuilt Whisper waveform cache aren't shipped in this folder. Each
`train.py` has a comment block pointing at what it needs and where those paths
lived in the source repo.

## Running the demo

```
cd demo
python server.py       # or `python serve.py` for the static-only version, no live recording
```
Then open `http://localhost:8000` in a browser. (Must be served over HTTP, not
opened as a `file://` path — the page fetches `predictions.json`. Both servers send
`Cache-Control: no-store` so edits to any demo file always show up on the next
reload instead of a stale cached copy.)

**Record your own clip**: `server.py` adds a "Selbst aufnehmen" section — pick a
target word, speak it into the webcam, and the page crops the mouth region with
the exact same MediaPipe FaceMesh pipeline the training data went through
(`lipreading/preprocess_mouth_roi.py`, ported into `demo/live_infer.py`) before
running all three models. It shows the cropped clip back so you can see what the
models actually saw. Needs `flask` + `mediapipe==0.10.21` (see `requirements.txt`)
and a webcam; falls back to hidden if `server.py` isn't running the API.

Each clip shows the mouth-crop video with its original audio muxed back in, the
true word, and each model's top-1 (✓/✗ + confidence) and top-5 breakdown, including
whether the true word is anywhere in the top-5 — audio-only in particular often
ranks the right word 2nd-5th even when its top-1 guess is wrong, so top-1 alone
understates it. Aggregate accuracy over just these 16 clips is shown at the
bottom, separately from the full-test-set numbers quoted at the top — 16 clips is
not a statistically powered comparison, just an illustration of how the three
models actually behave.

**Audio condition selector**: a dropdown switches `audio_only` and `multimodal`
between clean audio and additive white/babble noise at 10dB/5dB SNR (same
noise-mixing formula as the paper's robustness sweep, reused from
`multimodal/fusion/snr_eval.py`). Playback switches too — each of the 16 clips
ships five muxed variants (`demo/clips/<name>_<condition>.mp4`) with the actual
noised audio baked in, so you hear what the models heard, not just their output.
`video_only` is audio-independent, so its prediction (and the video track) stay
the same across every condition — shown explicitly rather than hidden. These are
live measurements made by `build_demo_data.py` on the 500-class checkpoints
shipped here; they are not the paper's own SNR numbers, which were measured on an
earlier frozen-backbone checkpoint. `demo/babble_pool.npz` is a small
(~13MB) sample of real speech waveforms saved alongside, so the live-recording
feature can also mix in babble noise without needing the full dev-only audio cache.

`demo/build_demo_data.py` is the script that generated `predictions.json` and
`demo/clips/*.mp4` — it loads all three models straight from `models/`, runs real
inference (no mocked numbers), and muxes each mouth-crop clip with its original
audio via ffmpeg. It's included for provenance/inspection; rerunning it needs the
full GLips corpus (`lipreading/GLips_mouth/` and `lipreading/GLips/` in the main
repo), so it won't run standalone from inside this handoff folder.

## Caveats (recording feature)

- `mediapipe` must be `0.10.21`, not the current default (`1.0.x` dropped the
  legacy `solutions.face_mesh` API this pipeline uses). If you already have a
  newer mediapipe installed globally for another project, install this repo's
  pin into a virtualenv rather than upgrading/downgrading it system-wide.
- The crop quality (and therefore video-only accuracy) depends on face detection
  succeeding every frame, same as it did for the training data — `server.py`
  reports the detection rate and warns below 50%. Good lighting and a
  front-facing webcam angle help.

## Caveats (carried over from the paper)

- Single-seed models; GLips's stock split is not speaker-disjoint, so these are
  within-corpus rather than speaker-independent numbers.
- The multimodal model's audio branch is Whisper `base`, pretrained mostly on
  English and used frozen/unadapted to German.
- `video_only`'s 34.2% is a full-vocabulary *visual-only* floor — it's the
  weakest of the three by design (that's the point of measuring what fusion adds
  in `multimodal`), not a sign the video pipeline is broken.
