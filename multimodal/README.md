# Multimodal lip reading — visual GLipsNet + Whisper audio (cross-attention)

Fine-tunes the pretrained visual encoder with audio features from the Whisper `base`
encoder, fused into the visual token stream via gated cross-attention.

## Data layout (important)

- **Video**: mouth-crop mp4s in `../lipreading/GLips_mouth/lipread_files/<class>/<split>/<name>.mp4`.
  These are **video-only** — the mouth-cropping step dropped the audio.
- **Audio**: sibling `.m4a` files in the original tree
  `../lipreading/GLips/lipread_files/<class>/<split>/<name>.m4a` (same class/split/name).
  The dataset maps each clip to its `.m4a` automatically.
- **Visual checkpoint** (required): `../lipreading/Transformer_based/checkpoints_500/best_model.pth`.
  Training asserts on this at startup and partially loads it (visual tensors; the
  cross-attention keys start random).

## Audio cache (recommended)

GLips clips are ~1.2 s, so Whisper's default 30 s padding is ~96% silence. `train.py`
trims the mel to 2 s (100 encoder tokens) and runs Whisper **batched on the GPU** in the
loop. To avoid decoding a `.m4a` per step, prebuild a waveform cache:

```bash
pip install av            # in-process AAC decode, ~4x faster than an ffmpeg subprocess
python build_audio_cache.py
```

Produces `audio_cache/waveforms.dat` (~16 GB fp16 memmap, all 249,198 clips, split-agnostic)
and `audio_cache/index.json`. `train.py` auto-detects it; on a cache miss it falls back to
on-the-fly ffmpeg decode. The cache is portable across OSes (raw fp16 bytes + forward-slash
keys) — copy the `audio_cache/` dir rather than rebuilding. Delete it anytime to reclaim disk.

## Train

```bash
pip install openai-whisper
python train.py
```

- **Split**: `group_split=False` (stock on-disk train/val/test folders) to match the GLips
  paper, which did not do source-disjoint grouping. ~199k train samples.
- Checkpoints in `./checkpoints/`: `best_model.pth` (best val top-1), `checkpoint_latest.pth`
  (resume), periodic snapshots, `metrics.csv`. Re-running resumes model weights from
  `checkpoint_latest.pth` (optimizer/LR schedule restart from epoch 0).

## Performance notes

- Bottleneck is **CPU mp4 decode** (194k tiny ~7 KB files); the GPU is under-fed because the
  model is small. The audio cache removes the audio half of the decode cost.
- **On Linux**: bump `NUM_WORKERS` (currently 4 — forked workers are cheap) and `torch.compile`
  auto-activates (it's gated behind `os.name != 'nt'`). For a bigger win, use a torchvision
  build with GPU/NVDEC video decode (unavailable in the Windows wheels — no `VideoReader`).

## What each file does

- `train.py` — model, dataset (video + cached/`.m4a` audio), batched mel-trimmed Whisper
  encoder, training loop.
- `build_audio_cache.py` — one-time `.m4a` → fixed-2 s fp16 waveform memmap builder (pyav).
- `test.py` — evaluation.
