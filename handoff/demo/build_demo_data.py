"""Generates the assets the static demo site (index.html) consumes: a small set
of real GLips test clips, muxed to a single playable mp4 (mouth-crop video +
original audio), plus predictions.json holding each of the three packaged
models' actual top-5 output on each clip -- at clean audio AND under additive
noise at several SNRs (white + babble), reusing the exact noise-mixing formula
from multimodal/fusion/snr_eval.py. video_only doesn't consume audio, so it is
computed once per clip and reused across every noise condition.

These are fresh, live measurements made by this script (not numbers copied from
the paper): the paper's SNR sweep ran on earlier frozen-backbone fusion heads, not
the fine-tuned checkpoints shipped in this handoff, and the full 500-class audio
probe didn't exist before this handoff either.

This is a DEV-ONLY build script — it reads the full ~250k-clip GLips corpus and
the 16GB audio waveform cache from the main repo, neither of which is shipped in
this handoff. It is included so the provenance of every number in the demo is
inspectable, not because a recipient of this folder needs to (or can) rerun it.
The demo site itself only needs clips/ and predictions.json, both already
generated.

Run from the main BWKI repo root (not from inside handoff/):
    python handoff/demo/build_demo_data.py
"""
import os
import sys
import json
import wave
import random
import subprocess
import importlib.util

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HANDOFF = os.path.join(REPO_ROOT, 'handoff')
VIDEO_ROOT = os.path.join(REPO_ROOT, 'lipreading', 'GLips_mouth', 'lipread_files')
AUDIO_ROOT = os.path.join(REPO_ROOT, 'lipreading', 'GLips', 'lipread_files')
AUDIO_CACHE_DIR = os.path.join(REPO_ROOT, 'multimodal', 'fusion', 'cache', 'audio_cache')
OUT_DIR = os.path.join(HANDOFF, 'demo')
CLIPS_DIR = os.path.join(OUT_DIR, 'clips')
N_CLIPS = 16
SEED = 7
NOISE_SEED = 1234
BABBLE_POOL_SIZE = 400

# (key, label, noise kind, snr_db) -- None snr_db means clean (no noise mixed in)
NOISE_CONDITIONS = [
    ('clean', 'Sauber', None, None),
    ('white_10db', 'Weißes Rauschen, 10 dB SNR', 'white', 10),
    ('white_5db', 'Weißes Rauschen, 5 dB SNR', 'white', 5),
    ('babble_10db', 'Stimmengewirr, 10 dB SNR', 'babble', 10),
    ('babble_5db', 'Stimmengewirr, 5 dB SNR', 'babble', 5),
]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# load each packaged model from the handoff copies themselves, so the demo
# also doubles as a smoke test that the handed-off code actually runs standalone.
video_dir = os.path.join(HANDOFF, 'models', 'video_only')
audio_dir = os.path.join(HANDOFF, 'models', 'audio_only')
mm_dir = os.path.join(HANDOFF, 'models', 'multimodal')

vmod = _load_module('video_model', os.path.join(video_dir, 'model.py'))
vdata = _load_module('video_dataset', os.path.join(video_dir, 'dataset.py'))
amod = _load_module('audio_train', os.path.join(audio_dir, 'train.py'))
mmod = _load_module('mm_model', os.path.join(mm_dir, 'model.py'))

with open(os.path.join(video_dir, 'classes.json'), encoding='utf-8') as f:
    CLASSES = json.load(f)
assert CLASSES == json.load(open(os.path.join(audio_dir, 'classes.json'), encoding='utf-8')) == \
    json.load(open(os.path.join(mm_dir, 'classes.json'), encoding='utf-8')), \
    "the three models must share the exact same class list for a fair side-by-side"
NUM_CLASSES = len(CLASSES)
print(f"classes: {NUM_CLASSES}")

# video-only GNet
video_model = vmod.GLipsNet(num_classes=NUM_CLASSES, pool='attn').to(device).eval()
video_model.load_state_dict(vmod._strip_orig_mod(
    torch.load(os.path.join(video_dir, 'best_model.pth'), map_location=device)))

# audio-only probe + its Whisper backbone
audio_ckpt = torch.load(os.path.join(audio_dir, 'probe.pth'), map_location=device)
audio_probe = amod.AudioProbe(num_classes=NUM_CLASSES).to(device).eval()
audio_probe.load_state_dict(audio_ckpt['state'])
print(f"audio probe checkpoint val_top1={audio_ckpt.get('val_top1')}")

# multimodal fused model (also serves as the audio feature extractor backbone)
extractor = mmod.WhisperExtractor(mmod.WHISPER_MODEL_NAME, device=device)
mm_model = mmod.GLipsNet(num_classes=NUM_CLASSES, use_audio=True).to(device).eval()
mm_model.load_state_dict(mmod._strip_orig_mod(
    torch.load(os.path.join(mm_dir, 'best_model.pth'), map_location=device)))

val_transform = vdata.VideoAugment(crop_size=88, resize_size=96, is_train=False)


# pick a small, honest (not cherry-picked) sample of test clips spanning
# distinct classes, using the exact stock split (group_split=False) all three
# checkpoints were evaluated on.
random.seed(SEED)
all_classes = sorted(CLASSES)
chosen_classes = random.sample(all_classes, N_CLIPS)
samples = []
for cls in chosen_classes:
    test_dir = os.path.join(VIDEO_ROOT, cls, 'test')
    if not os.path.isdir(test_dir):
        continue
    files = sorted(f for f in os.listdir(test_dir) if f.endswith('.mp4'))
    if not files:
        continue
    fname = random.choice(files)
    samples.append((cls, os.path.join(test_dir, fname)))
print(f"selected {len(samples)} clips across {len(set(c for c, _ in samples))} classes")


def load_video_tensor(path):
    video, _ = vdata._load_video_audio(path)
    video = video.float() / 255.0
    T = video.size(0)
    n = 25
    if T > n:
        idx = np.linspace(0, T - 1, num=n).astype(int)
        video = video[idx]
    elif T < n:
        pad = n - T
        video = torch.cat([video, video[-1:].repeat(pad, 1, 1, 1)], dim=0)
    video = val_transform(video)
    return video.permute(1, 0, 2, 3)  # C,T,H,W


def load_audio_wav(cls, split, fname):
    m4a = os.path.join(AUDIO_ROOT, cls, split, os.path.splitext(fname)[0] + '.m4a')
    try:
        import whisper
        wav = whisper.load_audio(m4a)
    except Exception as e:
        print(f"  WARNING: could not load audio for {m4a}: {e}")
        wav = None
    return mmod._fix_wav(wav)


def load_babble_pool():
    """A fixed sample of TRAIN-split waveforms (never the clips being evaluated)
    to mix into babble noise, exactly as multimodal/fusion/snr_eval.py does."""
    index_path = os.path.join(AUDIO_CACHE_DIR, 'index.json')
    if not os.path.exists(index_path):
        print(f"  no audio cache at {AUDIO_CACHE_DIR}; babble noise will fall back to white noise")
        return None
    with open(index_path) as f:
        index = json.load(f)
    wave_mm = np.memmap(os.path.join(AUDIO_CACHE_DIR, 'waveforms.dat'), dtype=np.float16,
                        mode='r', shape=(index['n'], index['audio_samples']))
    train_rows = sorted(r for k, r in index['rows'].items() if k.split('/')[1] == 'train')
    rng = np.random.default_rng(NOISE_SEED)
    pool_rows = rng.choice(train_rows, size=min(BABBLE_POOL_SIZE, len(train_rows)), replace=False)
    pool = torch.from_numpy(np.asarray(wave_mm[np.sort(pool_rows)], dtype=np.float32)).to(device)
    print(f"babble pool: {tuple(pool.shape)}")
    return pool


def add_noise(wav, noise, snr_db):
    """Scale `noise` to hit `snr_db` against `wav`, then mix. Same formula as
    multimodal/fusion/snr_eval.py: SNR = 10*log10(P_signal / P_noise)."""
    p_sig = wav.pow(2).mean(dim=1, keepdim=True)
    p_noi = noise.pow(2).mean(dim=1, keepdim=True).clamp_min(1e-12)
    gain = torch.sqrt(p_sig / (p_noi * (10.0 ** (snr_db / 10.0))))
    return wav + gain * noise


def make_noisy_wav(wav, kind, snr_db, babble_pool, gen):
    if kind is None:
        return wav
    if kind == 'white' or babble_pool is None:
        noise = torch.randn(wav.shape, generator=gen, device=wav.device, dtype=wav.dtype)
    else:
        n_speakers = 5
        idx = torch.randint(0, len(babble_pool), (wav.shape[0], n_speakers),
                            generator=gen, device=wav.device)
        noise = babble_pool[idx].mean(dim=1)
    return add_noise(wav, noise, snr_db)


CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


def topk(probs, k=5):
    vals, idx = probs.topk(k)
    return [{'word': CLASSES[i], 'prob': round(float(v), 4)} for v, i in zip(vals.tolist(), idx.tolist())]


def mux_clip(video_path, audio_path, out_path):
    """Mux the mouth-crop (silent) video with an audio file (.m4a or .wav) into
    one playable mp4, trimmed to the shorter of the two streams."""
    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ffmpeg, '-y', '-i', video_path, '-i', audio_path,
           '-map', '0:v:0', '-map', '1:a:0',
           '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23',
           '-c:a', 'aac', '-shortest', out_path]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def write_wav_int16(path, wav_np, sr=16000):
    """wav_np: 1-D float32 array, roughly [-1, 1] -- write as 16-bit PCM mono."""
    pcm = np.clip(wav_np, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    with wave.open(path, 'wb') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(pcm.tobytes())


babble_pool = load_babble_pool()
gen = torch.Generator(device=device).manual_seed(NOISE_SEED)

if babble_pool is not None:
    # ship a copy so live_infer.py can do babble noise on self-recordings too,
    # without needing the 16GB dev-only audio cache this pool was drawn from.
    np.savez_compressed(os.path.join(OUT_DIR, 'babble_pool.npz'),
                        wav=babble_pool.cpu().numpy().astype(np.float16))
    print(f"wrote {OUT_DIR}/babble_pool.npz {tuple(babble_pool.shape)}")


def entry(probs, cls):
    top5 = topk(probs, 5)
    return {
        'top5': top5,
        'top1': top5[0]['word'],
        'correct': top5[0]['word'] == cls,
        'top5_correct': any(item['word'] == cls for item in top5),
        # confidence the model actually assigned to the TRUE word, even when it
        # wasn't the top pick -- what the red/yellow/green heatmap in the UI shows.
        'true_word_prob': round(float(probs[CLASS_TO_IDX[cls]]), 4),
    }


os.makedirs(CLIPS_DIR, exist_ok=True)
results = []
for i, (cls, video_path) in enumerate(samples):
    fname = os.path.basename(video_path)
    print(f"[{i+1}/{len(samples)}] {cls} / {fname}")

    video_t = load_video_tensor(video_path).unsqueeze(0).to(device)
    clean_wav = load_audio_wav(cls, 'test', fname).unsqueeze(0).to(device)

    with torch.no_grad():
        v_logits = video_model(video_t)
        v_probs = F.softmax(v_logits, dim=1)[0].cpu()

    m4a_path = os.path.join(AUDIO_ROOT, cls, 'test', os.path.splitext(fname)[0] + '.m4a')

    audio_only_by_cond, multimodal_by_cond, clip_files = {}, {}, {}
    for key, label, kind, snr in NOISE_CONDITIONS:
        wav_t = make_noisy_wav(clean_wav, kind, snr, babble_pool, gen)
        with torch.no_grad():
            audio_tokens = extractor.encode(wav_t)              # (1, T, 512)
            a_logits = audio_probe(audio_tokens.mean(dim=1))
            a_probs = F.softmax(a_logits, dim=1)[0].cpu()

            mm_logits = mm_model(video_t, audio_tokens)
            mm_probs = F.softmax(mm_logits, dim=1)[0].cpu()

        audio_only_by_cond[key] = entry(a_probs, cls)
        multimodal_by_cond[key] = entry(mm_probs, cls)

        # so the demo can actually play back what each condition sounds like,
        # not just report the model's prediction under it.
        out_name = f"{i:02d}_{cls}.mp4" if key == 'clean' else f"{i:02d}_{cls}_{key}.mp4"
        out_path = os.path.join(CLIPS_DIR, out_name)
        try:
            if key == 'clean':
                mux_clip(video_path, m4a_path, out_path)   # full-fidelity original audio
            else:
                wav_path = os.path.join(CLIPS_DIR, f"_tmp_{i:02d}_{key}.wav")
                write_wav_int16(wav_path, wav_t[0].cpu().numpy())
                mux_clip(video_path, wav_path, out_path)
                os.remove(wav_path)
            clip_files[key] = out_name
        except Exception as e:
            print(f"  WARNING: mux failed for {key} ({e}); reusing clean clip")
            clip_files[key] = clip_files.get('clean', f"{i:02d}_{cls}.mp4")

    results.append({
        'id': i,
        'clip_files': clip_files,
        'source_file': fname,
        'ground_truth': cls,
        'video_only': entry(v_probs, cls),   # audio-independent: same across all noise conditions
        'audio_only': audio_only_by_cond,
        'multimodal': multimodal_by_cond,
    })

with open(os.path.join(OUT_DIR, 'predictions.json'), 'w', encoding='utf-8') as f:
    json.dump({
        'classes_n': NUM_CLASSES,
        'noise_conditions': [{'key': k, 'label': lbl} for k, lbl, _, _ in NOISE_CONDITIONS],
        'model_headline_accuracy': {
            'video_only': {'test_top1': 0.342, 'test_top5': 0.539, 'label': 'GNet, visual-only, 500-class GLips'},
            'audio_only': {'val_top1': round(float(audio_ckpt.get('val_top1', 0)), 4),
                          'val_top5': round(float(audio_ckpt.get('val_top5', 0)), 4),
                          'label': 'Whisper-base probe, audio-only, 500-class GLips'},
            'multimodal': {'test_top1': 0.720, 'label': 'GNet + Whisper, fine-tuned end-to-end cross-attention, 500-class GLips'},
        },
        'clips': results,
    }, f, ensure_ascii=False, indent=1)

n = len(results)
print()
for key_id, label, _, _ in NOISE_CONDITIONS:
    v1 = sum(r['video_only']['correct'] for r in results) / n
    v5 = sum(r['video_only']['top5_correct'] for r in results) / n
    a1 = sum(r['audio_only'][key_id]['correct'] for r in results) / n
    a5 = sum(r['audio_only'][key_id]['top5_correct'] for r in results) / n
    m1 = sum(r['multimodal'][key_id]['correct'] for r in results) / n
    m5 = sum(r['multimodal'][key_id]['top5_correct'] for r in results) / n
    print(f"{label:>24}: video={v1:.2f}/{v5:.2f}  audio={a1:.2f}/{a5:.2f}  "
          f"multimodal={m1:.2f}/{m5:.2f}  (top1/top5, n={n})")
print(f"\nWrote {OUT_DIR}/predictions.json and {n} clips to {CLIPS_DIR}")
