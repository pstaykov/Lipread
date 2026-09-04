"""Loads the three packaged models and runs live inference on a freshly recorded
clip: crops the mouth region with the same MediaPipe FaceMesh pipeline the GLips
training data went through (ported from lipreading/preprocess_mouth_roi.py), then
runs the video-only, audio-only and multimodal models on it.

Used by server.py; not meant to be run directly.
"""
import os
import sys
import json
import importlib.util

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import mediapipe as mp

HERE = os.path.dirname(os.path.abspath(__file__))
HANDOFF = os.path.dirname(HERE)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Same noise conditions as build_demo_data.py / the frontend's dropdown, so a
# self-recording is comparable to what's shown for the 16 fixed clips.
NOISE_CONDITIONS = {
    'clean': (None, None),
    'white_10db': ('white', 10),
    'white_5db': ('white', 5),
    'babble_10db': ('babble', 10),
    'babble_5db': ('babble', 5),
}
_babble_pool = None  # lazy-loaded, cached


def _load_babble_pool():
    global _babble_pool
    if _babble_pool is not None:
        return _babble_pool
    path = os.path.join(HERE, 'babble_pool.npz')
    if not os.path.exists(path):
        _babble_pool = False  # sentinel: tried, not available
        return None
    arr = np.load(path)['wav'].astype(np.float32)
    _babble_pool = torch.from_numpy(arr).to(device)
    return _babble_pool


def _add_noise(wav, noise, snr_db):
    """Same formula as multimodal/fusion/snr_eval.py and build_demo_data.py."""
    p_sig = wav.pow(2).mean(dim=1, keepdim=True)
    p_noi = noise.pow(2).mean(dim=1, keepdim=True).clamp_min(1e-12)
    gain = torch.sqrt(p_sig / (p_noi * (10.0 ** (snr_db / 10.0))))
    return wav + gain * noise


def make_noisy_wav(wav_t, condition):
    """wav_t: (1, AUDIO_SAMPLES) tensor, clean. Returns a (possibly) noised copy."""
    kind, snr = NOISE_CONDITIONS.get(condition, (None, None))
    if kind is None:
        return wav_t
    if kind == 'babble':
        pool = _load_babble_pool()
    else:
        pool = None
    if kind == 'white' or pool is None:
        noise = torch.randn_like(wav_t)
    else:
        n_speakers = 5
        idx = torch.randint(0, len(pool), (wav_t.shape[0], n_speakers), device=wav_t.device)
        noise = pool[idx].mean(dim=1)
    return _add_noise(wav_t, noise, snr)

# mouth-ROI crop, ported from lipreading/preprocess_mouth_roi.py so a live
# recording goes through the exact same crop the training data used.
LIP_IDX = sorted({i for pair in mp.solutions.face_mesh.FACEMESH_LIPS for i in pair})
OUT_SIZE = 96
BOX_SCALE = 1.6
ALPHA = 0.4

_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1,
    min_detection_confidence=0.5, min_tracking_confidence=0.5,
)


def _lip_box(landmarks, w, h):
    xs = [landmarks[i].x * w for i in LIP_IDX]
    ys = [landmarks[i].y * h for i in LIP_IDX]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = max(max(x1 - x0, y1 - y0) * BOX_SCALE, 24.0)
    return cx, cy, side


def _smooth(prev, cur):
    if prev is None:
        return cur
    return tuple(ALPHA * c + (1 - ALPHA) * p for p, c in zip(prev, cur))


def _crop_square(frame, box):
    h, w = frame.shape[:2]
    cx, cy, side = box
    side = max(1, int(round(side)))
    left, top = int(round(cx - side / 2)), int(round(cy - side / 2))
    right, bottom = left + side, top + side
    pl, pt = max(0, -left), max(0, -top)
    pr, pb = max(0, right - w), max(0, bottom - h)
    bl, bt = max(0, left), max(0, top)
    br, bb = min(w, right), min(h, bottom)
    crop = frame[bt:bb, bl:br]
    if crop.size == 0:
        crop = frame
    if pl or pt or pr or pb:
        crop = cv2.copyMakeBorder(crop, pt, pb, pl, pr, cv2.BORDER_REPLICATE)
    return cv2.resize(crop, (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_CUBIC)


def crop_mouth_frames(frames_bgr):
    """frames_bgr: list of HxWx3 BGR uint8 frames -> list of OUT_SIZExOUT_SIZEx3
    BGR uint8 mouth crops, and the detection rate (fraction of frames with a
    detected face) so the caller can warn if the crop is unreliable."""
    out, last_box, n_detect = [], None, 0
    for frame in frames_bgr:
        h, w = frame.shape[:2]
        res = _face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if res.multi_face_landmarks:
            n_detect += 1
            last_box = _smooth(last_box, _lip_box(res.multi_face_landmarks[0].landmark, w, h))
        elif last_box is None:
            last_box = (w / 2, h / 2, min(w, h) * 0.4)
        out.append(_crop_square(frame, last_box))
    detect_rate = n_detect / len(frames_bgr) if frames_bgr else 0.0
    return out, detect_rate


# load the three packaged models exactly as build_demo_data.py does.
def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


video_dir = os.path.join(HANDOFF, 'models', 'video_only')
audio_dir = os.path.join(HANDOFF, 'models', 'audio_only')
mm_dir = os.path.join(HANDOFF, 'models', 'multimodal')

vmod = _load_module('live_video_model', os.path.join(video_dir, 'model.py'))
vdata = _load_module('live_video_dataset', os.path.join(video_dir, 'dataset.py'))
amod = _load_module('live_audio_train', os.path.join(audio_dir, 'train.py'))
mmod = _load_module('live_mm_model', os.path.join(mm_dir, 'model.py'))

with open(os.path.join(video_dir, 'classes.json'), encoding='utf-8') as f:
    CLASSES = json.load(f)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)

video_model = vmod.GLipsNet(num_classes=NUM_CLASSES, pool='attn').to(device).eval()
video_model.load_state_dict(vmod._strip_orig_mod(
    torch.load(os.path.join(video_dir, 'best_model.pth'), map_location=device)))

audio_ckpt = torch.load(os.path.join(audio_dir, 'probe.pth'), map_location=device)
audio_probe = amod.AudioProbe(num_classes=NUM_CLASSES).to(device).eval()
audio_probe.load_state_dict(audio_ckpt['state'])

extractor = mmod.WhisperExtractor(mmod.WHISPER_MODEL_NAME, device=device)
mm_model = mmod.GLipsNet(num_classes=NUM_CLASSES, use_audio=True).to(device).eval()
mm_model.load_state_dict(mmod._strip_orig_mod(
    torch.load(os.path.join(mm_dir, 'best_model.pth'), map_location=device)))

val_transform = vdata.VideoAugment(crop_size=88, resize_size=96, is_train=False)
print(f"live_infer: {NUM_CLASSES} classes, models loaded on {device}")


def frames_to_tensor(mouth_crops_bgr):
    """List of 96x96x3 BGR uint8 crops -> the exact tensor GLipsFullClipDataset
    would produce: resample to 25 frames, val-transform, (C,T,H,W)."""
    rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in mouth_crops_bgr]
    video = torch.from_numpy(np.stack(rgb)).float() / 255.0   # (T,H,W,C)
    video = video.permute(0, 3, 1, 2)                          # (T,C,H,W)
    T = video.size(0)
    n = 25
    if T > n:
        idx = np.linspace(0, T - 1, num=n).astype(int)
        video = video[idx]
    elif T < n:
        pad = n - T
        video = torch.cat([video, video[-1:].repeat(pad, 1, 1, 1)], dim=0)
    video = val_transform(video)
    return video.permute(1, 0, 2, 3)  # (C,T,H,W)


def topk(probs, k=5):
    vals, idx = probs.topk(k)
    return [{'word': CLASSES[i], 'prob': round(float(v), 4)} for v, i in zip(vals.tolist(), idx.tolist())]


def _entry(probs, target):
    top5 = topk(probs, 5)
    out = {'top5': top5, 'top1': top5[0]['word']}
    if target is not None and target in CLASS_TO_IDX:
        out['correct'] = top5[0]['word'] == target
        out['top5_correct'] = any(item['word'] == target for item in top5)
        out['true_word_prob'] = round(float(probs[CLASS_TO_IDX[target]]), 4)
    return out


@torch.no_grad()
def predict(mouth_crops_bgr, wav, target=None, condition='clean'):
    """mouth_crops_bgr: list of 96x96x3 BGR uint8 frames (already cropped).
    wav: 1-D float32 numpy array, 16 kHz. target: optional word from CLASSES.
    condition: one of NOISE_CONDITIONS' keys -- mixed into the audio before both
    the audio-only and multimodal models see it (video-only is unaffected).
    Returns (predictions dict, noisy_wav numpy array clipped to [-1,1]) so the
    caller can mux exactly what the models heard back into a preview clip."""
    video_t = frames_to_tensor(mouth_crops_bgr).unsqueeze(0).to(device)
    wav_t = mmod._fix_wav(wav).unsqueeze(0).to(device)
    wav_t = make_noisy_wav(wav_t, condition)

    v_probs = F.softmax(video_model(video_t), dim=1)[0].cpu()

    audio_tokens = extractor.encode(wav_t)
    a_probs = F.softmax(audio_probe(audio_tokens.mean(dim=1)), dim=1)[0].cpu()

    mm_probs = F.softmax(mm_model(video_t, audio_tokens), dim=1)[0].cpu()

    preds = {
        'video_only': _entry(v_probs, target),
        'audio_only': _entry(a_probs, target),
        'multimodal': _entry(mm_probs, target),
    }
    noisy_wav = wav_t[0].clamp(-1.0, 1.0).cpu().numpy()
    return preds, noisy_wav
