"""Noise-robustness sweep: multimodal vs visual-only vs audio-only across SNRs.

Evaluates all three systems on the same GLips validation clips under additive noise
at a set of SNRs, for two noise types:

  white   -- additive white Gaussian noise
  babble  -- a mix of other GLips waveforms, i.e. real German speech-shaped babble
             drawn from the same corpus

Visual-only is audio-independent, so it is computed once and reported as a flat
reference line.

Usage:  python snr_eval.py
Writes: snr_results.csv
"""
import os
import sys
import csv
import json
import importlib.util

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
# Register under a real module name before executing: DataLoader workers spawn fresh
# interpreters on Windows and unpickle the dataset, whose class identifies itself as
# living in 'mmtrain'. Without the sys.modules entry the child cannot resolve it.
if 'mmtrain' in sys.modules:
    m = sys.modules['mmtrain']
else:
    _spec = importlib.util.spec_from_file_location('mmtrain', os.path.join(HERE, 'train.py'))
    m = importlib.util.module_from_spec(_spec)
    sys.modules['mmtrain'] = m
    _spec.loader.exec_module(m)

from train_audio_probe import AudioProbe  # noqa: E402

ROOT_DIR = '../../lipreading/GLips_mouth/lipread_files'
AUDIO_ROOT = '../../lipreading/GLips/lipread_files'
CACHE_DIR = './cache/audio_cache'
FUSED_CKPT = './models/checkpoints_reg/best_model.pth'
PROBE_CKPT = './models/audio_probe/probe.pth'
SPLIT = os.environ.get('SPLIT', 'validation')
OUT_CSV = os.environ.get('OUT_CSV', './results/snr_results.csv')
EXCLUDED = ('hier', 'soll')
SNRS = [None, 15, 10, 5]        # None == clean
NOISES = ['white', 'babble']
BATCH = 16
SEED = 1234
# LIMIT=n caps batches per condition, for a fast end-to-end smoke test.
LIMIT = int(os.environ.get('LIMIT', '0')) or None


def add_noise(wav, noise, snr_db):
    """Scale `noise` to hit `snr_db` against `wav`, per clip, then mix.

    SNR = 10*log10(P_signal / P_noise), so the required noise gain is
    sqrt(P_signal / (P_noise * 10^(snr/10))).
    """
    if snr_db is None:
        return wav
    p_sig = wav.pow(2).mean(dim=1, keepdim=True)
    p_noi = noise.pow(2).mean(dim=1, keepdim=True).clamp_min(1e-12)
    gain = torch.sqrt(p_sig / (p_noi * (10.0 ** (snr_db / 10.0))))
    return wav + gain * noise


def make_noise(kind, wav, babble_pool, gen):
    """White: iid Gaussian. Babble: mean of several random corpus waveforms."""
    if kind == 'white':
        return torch.randn(wav.shape, generator=gen, device=wav.device, dtype=wav.dtype)
    n_speakers = 5
    idx = torch.randint(0, len(babble_pool), (wav.shape[0], n_speakers),
                        generator=gen, device=wav.device)
    return babble_pool[idx].mean(dim=1)


def fused_logits(model, video, audio, drop_audio):
    """Forward the cross-attention model, optionally masking the audio residual
    (the same path modality dropout used during training)."""
    x = model.cnn(video)
    B, T, C, H, W = x.size()
    x = model.avgpool(x.view(B * T, C, H, W)).flatten(1).view(B, T, m.FEAT_DIM)
    x = model.proj(x)
    x = model.ms_tcn(x)
    x = x + model.pos_embed[:, :T, :]
    x = model.transformer(x)
    mask = torch.zeros(B, 1, 1, device=x.device, dtype=x.dtype) if drop_audio else None
    x, _ = model.cross_attn(x, audio, mask)
    return model.classifier(model.head_drop(x.mean(dim=1)))


def main():
    device = torch.device('cuda')
    gen = torch.Generator(device=device).manual_seed(SEED)

    classes = sorted(d for d in os.listdir(ROOT_DIR)
                     if os.path.isdir(os.path.join(ROOT_DIR, d)) and d not in EXCLUDED)
    assert len(classes) == 498

    val_tf = m.VideoAugment(crop_size=88, resize_size=96, is_train=False)
    val_ds = m.MultimodalGLipsDataset(ROOT_DIR, split=SPLIT, num_frames=25,
                                      transform=val_tf, audio_root=AUDIO_ROOT,
                                      cache_dir=CACHE_DIR, group_split=False,
                                      classes=classes)
    print(f'split={SPLIT}')
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=4,
                            pin_memory=True, persistent_workers=True,
                            collate_fn=m.collate_fn)
    print(f'val clips: {len(val_ds)}')

    # Babble pool: a fixed sample of train waveforms, held on GPU for cheap mixing.
    with open(os.path.join(CACHE_DIR, 'index.json')) as f:
        index = json.load(f)
    wave_mm = np.memmap(os.path.join(CACHE_DIR, 'waveforms.dat'), dtype=np.float16,
                        mode='r', shape=(index['n'], index['audio_samples']))
    # cache keys use the on-disk split dir names: train / val / test
    train_rows = sorted(r for k, r in index['rows'].items() if k.split('/')[1] == 'train')
    assert train_rows, 'empty train split — check cache key format'
    rng = np.random.default_rng(SEED)
    pool_rows = rng.choice(train_rows, size=2000, replace=False)
    babble_pool = torch.from_numpy(
        np.asarray(wave_mm[np.sort(pool_rows)], dtype=np.float32)).to(device)
    print(f'babble pool: {tuple(babble_pool.shape)}')

    model = m.GLipsNet(num_classes=498, use_audio=True).to(device)
    raw = torch.load(FUSED_CKPT, map_location=device, weights_only=True)
    model.load_state_dict({k.replace('_orig_mod.', ''): v for k, v in raw.items()})
    model.eval()

    pck = torch.load(PROBE_CKPT, map_location=device, weights_only=False)
    probe = AudioProbe(num_classes=498).to(device)
    probe.load_state_dict(pck['state'])
    probe.eval()
    print(f'fused gate={model.cross_attn.gate.item():+.4f} | '
          f'probe clean val_top1={pck["val_top1"]:.4f}')

    extractor = m.WhisperExtractor(m.WHISPER_MODEL_NAME, device=device)
    amp = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    conditions = [('clean', None)] + [(nz, s) for nz in NOISES for s in SNRS if s is not None]
    results = []

    for cond_i, (noise_kind, snr) in enumerate(conditions):
        acc = {k: [0, 0] for k in ('multimodal', 'audio_only')}
        vis = [0, 0]
        total = 0
        label = 'clean' if snr is None else f'{noise_kind} {snr}dB'
        with torch.no_grad():
            for bi, (video, wav, target) in enumerate(
                    tqdm(val_loader, desc=f'[{cond_i+1}/{len(conditions)}] {label}', unit='batch')):
                if LIMIT and bi >= LIMIT:
                    break
                video = video.to(device, non_blocking=True)
                wav = wav.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

                if snr is not None:
                    noise = make_noise(noise_kind, wav, babble_pool, gen)
                    wav = add_noise(wav, noise, snr)

                audio = extractor.encode(wav)
                with torch.amp.autocast('cuda', dtype=amp):
                    lg = fused_logits(model, video, audio, drop_audio=False)
                    acc['multimodal'][0] += (lg.argmax(1) == target).sum().item()
                    acc['multimodal'][1] += (lg.topk(5, 1).indices == target.unsqueeze(1)).any(1).sum().item()

                    la = probe(audio.mean(dim=1).float())
                    acc['audio_only'][0] += (la.argmax(1) == target).sum().item()
                    acc['audio_only'][1] += (la.topk(5, 1).indices == target.unsqueeze(1)).any(1).sum().item()

                    # visual-only is audio-independent: compute once, on the clean pass
                    if cond_i == 0:
                        lv = fused_logits(model, video, audio, drop_audio=True)
                        vis[0] += (lv.argmax(1) == target).sum().item()
                        vis[1] += (lv.topk(5, 1).indices == target.unsqueeze(1)).any(1).sum().item()
                total += target.size(0)

        if cond_i == 0:
            visual_top1, visual_top5 = vis[0] / total, vis[1] / total
            results.append(['visual_only', 'n/a', 'n/a', f'{visual_top1:.4f}', f'{visual_top5:.4f}'])
            print(f'  visual_only        top1={visual_top1:.4f} top5={visual_top5:.4f}')

        for name, (c1, c5) in acc.items():
            results.append([name, noise_kind if snr is not None else 'none',
                            snr if snr is not None else 'clean',
                            f'{c1/total:.4f}', f'{c5/total:.4f}'])
            print(f'  {name:<18} top1={c1/total:.4f} top5={c5/total:.4f}')

        with open(OUT_CSV, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['system', 'noise', 'snr_db', 'top1', 'top5'])
            w.writerows(results)

    print(f'\nwrote {OUT_CSV}')


if __name__ == '__main__':
    main()
