"""
Audio-only baseline on the full 500-class GLips vocabulary.

Original usage (from multimodal/fusion/ in the full repo):
    python train_audio_probe_500.py
Writes: models/audio_probe_500/probe.pth  (+ cached train/val features)
"""
import os
import sys
import json
import importlib.util

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
if 'mmtrain' in sys.modules:
    m = sys.modules['mmtrain']
else:
    _spec = importlib.util.spec_from_file_location('mmtrain', os.path.join(HERE, 'audio_backbone.py'))
    m = importlib.util.module_from_spec(_spec)
    sys.modules['mmtrain'] = m
    _spec.loader.exec_module(m)

ROOT_DIR = './GLips_mouth/lipread_files'  # not shipped, see note above
CACHE_DIR = './audio_cache'  # not shipped, see note above
OUT_DIR = './audio_probe_500'
ENCODE_BATCH = 128
PROBE_EPOCHS = 30
PROBE_BATCH = 1024


def build_split_index(rows, classes, split):
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    items = []
    for key, row in rows.items():
        cls, sp, _ = key.split('/', 2)
        if sp == split and cls in cls_to_idx:
            items.append((row, cls_to_idx[cls]))
    items.sort()
    return np.array([r for r, _ in items], dtype=np.int64), \
           np.array([l for _, l in items], dtype=np.int64)


@torch.no_grad()
def encode_split(extractor, wave_mm, rows_idx, device, desc):
    feats = np.empty((len(rows_idx), m.AUDIO_DIM), dtype=np.float16)
    for i in tqdm(range(0, len(rows_idx), ENCODE_BATCH), desc=desc, unit='batch'):
        chunk = rows_idx[i:i + ENCODE_BATCH]
        wav = torch.from_numpy(np.asarray(wave_mm[chunk], dtype=np.float32)).to(device)
        enc = extractor.encode(wav)
        feats[i:i + len(chunk)] = enc.mean(dim=1).half().cpu().numpy()
    return feats


class AudioProbe(nn.Module):
    def __init__(self, in_dim=m.AUDIO_DIM, num_classes=500, hidden=1024, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device('cuda')
    assert torch.cuda.is_available()

    classes = sorted(d for d in os.listdir(ROOT_DIR) if os.path.isdir(os.path.join(ROOT_DIR, d)))
    assert len(classes) == 500, f'expected 500 classes, got {len(classes)}'

    with open(os.path.join(CACHE_DIR, 'index.json')) as f:
        index = json.load(f)
    wave_mm = np.memmap(os.path.join(CACHE_DIR, 'waveforms.dat'), dtype=np.float16,
                        mode='r', shape=(index['n'], index['audio_samples']))

    feat_path = os.path.join(OUT_DIR, 'features.npz')
    if os.path.exists(feat_path):
        print(f'reusing cached features: {feat_path}')
        d = np.load(feat_path)
        Xtr, ytr, Xva, yva = d['Xtr'], d['ytr'], d['Xva'], d['yva']
    else:
        extractor = m.WhisperExtractor(m.WHISPER_MODEL_NAME, device=device)
        tr_rows, ytr = build_split_index(index['rows'], classes, 'train')
        va_rows, yva = build_split_index(index['rows'], classes, 'val')
        print(f'train {len(tr_rows)} clips | val {len(va_rows)} clips')
        assert len(tr_rows) and len(va_rows), 'empty split — check cache key format'
        Xtr = encode_split(extractor, wave_mm, tr_rows, device, 'encode train')
        Xva = encode_split(extractor, wave_mm, va_rows, device, 'encode val')
        np.savez(feat_path, Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva)
        print(f'wrote {feat_path}')

    Xtr_t = torch.from_numpy(Xtr).float()
    ytr_t = torch.from_numpy(ytr).long()
    Xva_t = torch.from_numpy(Xva).float().to(device)
    yva_t = torch.from_numpy(yva).long().to(device)

    probe = AudioProbe(num_classes=len(classes)).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=PROBE_EPOCHS)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    best = 0.0
    n = len(Xtr_t)
    for ep in range(PROBE_EPOCHS):
        probe.train()
        perm = torch.randperm(n)
        tot = corr = 0
        for i in range(0, n, PROBE_BATCH):
            b = perm[i:i + PROBE_BATCH]
            xb, yb = Xtr_t[b].to(device, non_blocking=True), ytr_t[b].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = probe(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            corr += (logits.argmax(1) == yb).sum().item()
            tot += len(b)
        sched.step()

        probe.eval()
        with torch.no_grad():
            vl = probe(Xva_t)
            top1 = (vl.argmax(1) == yva_t).float().mean().item()
            top5 = (vl.topk(5, 1).indices == yva_t.unsqueeze(1)).any(1).float().mean().item()
        if top1 > best:
            best = top1
            torch.save({'state': probe.state_dict(), 'classes': classes,
                        'val_top1': top1, 'val_top5': top5},
                       os.path.join(OUT_DIR, 'probe.pth'))
        print(f'[{ep+1:>2}/{PROBE_EPOCHS}] train={corr/tot:.4f} '
              f'val_top1={top1:.4f} val_top5={top5:.4f}')

    print(f'\nBEST audio-only (500-class) val_top1={best:.4f}  ->  {OUT_DIR}/probe.pth')


if __name__ == '__main__':
    main()
