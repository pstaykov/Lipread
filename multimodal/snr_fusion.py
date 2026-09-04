"""Noise-robustness sweep (like snr_eval.py) for a trained fusion head on cached visual tokens.

    python snr_fusion.py [variant]      SPLIT=test (default)

Needs the visual token cache for SPLIT (SPLITS=<split> python cache_visual_tokens.py).
Writes snr_fusion_<variant>_<split>.csv.
"""
import os
import sys
import csv
import json
import importlib.util
import numpy as np
import torch
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
if 'mmtrain' not in sys.modules:
    spec = importlib.util.spec_from_file_location('mmtrain', os.path.join(HERE, 'train.py'))
    m = importlib.util.module_from_spec(spec)
    sys.modules['mmtrain'] = m
    spec.loader.exec_module(m)
else:
    m = sys.modules['mmtrain']

from fusion_heads import HEADS               # noqa: E402
from train_audio_probe import AudioProbe     # noqa: E402

VIS_CACHE = os.environ.get('VIS_CACHE', os.path.join(HERE, 'cache', 'visual_token_cache_500'))
AUD_CACHE = os.path.join(HERE, 'cache', 'audio_cache')
RUNS = os.environ.get('RUNS', os.path.join(HERE, 'models', 'fusion_runs_500'))
PROBE_CKPT = os.environ.get('PROBE_CKPT', os.path.join(HERE, 'models', 'audio_probe_500', 'probe.pth'))
SPLIT = os.environ.get('SPLIT', 'test')
SNRS = [None, 15, 10, 5]
NOISES = ['white', 'babble']
BATCH = 160
SEED = 1234


def add_noise(wav, noise, snr_db):
    if snr_db is None:
        return wav
    p_sig = wav.pow(2).mean(1, keepdim=True)
    p_noi = noise.pow(2).mean(1, keepdim=True).clamp_min(1e-12)
    return wav + torch.sqrt(p_sig / (p_noi * 10.0 ** (snr_db / 10.0))) * noise


def make_noise(kind, wav, babble, gen):
    if kind == 'white':
        return torch.randn(wav.shape, generator=gen, device=wav.device, dtype=wav.dtype)
    idx = torch.randint(0, len(babble), (wav.shape[0], 5), generator=gen, device=wav.device)
    return babble[idx].mean(1)


def load_split(split):
    with open(os.path.join(VIS_CACHE, f'{split}_index.json')) as f:
        vidx = json.load(f)
    n, Tv, Dv = vidx['n'], vidx['Tv'], vidx['Dv']
    vtok = np.memmap(os.path.join(VIS_CACHE, f'{split}_tokens.dat'), dtype=np.float16,
                     mode='r', shape=(n, Tv, Dv))
    labels = np.load(os.path.join(VIS_CACHE, f'{split}_labels.npy'))
    with open(os.path.join(AUD_CACHE, 'index.json')) as f:
        aidx = json.load(f)
    wav = np.memmap(os.path.join(AUD_CACHE, 'waveforms.dat'), dtype=np.float16,
                    mode='r', shape=(aidx['n'], aidx['audio_samples']))
    inv = {r: k for k, r in vidx['rows'].items()}
    arows = np.array([aidx['rows'].get(inv[r], -1) for r in range(n)], dtype=np.int64)
    train_rows = sorted(r for k, r in aidx['rows'].items() if k.split('/')[1] == 'train')
    return (torch.from_numpy(np.asarray(vtok)), torch.from_numpy(labels).long(),
            wav, arows, np.array(train_rows), vidx['classes'])


def main():
    variant = sys.argv[1] if len(sys.argv) > 1 else 'joint_tf'
    assert variant in HEADS, f'unknown variant {variant}'
    device = torch.device('cuda')
    amp = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    gen = torch.Generator(device=device).manual_seed(SEED)

    Vt, y, wav, arows, train_rows, classes = load_split(SPLIT)
    n = int(os.environ.get('LIMIT', '0')) or len(y)   # LIMIT>0 caps n for a smoke test
    n = min(n, len(y))
    print(f'variant={variant} split={SPLIT} n={n} classes={len(classes)}')

    rng = np.random.default_rng(SEED)
    pool_rows = np.sort(rng.choice(train_rows, size=2000, replace=False))
    babble = torch.from_numpy(np.asarray(wav[pool_rows], dtype=np.float32)).to(device)

    d = torch.load(os.path.join(RUNS, variant, 'best.pth'), map_location=device, weights_only=False)
    head = HEADS[variant](num_classes=len(classes)).to(device)
    head.load_state_dict(d['state']); head.eval()
    probe = None
    if os.path.exists(PROBE_CKPT):
        pd = torch.load(PROBE_CKPT, map_location=device, weights_only=False)
        probe = AudioProbe(num_classes=len(classes)).to(device)
        probe.load_state_dict(pd['state']); probe.eval()

    extractor = m.WhisperExtractor(m.WHISPER_MODEL_NAME, device=device)

    def enc(rows, snr, kind):
        w = np.zeros((len(rows), wav.shape[1]), dtype=np.float32)
        v = rows >= 0
        if v.any():
            w[v] = np.asarray(wav[rows[v]], dtype=np.float32)
        wt = torch.from_numpy(w).to(device)
        if snr is not None:
            wt = add_noise(wt, make_noise(kind, wt, babble, gen), snr)
        return extractor.encode(wt)

    conditions = [('clean', None)] + [(k, s) for k in NOISES for s in SNRS if s is not None]
    out = []
    out_csv = os.path.join(HERE, 'results', f'snr_fusion_{variant}_{SPLIT}_500.csv')

    for ci, (kind, snr) in enumerate(conditions):
        acc = {variant: [0, 0], 'audio_only': [0, 0]}
        vis = [0, 0]
        label = 'clean' if snr is None else f'{kind} {snr}dB'
        with torch.no_grad():
            for i in tqdm(range(0, n, BATCH), desc=f'[{ci+1}/{len(conditions)}] {label}', unit='batch'):
                v = Vt[i:i + BATCH].to(device).float()
                yb = y[i:i + v.size(0)].to(device)
                a = enc(arows[i:i + BATCH], snr, kind)
                with torch.amp.autocast('cuda', dtype=amp):
                    lg = head(v, a)
                    acc[variant][0] += (lg.argmax(1) == yb).sum().item()
                    acc[variant][1] += (lg.topk(5, 1).indices == yb.unsqueeze(1)).any(1).sum().item()
                    if probe is not None:
                        la = probe(a.mean(1).float())
                        acc['audio_only'][0] += (la.argmax(1) == yb).sum().item()
                        acc['audio_only'][1] += (la.topk(5, 1).indices == yb.unsqueeze(1)).any(1).sum().item()
                    if ci == 0:
                        lv = head(v, a, drop_audio=True)
                        vis[0] += (lv.argmax(1) == yb).sum().item()
                        vis[1] += (lv.topk(5, 1).indices == yb.unsqueeze(1)).any(1).sum().item()
        if ci == 0:
            out.append(['visual_only', 'n/a', 'n/a', f'{vis[0]/n:.4f}', f'{vis[1]/n:.4f}'])
            print(f'  visual_only  top1={vis[0]/n:.4f} top5={vis[1]/n:.4f}')
        for name, (c1, c5) in acc.items():
            out.append([name, kind if snr is not None else 'none', snr if snr is not None else 'clean',
                        f'{c1/n:.4f}', f'{c5/n:.4f}'])
            print(f'  {name:12s} top1={c1/n:.4f} top5={c5/n:.4f}')
        with open(out_csv, 'w', newline='') as f:
            w = csv.writer(f); w.writerow(['system', 'noise', 'snr_db', 'top1', 'top5']); w.writerows(out)

    print(f'\nwrote {out_csv}')


if __name__ == '__main__':
    main()
