"""Comparison table across trained fusion heads + audio-only probe, on cached val features."""
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

from fusion_heads import HEADS                     # noqa: E402
from train_audio_probe import AudioProbe           # noqa: E402

VIS_CACHE = os.path.join(HERE, 'cache', 'visual_token_cache_500')
AUD_CACHE = os.path.join(HERE, 'cache', 'audio_cache')
RUNS = os.path.join(HERE, 'models', 'fusion_runs_500')
PROBE_CKPT = os.path.join(HERE, 'models', 'audio_probe_500', 'probe.pth')
OUT_CSV = os.path.join(HERE, 'results', 'fusion_comparison_500.csv')
BATCH = 160


def load_val():
    with open(os.path.join(VIS_CACHE, 'val_index.json')) as f:
        vidx = json.load(f)
    n, Tv, Dv = vidx['n'], vidx['Tv'], vidx['Dv']
    vtok = np.memmap(os.path.join(VIS_CACHE, 'val_tokens.dat'), dtype=np.float16,
                     mode='r', shape=(n, Tv, Dv))
    labels = np.load(os.path.join(VIS_CACHE, 'val_labels.npy'))
    with open(os.path.join(AUD_CACHE, 'index.json')) as f:
        aidx = json.load(f)
    wav = np.memmap(os.path.join(AUD_CACHE, 'waveforms.dat'), dtype=np.float16,
                    mode='r', shape=(aidx['n'], aidx['audio_samples']))
    inv = {r: k for k, r in vidx['rows'].items()}
    arows = np.array([aidx['rows'].get(inv[r], -1) for r in range(n)], dtype=np.int64)
    return (torch.from_numpy(np.asarray(vtok)).float(), torch.from_numpy(labels).long(),
            wav, arows, vidx['classes'])


def main():
    device = torch.device('cuda')
    amp = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    Vva, yva, wav, arows, classes = load_val()
    n = len(yva)
    y = yva.to(device)
    extractor = m.WhisperExtractor(m.WHISPER_MODEL_NAME, device=device)

    def enc_audio(rows):
        w = np.zeros((len(rows), wav.shape[1]), dtype=np.float32)
        v = rows >= 0
        if v.any():
            w[v] = np.asarray(wav[rows[v]], dtype=np.float32)
        return extractor.encode(torch.from_numpy(w).to(device))

    heads = {}
    for name in HEADS:
        ck = os.path.join(RUNS, name, 'best.pth')
        if os.path.exists(ck):
            d = torch.load(ck, map_location=device, weights_only=False)
            h = HEADS[name](num_classes=len(classes)).to(device)
            h.load_state_dict(d['state'])
            h.eval()
            heads[name] = h
    probe = None
    if os.path.exists(PROBE_CKPT):
        d = torch.load(PROBE_CKPT, map_location=device, weights_only=False)
        probe = AudioProbe(num_classes=len(classes)).to(device)
        probe.load_state_dict(d['state'])
        probe.eval()

    counts = {k: [0, 0] for k in list(heads) + (['audio_only'] if probe else [])}
    with torch.no_grad():
        for i in tqdm(range(0, n, BATCH), desc='eval', unit='batch'):
            v = Vva[i:i + BATCH].to(device)
            a = enc_audio(arows[i:i + BATCH])
            yb = y[i:i + v.size(0)]
            with torch.amp.autocast('cuda', dtype=amp):
                for name, h in heads.items():
                    lg = h(v, a)
                    counts[name][0] += (lg.argmax(1) == yb).sum().item()
                    counts[name][1] += (lg.topk(5, 1).indices == yb.unsqueeze(1)).any(1).sum().item()
                if probe is not None:
                    lg = probe(a.mean(1).float())
                    counts['audio_only'][0] += (lg.argmax(1) == yb).sum().item()
                    counts['audio_only'][1] += (lg.topk(5, 1).indices == yb.unsqueeze(1)).any(1).sum().item()

    rows_out = []
    print(f'\n{"system":16s}{"top1":>9}{"top5":>9}   (n={n}, {len(classes)} classes)')
    for name in list(heads) + (['audio_only'] if probe else []):
        c1, c5 = counts[name]
        rows_out.append([name, f'{c1/n:.4f}', f'{c5/n:.4f}'])
        print(f'{name:16s}{c1/n:>9.4f}{c5/n:>9.4f}')
    with open(OUT_CSV, 'w', newline='') as f:
        wtr = csv.writer(f)
        wtr.writerow(['system', 'val_top1', 'val_top5'])
        wtr.writerows(rows_out)
    print(f'\nwrote {OUT_CSV}')


if __name__ == '__main__':
    main()
