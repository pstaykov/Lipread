"""Assemble one master comparison table: every architecture on the same val set."""
import os
import sys
import csv
import json
import importlib.util
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
LIPREAD = os.path.abspath(os.path.join(HERE, '..', '..', 'lipreading'))
sys.path.insert(0, os.path.join(LIPREAD, 'Transformer_based'))
from model import GLipsNet, _strip_orig_mod   # noqa: E402

VIS_CACHE = os.path.join(HERE, 'cache', 'visual_token_cache_500')
CKPT = os.path.join(LIPREAD, 'Transformer_based', 'checkpoints_500', 'best_model.pth')
REG_METRICS = os.path.join(HERE, 'models', 'checkpoints_500_full', 'metrics.csv')
FUSION_CSV = os.path.join(HERE, 'results', 'fusion_comparison_500.csv')
OUT = os.path.join(HERE, 'results', 'master_comparison_500.csv')


def visual_only_on_cache():
    """Run the standalone visual model's attn-pool + classifier over cached val tokens."""
    device = torch.device('cuda')
    with open(os.path.join(VIS_CACHE, 'val_index.json')) as f:
        vidx = json.load(f)
    n, Tv, Dv = vidx['n'], vidx['Tv'], vidx['Dv']
    vtok = np.memmap(os.path.join(VIS_CACHE, 'val_tokens.dat'), dtype=np.float16,
                     mode='r', shape=(n, Tv, Dv))
    y = torch.from_numpy(np.load(os.path.join(VIS_CACHE, 'val_labels.npy'))).long().to(device)

    model = GLipsNet(num_classes=len(vidx['classes']), pool='attn').to(device)
    model.load_state_dict(_strip_orig_mod(torch.load(CKPT, map_location=device, weights_only=True)))
    model.eval()

    c1 = c5 = 0
    with torch.no_grad():
        for i in range(0, n, 512):
            x = torch.from_numpy(np.asarray(vtok[i:i + 512])).float().to(device)
            pooled = model.attn_pool(x) if model.attn_pool is not None else x.mean(1)
            logits = model.classifier(model.head_drop(pooled))
            yb = y[i:i + x.size(0)]
            c1 += (logits.argmax(1) == yb).sum().item()
            c5 += (logits.topk(5, 1).indices == yb.unsqueeze(1)).any(1).sum().item()
    return n, c1 / n, c5 / n


def shipped_val():
    """Best val row of the shipped fine-tuned cross_attn run."""
    best = None
    with open(REG_METRICS) as f:
        for r in csv.DictReader(f):
            t1 = float(r['val_top1'])
            if best is None or t1 > best[0]:
                best = (t1, float(r['val_top5']))
    return best


def main():
    n, v1, v5 = visual_only_on_cache()
    print(f'visual_only (standalone, {n} clips): top1={v1:.4f} top5={v5:.4f}')

    rows = {}
    with open(FUSION_CSV) as f:
        for r in csv.DictReader(f):
            rows[r['system']] = (float(r['val_top1']), float(r['val_top5']))

    s1, s5 = shipped_val()

    table = [
        ('visual_only (standalone)',        v1, v5, 'frozen',     '—'),
        ('audio_only (Whisper probe)',      *rows['audio_only'], 'frozen', '—'),
        ('late fusion',                     *rows['late'],       'frozen backbone', '0.38M'),
        ('concat fusion',                   *rows['concat'],     'frozen backbone', '0.98M'),
        ('cross_attn (frozen)',             *rows['cross_attn'], 'frozen backbone', '0.52M'),
        ('cross_attn (shipped, finetuned)', s1, s5,              'backbone finetuned', '—'),
        ('joint_tf / AV-HuBERT-style',      *rows['joint_tf'],   'frozen backbone', '2.70M'),
    ]
    table.sort(key=lambda r: r[1], reverse=True)

    with open(OUT, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['system', 'val_top1', 'val_top5', 'regime', 'fusion_params'])
        for name, t1, t5, reg, p in table:
            w.writerow([name, f'{t1:.4f}', f'{t5:.4f}', reg, p])

    print(f'\nmaster_comparison_500.csv (val, {n} clips, 500-class):')
    print(f'{"system":34s}{"top1":>8}{"top5":>8}   regime')
    for name, t1, t5, reg, p in table:
        print(f'{name:34s}{t1:>8.4f}{t5:>8.4f}   {reg}')
    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    main()
