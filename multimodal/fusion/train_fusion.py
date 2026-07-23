"""Train one fusion head on cached visual tokens + live Whisper audio.

Because the visual trunk is frozen and its tokens are cached, and audio is encoded
from the cached waveform (no video decode), an epoch is ~3-4 min. Every variant
trains under identical conditions, so the resulting numbers compare fusion
mechanisms directly.

    python train_fusion.py <variant>      # variant in fusion_heads.HEADS

Writes fusion_runs/<variant>/{best.pth, metrics.csv}.
"""
import os
import sys
import csv
import json
import importlib.util
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
if 'mmtrain' not in sys.modules:
    spec = importlib.util.spec_from_file_location('mmtrain', os.path.join(HERE, 'train.py'))
    m = importlib.util.module_from_spec(spec)
    sys.modules['mmtrain'] = m
    spec.loader.exec_module(m)
else:
    m = sys.modules['mmtrain']

from fusion_heads import HEADS   # noqa: E402

VIS_CACHE = os.environ.get('VIS_CACHE', os.path.join(HERE, 'cache', 'visual_token_cache'))
AUD_CACHE = os.path.join(HERE, 'cache', 'audio_cache')
OUT_ROOT = os.environ.get('OUT_ROOT', os.path.join(HERE, 'models', 'fusion_runs'))
EPOCHS = int(os.environ.get('EPOCHS', '20'))
PATIENCE = int(os.environ.get('PATIENCE', '4'))
BATCH = int(os.environ.get('BATCH', '160'))
LR = float(os.environ.get('LR', '3e-4'))


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
    # map each visual row -> audio row via the shared clip key
    arows = np.empty(n, dtype=np.int64)
    miss = 0
    inv = {r: k for k, r in vidx['rows'].items()}
    for vr in range(n):
        key = inv[vr]
        ar = aidx['rows'].get(key)
        if ar is None:
            arows[vr] = -1
            miss += 1
        else:
            arows[vr] = ar
    if miss:
        print(f'  {split}: {miss}/{n} clips have no cached waveform (will use silence)')
    return vtok, labels, wav, arows, vidx['classes']


def main():
    variant = sys.argv[1] if len(sys.argv) > 1 else 'cross_attn'
    assert variant in HEADS, f'unknown variant {variant}; choices {list(HEADS)}'
    out_dir = os.path.join(OUT_ROOT, variant)
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device('cuda')
    amp = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    print(f'=== fusion variant: {variant} ===')
    vtok_tr, y_tr, wav, arow_tr, classes = load_split('train')
    vtok_va, y_va, _,   arow_va, _       = load_split('val')
    n_tr, n_va = len(y_tr), len(y_va)
    print(f'train {n_tr} | val {n_va} | classes {len(classes)}')

    # hold visual tokens in RAM as fp16 (~2.5GB train); cast per batch. waveforms
    # stay on disk (memmap) and are encoded live.
    Vtr = torch.from_numpy(np.asarray(vtok_tr))   # fp16
    Vva = torch.from_numpy(np.asarray(vtok_va))
    ytr = torch.from_numpy(y_tr).long()
    yva = torch.from_numpy(y_va).long()

    extractor = m.WhisperExtractor(m.WHISPER_MODEL_NAME, device=device)
    model = HEADS[variant](num_classes=len(classes)).to(device)
    print(f'params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    def encode_audio(rows):
        w = np.zeros((len(rows), wav.shape[1]), dtype=np.float32)
        valid = rows >= 0
        if valid.any():
            w[valid] = np.asarray(wav[rows[valid]], dtype=np.float32)
        wt = torch.from_numpy(w).to(device, non_blocking=True)
        return extractor.encode(wt)

    metrics_path = os.path.join(out_dir, 'metrics.csv')
    with open(metrics_path, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'train_acc', 'val_top1', 'val_top5', 'lr'])

    best = 0.0
    since = 0
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n_tr)
        corr = tot = 0
        for i in tqdm(range(0, n_tr, BATCH), desc=f'{variant} ep{ep+1}/{EPOCHS}', unit='batch'):
            b = perm[i:i + BATCH].numpy()
            v = Vtr[b].to(device, non_blocking=True).float()
            a = encode_audio(arow_tr[b])
            yb = ytr[b].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', dtype=amp):
                logits = model(v, a)
                loss = crit(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            corr += (logits.argmax(1) == yb).sum().item()
            tot += len(b)
        sched.step()

        model.eval()
        c1 = c5 = 0
        with torch.no_grad():
            for i in range(0, n_va, BATCH):
                v = Vva[i:i + BATCH].to(device, non_blocking=True).float()
                a = encode_audio(arow_va[i:i + BATCH])
                yb = yva[i:i + BATCH].to(device)
                with torch.amp.autocast('cuda', dtype=amp):
                    logits = model(v, a)
                c1 += (logits.argmax(1) == yb).sum().item()
                c5 += (logits.topk(5, 1).indices == yb.unsqueeze(1)).any(1).sum().item()
        top1, top5 = c1 / n_va, c5 / n_va
        lr = sched.get_last_lr()[0]
        with open(metrics_path, 'a', newline='') as f:
            csv.writer(f).writerow([ep + 1, f'{corr/tot:.4f}', f'{top1:.4f}', f'{top5:.4f}', f'{lr:.6f}'])
        print(f'[{variant}] ep{ep+1}: train={corr/tot:.4f} val_top1={top1:.4f} val_top5={top5:.4f}')

        if top1 > best:
            best, since = top1, 0
            torch.save({'state': model.state_dict(), 'variant': variant,
                        'val_top1': top1, 'val_top5': top5, 'classes': classes},
                       os.path.join(out_dir, 'best.pth'))
        else:
            since += 1
        if since >= PATIENCE:
            print(f'[{variant}] early stop (no val gain for {since} epochs)')
            break

    print(f'[{variant}] BEST val_top1={best:.4f}  -> {out_dir}/best.pth')


if __name__ == '__main__':
    main()
