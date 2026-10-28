"""Train one input variant (docs/MODELS.md) on BTLips, pick the epoch with the best val top-1, test it.

    python -m model.train --variant V4 --epochs 30

Same recipe for every variant: AdamW (lr 3e-4 at 32 clips per step, scaled by sqrt(batch / 32); wd 1e-2), cosine
schedule with 1 warm-up epoch, label smoothing 0.1, mixed precision. --batch 128 for the coordinate variants:
their step time is launch-bound, so 128 clips cost the same time as 32. Image variants (V0-V3) are flipped horizontally at random.
Coordinate features (aligned + pose-corrected contour points) are computed once into runs/cache/.
Output runs/<variant>/: train.log, metrics.csv, best.pt, test.json (top-1, top-5, macro-F1).
"""
import argparse
import csv
import json
import math
import time
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

from .data import clips
from .data.landmarks import CONTOUR_IDX, LIP_IDX, FaceNorm
from .data.variants import VARIANTS, make_inputs, uses_video
from .nets import LipreadingModel

RUNS = clips.REPO / 'runs'
CACHE = RUNS / 'cache'
BATCH = 32
NORM_CLIPS = 4000


def facenorm():
    path = CACHE / 'facenorm.npz'
    if path.exists():
        return FaceNorm.load(path)
    idx = clips.read_split('train')
    fm = clips.facemesh('train')
    pick = np.sort(np.random.default_rng(0).choice(len(idx), NORM_CLIPS, replace=False))
    norm = FaceNorm.fit([clips.to_pixels(fm[i]) for i in pick], [idx[i]['speaker'] for i in pick])
    CACHE.mkdir(parents=True, exist_ok=True)
    norm.save(path)
    return norm


def _contour(args):
    split, lo, hi = args
    norm, fm = FaceNorm.load(CACHE / 'facenorm.npz'), clips.facemesh(split)
    return np.stack([norm(clips.to_pixels(fm[i]))[0][:, CONTOUR_IDX] for i in range(lo, hi)]).astype(np.float16)


def contour_cache(split):
    """(N, 29, 128, 3) float16 aligned, pose-corrected contour points (V4; V5 = its lip subset)."""
    path = CACHE / f'contour_{split}.npy'
    if not path.exists():
        n = len(clips.read_split(split))
        with Pool(10) as p:
            parts = p.map(_contour, [(split, lo, min(lo + 2000, n)) for lo in range(0, n, 2000)])
        np.save(path, np.concatenate(parts))
    return np.load(path)


class VideoSet(torch.utils.data.Dataset):
    def __init__(self, variant, split, labels, norm, train):
        self.variant, self.split, self.labels, self.norm, self.train = variant, split, labels, norm, train
        self.items = clips.read_split(split)
        self.fm = None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        if self.fm is None:                                   # one memory map per worker
            self.fm = clips.facemesh(self.split)
        seq = clips.to_pixels(self.fm[i])
        x = make_inputs(self.variant, seq, self.norm, clips.load_frames(self.items[i]['name']))
        if self.train and 'coords' not in x and torch.rand(()) < 0.5:
            x['video'] = x['video'].flip(-1)
        return {k: v.half() for k, v in x.items()}, self.labels[i]


def coord_batches(X, y, train, batch):
    order = torch.randperm(len(y), device=X.device) if train else torch.arange(len(y), device=X.device)
    step = batch if train else 256
    for b in range(0, len(y), step):
        j = order[b:b + step]
        yield {'coords': X[j].float()}, y[j]


def evaluate(model, batches, dev):
    model.eval()
    logits, ys = [], []
    with torch.no_grad(), torch.autocast('cuda', torch.float16):
        for x, y in batches:
            logits.append(model(**{k: v.to(dev, non_blocking=True).float() for k, v in x.items()}).float().cpu())
            ys.append(y.cpu())
    logits, ys = torch.cat(logits), torch.cat(ys)
    top5 = logits.topk(5, 1).indices
    pred = top5[:, 0]
    return {'top1': (pred == ys).float().mean().item(), 'top5': (top5 == ys[:, None]).any(1).float().mean().item(),
            'f1': f1_score(ys, pred, average='macro'), 'loss': F.cross_entropy(logits, ys).item()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', required=True, choices=VARIANTS)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch', type=int, default=BATCH, help='clips per optimizer step')
    ap.add_argument('--micro', type=int, default=16, help='clips per forward pass (image variants)')
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    out = RUNS / args.variant
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / 'train.log', 'a', encoding='utf-8')

    def say(s):
        print(s, flush=True)
        log.write(s + '\n')
        log.flush()

    dev = torch.device('cuda')
    word_ix = {w: i for i, w in enumerate(clips.words())}
    labels = {s: torch.tensor([word_ix[c['word']] for c in clips.read_split(s)]) for s in clips.SPLITS}
    norm = facenorm()
    video = uses_video(args.variant)
    if video:
        def loader(s, train):
            ds = VideoSet(args.variant, s, labels[s], norm, train)
            return torch.utils.data.DataLoader(ds, args.micro if train else 32, shuffle=train, drop_last=train,
                                               num_workers=args.workers if train else 6, pin_memory=True,
                                               persistent_workers=train, prefetch_factor=4)
        loaders = {s: loader(s, s == 'train') for s in clips.SPLITS}
        batches = lambda s, train: iter(loaders[s])
        steps = len(loaders['train']) * args.micro // args.batch
    else:
        sub = np.searchsorted(CONTOUR_IDX, VARIANTS[args.variant][1])        # V5: lip points within the contour
        X = {s: torch.from_numpy(contour_cache(s)[:, :, sub].reshape(len(labels[s]), 29, -1)).to(dev)
             for s in clips.SPLITS}
        Y = {s: labels[s].to(dev) for s in clips.SPLITS}
        batches = lambda s, train: coord_batches(X[s], Y[s], train, args.batch)
        steps = -(-len(Y['train']) // args.batch)

    model = LipreadingModel(args.variant, len(word_ix)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4 * math.sqrt(args.batch / BATCH), weight_decay=1e-2)
    total, warm = steps * args.epochs, steps
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda k: (k + 1) / warm if k < warm else 0.5 * (1 + math.cos(math.pi * (k - warm) / (total - warm))))
    scaler = torch.amp.GradScaler()
    accum = args.batch // args.micro if video else 1
    say(f'{args.variant}: {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M parameters, '
        f'{len(labels["train"])} train clips, {steps} steps/epoch, {args.epochs} epochs')
    best = -1
    with open(out / 'metrics.csv', 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'train_loss', 'val_top1', 'val_top5', 'val_f1', 'val_loss', 'minutes'])
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, losses = time.time(), []
        for i, (x, y) in enumerate(batches('train', True)):
            with torch.autocast('cuda', torch.float16):
                logits = model(**{k: v.to(dev, non_blocking=True).float() for k, v in x.items()})
                loss = F.cross_entropy(logits.float(), y.to(dev), label_smoothing=0.1) / accum
            scaler.scale(loss).backward()
            if (i + 1) % accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
            losses.append(loss.item() * accum)
            if video and i % 500 == 0:
                say(f'  ep {ep} step {i // accum}/{steps} loss {np.mean(losses[-500:]):.3f} '
                    f'{(time.time() - t0) / 60:.1f} min')
        m = evaluate(model, batches('val', False), dev)
        mins = (time.time() - t0) / 60
        say(f'ep {ep}: train loss {np.mean(losses):.3f}, val top1 {m["top1"]:.4f} top5 {m["top5"]:.4f} '
            f'f1 {m["f1"]:.4f}, {mins:.1f} min')
        with open(out / 'metrics.csv', 'a', newline='') as f:
            csv.writer(f).writerow([ep, round(float(np.mean(losses)), 4), round(m['top1'], 4), round(m['top5'], 4),
                                    round(m['f1'], 4), round(m['loss'], 4), round(mins, 1)])
        if m['top1'] > best:
            best = m['top1']
            torch.save({'model': model.state_dict(), 'epoch': ep, 'val': m, 'words': list(word_ix)}, out / 'best.pt')
    ck = torch.load(out / 'best.pt', map_location=dev)
    model.load_state_dict(ck['model'])
    m = evaluate(model, batches('test', False), dev)
    res = {'variant': args.variant, 'best_epoch': ck['epoch'], 'val': ck['val'], 'test': m}
    (out / 'test.json').write_text(json.dumps(res, indent=1))
    say(f'test (epoch {ck["epoch"]}): top1 {m["top1"]:.4f} top5 {m["top5"]:.4f} f1 {m["f1"]:.4f}')


if __name__ == '__main__':
    main()
