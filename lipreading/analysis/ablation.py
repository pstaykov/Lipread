"""Post-hoc ablation tables + confusion matrices for the GLips lip-reading models.

Scores every trained checkpoint on the SAME stock *test* split (group_split=False,
Ameer et al.'s 15 classes) and prints markdown ablation tables for both scales:

  * GLips-500 : MS-TCN vs GLipsNet (from-scratch)
  * GLips-15  : MS-TCN vs GLipsNet  x  {from-scratch, GLips500->15 transfer}
                + architecture ablations (mean-pool, no conv stem, no mouth-ROI)
                + the softmax-average ensemble of the two best backends.

For every row it reports Top-1, Top-5 AND macro-F1, and writes the full confusion
matrix (``.npy`` always, ``.png`` when matplotlib is available) plus a per-class-F1
CSV to ``analysis/figures/confusion/``. One identical protocol per row (same split,
preprocessing, num_frames), so the numbers are directly comparable.

    python analysis/ablation.py                # both tables, test split
    python analysis/ablation.py --split val    # sanity-check against metrics.csv
"""
import os
import sys
import csv
import argparse
import importlib.util

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
_LIPREAD = os.path.dirname(_HERE)
sys.path.insert(0, _LIPREAD)                                      # dataset.py, glips15, train_loop
sys.path.insert(0, os.path.join(_LIPREAD, 'Transformer_based'))  # model.py (GLipsNet)

from model import GLipsNet, _strip_orig_mod          # noqa: E402  (Transformer_based)
from dataset import GLipsFullClipDataset, VideoAugment  # noqa: E402
from dataset15 import CLASSES, ROI_ROOT, FULL_ROOT, stock_complete_classes  # noqa: E402
from train_loop import macro_f1_from_confusion        # noqa: E402

# MS-TCN model lives in a sibling model.py — load by path to dodge the name clash.
_MSTCN_PATH = os.path.join(_LIPREAD, 'mstcn_baseline', 'model.py')
_spec = importlib.util.spec_from_file_location('mstcn_model', _MSTCN_PATH)
mstcn_model = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mstcn_model)
TCNLipNet = mstcn_model.TCNLipNet

CONF_DIR = os.path.join(_HERE, 'figures', 'confusion')
GROUP_SPLIT = False  # stock split, to compare head-to-head with Ameer et al.
# A confusion matrix is only meaningful when each class has enough test clips: the
# 15-word task has 50/class (readable 15x15, like Ameer's). The 500-word task has
# ~0.1 samples per 500x500 cell — unreadably sparse — so we skip its matrix entirely
# and keep only per-class F1 + summary metrics. Threshold well above 15, below 500.
MATRIX_MAX_CLASSES = 50


def _ckpt(*parts):
    return os.path.join(_LIPREAD, *parts, 'best_model.pth')


# (label, backend, kwargs, checkpoint, root_dir). backend in {'tf','mstcn'}.
TASKS = {
    500: {
        # same complete-class list the 500-class backbone was trained on (drops the
        # corpus-degenerate classes with no val folder, e.g. soll/hier).
        'classes': stock_complete_classes(ROI_ROOT),
        'rows': [
            ('MS-TCN',   'mstcn', {},              _ckpt('mstcn_baseline', 'checkpoints'),      ROI_ROOT),
            ('GLipsNet', 'tf',    {'pool': 'attn'}, _ckpt('Transformer_based', 'checkpoints'),  ROI_ROOT),
        ],
    },
    15: {
        'classes': CLASSES,
        'rows': [
            ('MS-TCN/scratch',       'mstcn', {},                                  _ckpt('mstcn_baseline', 'checkpoints_15'),           ROI_ROOT),
            ('MS-TCN/transfer',      'mstcn', {},                                  _ckpt('mstcn_baseline', 'checkpoints_15_transfer'),  ROI_ROOT),
            ('GLipsNet/scratch',     'tf',    {'pool': 'attn'},                    _ckpt('Transformer_based', 'checkpoints_15'),          ROI_ROOT),
            ('GLipsNet/transfer',    'tf',    {'pool': 'attn'},                    _ckpt('Transformer_based', 'checkpoints_15_transfer'), ROI_ROOT),
            ('GLipsNet/mean-pool',   'tf',    {'pool': 'mean'},                    _ckpt('Transformer_based', 'checkpoints_15_meanpool'), ROI_ROOT),
            ('GLipsNet/no-stem',     'tf',    {'pool': 'attn', 'use_stem': False}, _ckpt('Transformer_based', 'checkpoints_15_nostem'),   ROI_ROOT),
            ('GLipsNet/no-ROI',      'tf',    {'pool': 'attn'},                    _ckpt('Transformer_based', 'checkpoints_15_noroi'),    FULL_ROOT),
            # Ameer-matched (uncropped 128x128, 16 frames, min-max) — head-to-head vs
            # Ameer et al.'s NASNetMobile (0.484 acc / 0.485 F1). Scored on Ameer's
            # own preprocessing (see _loader_for's 'ameer' branch).
            ('GLipsNet/ameer',       'tf',    {'pool': 'attn'},                    _ckpt('Transformer_based', 'checkpoints_15_ameer'),    FULL_ROOT),
            ('MS-TCN/ameer',         'mstcn', {},                                  _ckpt('mstcn_baseline', 'checkpoints_15_ameer'),       FULL_ROOT),
        ],
    },
}


def _build_model(backend, n, device, ckpt_path, kwargs):
    model = GLipsNet(num_classes=n, **kwargs) if backend == 'tf' else TCNLipNet(num_classes=n, **kwargs)
    model.load_state_dict(_strip_orig_mod(torch.load(ckpt_path, map_location=device)))
    return model.to(device).eval()


def _loader_for(root_dir, classes, split, batch_size,
                resize=96, crop=88, num_frames=25, normalize='imagenet'):
    """Cached test/val loader. Preprocessing defaults to our ROI recipe; the Ameer
    rows pass resize=crop=128, num_frames=16, normalize='minmax' to match Ameer."""
    val_transform = VideoAugment(crop_size=crop, resize_size=resize, is_train=False,
                                 normalize=normalize)
    dataset = GLipsFullClipDataset(root_dir, split=split, num_frames=num_frames,
                                   transform=val_transform, classes=classes,
                                   group_split=GROUP_SPLIT, require_all_classes=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=min(6, os.cpu_count() or 0), pin_memory=True)
    targets = torch.tensor([lbl for _, lbl in dataset.samples])
    return loader, targets, len(dataset.classes)


@torch.no_grad()
def _logits_for(model, loader, device, amp_dtype):
    out = []
    for data, _ in loader:
        data = data.to(device, non_blocking=True)
        with torch.amp.autocast('cuda', dtype=amp_dtype):
            out.append(model(data).float().cpu())
    return torch.cat(out, dim=0)


def _confusion(pred, targets, n):
    idx = targets * n + pred
    return torch.bincount(idx, minlength=n * n).view(n, n)


def _metrics(logits, targets, n):
    k5 = min(5, n)
    pred = logits.argmax(1)
    top1 = (pred == targets).float().mean().item()
    top5 = (logits.topk(k5, dim=1).indices == targets.unsqueeze(1)).any(1).float().mean().item()
    conf = _confusion(pred, targets, n)
    return top1, top5, macro_f1_from_confusion(conf), conf


def _save_confusion(conf, class_names, tag):
    """Always write the per-class F1 CSV. Only write the confusion matrix (.npy + a
    row-normalized .png, Ameer-style) when the class count is small enough for it to
    be meaningful (<= MATRIX_MAX_CLASSES) — so the 15-word tasks get matrices and the
    500-word task does not (0.1 samples/cell is unreadably sparse)."""
    os.makedirs(CONF_DIR, exist_ok=True)
    import numpy as np

    # per-class F1 CSV — useful at every scale (e.g. best/worst of the 500 classes)
    c = conf.double()
    tp, fp, fn = c.diag(), c.sum(0) - c.diag(), c.sum(1) - c.diag()
    denom = 2 * tp + fp + fn
    f1 = torch.where(denom > 0, 2 * tp / denom, torch.zeros_like(denom))
    with open(os.path.join(CONF_DIR, f'{tag}_per_class_f1.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['class', 'support', 'f1'])
        for i, name in enumerate(class_names):
            w.writerow([name, int(c[i].sum().item()), f'{f1[i].item():.6f}'])

    if len(class_names) > MATRIX_MAX_CLASSES:
        print(f"  (confusion matrix skipped for {tag}: {len(class_names)} classes too many "
              f"to be meaningful; per-class F1 CSV written)")
        return None

    np.save(os.path.join(CONF_DIR, f'{tag}.npy'), conf.numpy())
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        cm = conf.numpy()
        row = cm.sum(1, keepdims=True)
        norm = cm / np.clip(row, 1, None)          # row-normalized (per true label), like Ameer
        fig, ax = plt.subplots(figsize=(max(6, len(class_names) * 0.62),) * 2)
        im = ax.imshow(norm, cmap='Blues', vmin=0, vmax=1)
        # Annotate each cell with its row-normalized rate over the raw count. Zero
        # cells stay blank so the populated ones read at a glance; the text flips to
        # white on dark cells to stay legible against the colormap.
        fs = max(6, min(11, int(150 / len(class_names))))
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                ax.text(j, i, f'{round(float(norm[i, j]), 2):g}', ha='center', va='center',
                        fontsize=fs, family='serif',
                        color='white' if norm[i, j] > 0.5 else '#1a1a6e')
        ax.set_xticks(range(len(class_names))); ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=90, fontsize=10, family='serif')
        ax.set_yticklabels(class_names, fontsize=10, family='serif')
        ax.set_xlabel('Predicted label', fontsize=11, family='serif')
        ax.set_ylabel('True label', fontsize=11, family='serif')
        ax.set_title(tag, fontsize=11, family='serif')
        fig.colorbar(im, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(os.path.join(CONF_DIR, f'{tag}.png'), dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  (confusion PNG skipped for {tag}: {e})")
    return tag


def run_task(scale, cfg, split, device, amp_dtype, batch_size):
    classes = cfg['classes']
    print(f"\n# GLips-{scale}  (split='{split}', stock split)\n")

    results = []       # (label, top1, top5, f1)
    cached = {}        # backend -> (logits, targets, n) for the best init, for ensembling
    for label, backend, kwargs, path, root_dir in cfg['rows']:
        if not os.path.exists(path):
            print(f"  ! skip {label}: missing {os.path.relpath(path, _LIPREAD)}")
            continue
        ameer = 'ameer' in label.lower()  # Ameer rows use Ameer's own preprocessing
        loader, targets, n = _loader_for(
            root_dir, classes, split, batch_size,
            resize=128 if ameer else 96, crop=128 if ameer else 88,
            num_frames=16 if ameer else 25, normalize='minmax' if ameer else 'imagenet')
        model = _build_model(backend, n, device, path, kwargs)
        logits = _logits_for(model, loader, device, amp_dtype)
        top1, top5, f1, conf = _metrics(logits, targets, n)
        results.append((label, top1, top5, f1))
        names = classes if classes is not None else [str(i) for i in range(n)]
        _save_confusion(conf, names, f'glips{scale}_{label.replace("/", "_")}')
        # cache best (transfer preferred) per backend for the ensemble
        if scale == 15 and (backend not in cached or 'transfer' in label):
            cached[backend] = (logits, targets, n)
        del model
        torch.cuda.empty_cache()

    if scale == 15 and 'tf' in cached and 'mstcn' in cached:
        (tf_l, tgt, n), (tcn_l, _, _) = cached['tf'], cached['mstcn']
        probs = (F.softmax(tf_l, 1) + F.softmax(tcn_l, 1)) / 2
        top1, top5, f1, _ = _metrics(probs.log(), tgt, n)
        results.append(('Ensemble (GLipsNet+MS-TCN)', top1, top5, f1))

    _print_table(scale, results)
    return results


def _print_table(scale, results):
    print(f"\n## GLips-{scale} ablation\n")
    print("| Model | Top-1 | Top-5 | Macro-F1 |")
    print("|---|---:|---:|---:|")
    for label, top1, top5, f1 in results:
        print(f"| {label} | {top1*100:.2f} | {top5*100:.2f} | {f1*100:.2f} |")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='test', choices=['test', 'val', 'validation', 'train'])
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--scales', default='500,15', help='comma list: 500,15')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    for scale in (int(s) for s in args.scales.split(',') if s.strip()):
        run_task(scale, TASKS[scale], args.split, device, amp_dtype, args.batch_size)
    print(f"Confusion matrices + per-class F1 written to {os.path.relpath(CONF_DIR, _LIPREAD)}")


if __name__ == '__main__':
    main()
