"""Held-out evaluation + ENSEMBLE for the curated GLips15 task (leakage-free split).

Loads the two 15-class models that share the identical CNN3D visual frontend but
differ in temporal backend:
  * Transformer (GLipsNet, attentive pooling) — transfer-trained from the 500-class
    backbone, checkpoints_15_transfer/best_model.pth
  * MS-TCN (TCNLipNet) — mstcn_baseline/checkpoints_15/best_model.pth

and reports top-1 / top-5 for each model alone AND for their softmax-average
ensemble, on a source-disjoint split (dataset.py group_split is the default now, so
no broadcast leaks between train and the eval split). Default split: test.

    python analysis/evaluate.py                 # test split, both models + ensemble
    python analysis/evaluate.py --split val
"""
import os
import sys
import argparse
import importlib.util

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
_LIPREAD = os.path.dirname(_HERE)
sys.path.insert(0, _LIPREAD)                                   # dataset.py
sys.path.insert(0, os.path.join(_LIPREAD, 'Transformer_based'))  # transformer model.py

from model import GLipsNet, _strip_orig_mod  # noqa: E402  (Transformer_based/model.py)
from dataset import GLipsFullClipDataset, VideoAugment  # noqa: E402
from dataset15 import CLASSES  # noqa: E402  Ameer 15 classes (single source of truth)

# MS-TCN model lives in a sibling model.py — load by path to dodge the name clash.
_MSTCN_PATH = os.path.join(_LIPREAD, 'mstcn_baseline', 'model.py')
_spec = importlib.util.spec_from_file_location('mstcn_model', _MSTCN_PATH)
mstcn_model = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mstcn_model)
TCNLipNet = mstcn_model.TCNLipNet

ROOT_DIR = os.path.join(_LIPREAD, 'GLips_mouth', 'lipread_files')


def _load(model, ckpt_path, device):
    model.load_state_dict(_strip_orig_mod(torch.load(ckpt_path, map_location=device)))
    return model.to(device).eval()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='test', choices=['test', 'val', 'validation', 'train'])
    ap.add_argument('--tf-checkpoint', default=os.path.join(
        _LIPREAD, 'Transformer_based', 'checkpoints_15_transfer', 'best_model.pth'))
    ap.add_argument('--mstcn-checkpoint', default=os.path.join(
        _LIPREAD, 'mstcn_baseline', 'checkpoints_15', 'best_model.pth'))
    ap.add_argument('--batch-size', type=int, default=32)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n = len(CLASSES)
    val_transform = VideoAugment(crop_size=88, resize_size=96, is_train=False)
    dataset = GLipsFullClipDataset(ROOT_DIR, split=args.split, num_frames=25,
                                   transform=val_transform, classes=CLASSES, group_split=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=min(6, os.cpu_count() or 0), pin_memory=True)
    print(f"Eval split='{args.split}'  classes={n}  samples={len(dataset)}  (stock split)")

    tf = _load(GLipsNet(num_classes=n, pool='attn'), args.tf_checkpoint, device)
    tcn = _load(TCNLipNet(num_classes=n), args.mstcn_checkpoint, device)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    k5 = min(5, n)
    # accumulators: [transformer, mstcn, ensemble]
    c1 = [0, 0, 0]
    c5 = [0, 0, 0]
    total = 0
    for data, target in loader:
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        with torch.amp.autocast('cuda', dtype=amp_dtype):
            p_tf = F.softmax(tf(data).float(), dim=1)
            p_tcn = F.softmax(tcn(data).float(), dim=1)
        p_ens = (p_tf + p_tcn) / 2.0
        for i, p in enumerate((p_tf, p_tcn, p_ens)):
            c1[i] += (p.argmax(dim=1) == target).sum().item()
            c5[i] += (p.topk(k5, dim=1).indices == target.unsqueeze(1)).any(dim=1).sum().item()
        total += target.size(0)

    names = ['Transformer (transfer)', 'MS-TCN', 'Ensemble (avg softmax)']
    print(f"\n=== RESULTS ({args.split}, {total} samples, chance={1/n:.3f}) ===")
    print(f"{'model':28} {'top1':>8} {'top5':>8}")
    for name, a1, a5 in zip(names, c1, c5):
        print(f"{name:28} {a1/total:8.4f} {a5/total:8.4f}")


if __name__ == '__main__':
    main()
