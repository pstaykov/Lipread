"""Held-out test-set evaluation for the 500-class GLipsNet backbone.

    python analysis/evaluate_500.py                 # test split
    python analysis/evaluate_500.py --split val
"""
import os
import sys
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
_LIPREAD = os.path.dirname(_HERE)
sys.path.insert(0, _LIPREAD)
sys.path.insert(0, os.path.join(_LIPREAD, 'Transformer_based'))

from model import GLipsNet, _strip_orig_mod  # noqa: E402
from dataset import GLipsFullClipDataset, VideoAugment  # noqa: E402
from dataset15 import stock_complete_classes  # noqa: E402
from bootstrap_ci import bootstrap_ci  # noqa: E402

ROOT_DIR = os.path.join(_LIPREAD, 'GLips_mouth', 'lipread_files')


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='test', choices=['test', 'val', 'validation', 'train'])
    ap.add_argument('--checkpoint', default=os.path.join(
        _LIPREAD, 'Transformer_based', 'checkpoints_500', 'best_model.pth'))
    ap.add_argument('--batch-size', type=int, default=32)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classes = stock_complete_classes(ROOT_DIR)
    n = len(classes)
    val_transform = VideoAugment(crop_size=88, resize_size=96, is_train=False)
    dataset = GLipsFullClipDataset(ROOT_DIR, split=args.split, num_frames=25,
                                   transform=val_transform, classes=classes, group_split=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=min(6, os.cpu_count() or 0), pin_memory=True)
    print(f"Eval split='{args.split}'  classes={n}  samples={len(dataset)}  (stock split)")

    model = GLipsNet(num_classes=n, pool='attn').to(device)
    model.load_state_dict(_strip_orig_mod(torch.load(args.checkpoint, map_location=device)))
    model.eval()

    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    k5 = min(5, n)
    correct1, correct5 = [], []
    for data, target in loader:
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        with torch.amp.autocast('cuda', dtype=amp_dtype):
            logits = model(data)
        correct1.append((logits.argmax(dim=1) == target).cpu().numpy())
        correct5.append((logits.topk(k5, dim=1).indices == target.unsqueeze(1)).any(dim=1).cpu().numpy())
    correct1 = np.concatenate(correct1)
    correct5 = np.concatenate(correct5)
    total = correct1.shape[0]

    m1, lo1, hi1, hw1 = bootstrap_ci(correct1)
    m5, lo5, hi5, hw5 = bootstrap_ci(correct5)
    print(f"\n=== RESULTS ({args.split}, {total} samples, chance={1/n:.4f}) ===")
    print(f"checkpoint={args.checkpoint}")
    print(f"top1={m1:.4f}  top5={m5:.4f}")
    print(f"top1 95% bootstrap CI: [{lo1:.4f}, {hi1:.4f}]  (+/-{hw1:.4f}, n_boot=10000)")
    print(f"top5 95% bootstrap CI: [{lo5:.4f}, {hi5:.4f}]  (+/-{hw5:.4f}, n_boot=10000)")


if __name__ == '__main__':
    main()
