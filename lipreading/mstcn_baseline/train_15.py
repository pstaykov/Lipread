"""Train the MS-TCN baseline (TCNLipNet, width 384) from scratch on GLips15.

Mirrors Transformer_based/train_15.py EXACTLY — Ameer's 15 classes, mouth-ROI
data, stock split, same recipe (Mixup/CutMix, weight-EMA, grayscale+erasing,
temporal speed-perturb, LR schedule, early stopping) — so the ONLY difference vs
GLipsNet is the temporal back-end (multi-scale TCN here, Transformer + attentive
pool there). Checkpoints/metrics: mstcn_baseline/checkpoints_15/.
"""
import os
import sys

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LIPREAD_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, LIPREAD_DIR)   # glips15, train_loop, dataset
sys.path.insert(0, SCRIPT_DIR)    # this package's model.py (must win the `model` name)

from model import TCNLipNet  # noqa: E402
from dataset15 import CLASSES, make_15_datasets  # noqa: E402
from train_loop import make_loaders, run_training  # noqa: E402

NUM_EPOCHS = 80
PATIENCE = 15


def main():
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

    train_ds, val_ds = make_15_datasets()
    print(f"GLips15 classes ({len(CLASSES)}): {train_ds.classes}")
    print(f"[stock split] Train: {len(train_ds)} | Val: {len(val_ds)}")
    train_loader, val_loader = make_loaders(train_ds, val_ds, batch_size=32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assert device.type == 'cuda', f"CUDA not available — got device '{device}'"
    print(f"Using device: {torch.cuda.get_device_name(device)}")

    model = TCNLipNet(num_classes=len(CLASSES), dropout=0.3).to(device)  # width=384 (default)
    best1, best5 = run_training(model, train_loader, val_loader, num_classes=len(CLASSES),
                                device=device, save_dir=os.path.join(SCRIPT_DIR, 'checkpoints_15'),
                                num_epochs=NUM_EPOCHS, patience=PATIENCE)
    print(f"MS-TCN GLips15 done. Best top-1: {best1:.4f} | top-5: {best5:.4f}")


if __name__ == '__main__':
    main()
