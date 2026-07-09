"""Train GLipsNet from scratch on Ameer et al.'s 15-class GLips benchmark.

Ameer's 15 classes, mouth-ROI input, stock GLips folder split (group_split=False)
so the result is head-to-head comparable to Ameer et al. Regularization for the
small set (~400 clips/class): attentive temporal pooling, Mixup + CutMix, random
grayscale + random-erasing, temporal speed-perturbation, and weight EMA (both
validation and best_model.pth come from the EMA weights). Early stopping caps the
run at the validation plateau.

``CLASSES``/``EMA``/``mixup_cutmix`` are re-exported here so existing
``from train_15 import ...`` call sites keep working.
"""
import os
import sys

import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_SCRIPT_DIR))  # lipreading/ for glips15, train_loop, dataset

from model import GLipsNet  # noqa: E402
from dataset15 import CLASSES, make_15_datasets, ROI_ROOT  # noqa: E402  (re-exported)
from train_loop import make_loaders, run_training, EMA, mixup_cutmix  # noqa: E402  (re-exported)

NUM_EPOCHS = 80
PATIENCE = 15


def build_model(num_classes, device, pool='attn', use_stem=True):
    return GLipsNet(num_classes=num_classes, dropout=0.3, pool=pool,
                    use_stem=use_stem).to(device)


def run_15(save_dir, *, root_dir=ROI_ROOT, pool='attn', use_stem=True):
    """Shared driver for the from-scratch GLips15 runs and their ablations."""
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

    train_ds, val_ds = make_15_datasets(root_dir=root_dir)
    print(f"GLips15 classes ({len(CLASSES)}): {train_ds.classes}")
    print(f"[stock split] Train: {len(train_ds)} | Val: {len(val_ds)} | root: {root_dir}")
    train_loader, val_loader = make_loaders(train_ds, val_ds, batch_size=32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assert device.type == 'cuda', f"CUDA not available — got device '{device}'"
    print(f"Using device: {torch.cuda.get_device_name(device)} | pool={pool} use_stem={use_stem}")

    model = build_model(len(CLASSES), device, pool=pool, use_stem=use_stem)
    return run_training(model, train_loader, val_loader, num_classes=len(CLASSES),
                        device=device, save_dir=save_dir,
                        num_epochs=NUM_EPOCHS, patience=PATIENCE)


if __name__ == '__main__':
    run_15(os.path.join(_SCRIPT_DIR, 'checkpoints_15'))
