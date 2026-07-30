"""Redo the GLips500->GLips15 in-domain transfer using the NEW complete 500-class
backbone (checkpoints_500/best_model.pth, warm-started+fine-tuned from the old
498-class run once 'hier'/'soll' were fixed) instead of the old 498-class one used
by train_15_transfer.py / checkpoints_15_transfer.

Everything else — classes, stock split, augmentation, schedule, early stopping — is
held identical to train_15_transfer.py for an apples-to-apples comparison against
the existing 0.663 test SOTA.

Run AFTER Transformer_based/train_500_finetune.py has produced checkpoints_500/best_model.pth:
    python Transformer_based/train_15_transfer_500full.py
"""
import os
import sys

import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_SCRIPT_DIR))  # lipreading/ for glips15, train_loop, dataset

from model import GLipsNet, load_transfer_weights  # noqa: E402
from dataset15 import CLASSES, make_15_datasets  # noqa: E402
from train_loop import make_loaders, run_training  # noqa: E402

PRETRAIN_CKPT = os.path.join(_SCRIPT_DIR, 'checkpoints_500', 'best_model.pth')
NUM_EPOCHS = 80
PATIENCE = 15


def main():
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

    if not os.path.exists(PRETRAIN_CKPT):
        raise SystemExit(f"500-class backbone not found at {PRETRAIN_CKPT}.\n"
                         f"Run Transformer_based/train_500_finetune.py first.")

    train_ds, val_ds = make_15_datasets()
    print(f"GLips15 classes ({len(CLASSES)}): {train_ds.classes}")
    print(f"[stock split] Train: {len(train_ds)} | Val: {len(val_ds)}")
    train_loader, val_loader = make_loaders(train_ds, val_ds, batch_size=32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assert device.type == 'cuda', f"CUDA not available — got device '{device}'"
    print(f"Using device: {torch.cuda.get_device_name(device)}")

    model = GLipsNet(num_classes=len(CLASSES), dropout=0.3, pool='attn').to(device)
    save_dir = os.path.join(_SCRIPT_DIR, 'checkpoints_15_transfer_500full')
    if not os.path.exists(os.path.join(save_dir, 'checkpoint_latest.pth')):
        load_transfer_weights(model, PRETRAIN_CKPT, device, skip_prefixes=('classifier',))

    best1, best5 = run_training(model, train_loader, val_loader, num_classes=len(CLASSES),
                                device=device, save_dir=save_dir,
                                num_epochs=NUM_EPOCHS, patience=PATIENCE)
    print(f"GLips15 (500-full-backbone transfer) done. Best val top-1: {best1:.4f} | top-5: {best5:.4f}")


if __name__ == '__main__':
    main()
