"""Fine-tune the MS-TCN baseline on GLips15, warm-started from the 500-class MS-TCN (MS-TCN counterpart of Transformer_based/train_15_transfer.py).

Run AFTER mstcn_baseline/train.py (+ train_500_finetune.py) has produced
checkpoints_500/best_model.pth:
    python mstcn_baseline/train.py
    python mstcn_baseline/train_15_transfer.py
"""
import os
import sys

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LIPREAD_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, LIPREAD_DIR)   # glips15, train_loop, dataset
sys.path.insert(0, SCRIPT_DIR)    # this package's model.py (must win the `model` name)

from model import TCNLipNet, load_transfer_weights  # noqa: E402
from dataset15 import CLASSES, make_15_datasets  # noqa: E402
from train_loop import make_loaders, run_training  # noqa: E402

PRETRAIN_CKPT = os.path.join(SCRIPT_DIR, 'checkpoints_500', 'best_model.pth')
NUM_EPOCHS = 27  # capped (early stopping off) so this matches the GLipsNet transfer run's span
PATIENCE = None


def main():
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

    if not os.path.exists(PRETRAIN_CKPT):
        raise SystemExit(f"500-class MS-TCN backbone not found at {PRETRAIN_CKPT}.\n"
                         f"Run mstcn_baseline/train.py first.")

    train_ds, val_ds = make_15_datasets()
    print(f"GLips15 classes ({len(CLASSES)}): {train_ds.classes}")
    print(f"[stock split] Train: {len(train_ds)} | Val: {len(val_ds)}")
    train_loader, val_loader = make_loaders(train_ds, val_ds, batch_size=32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assert device.type == 'cuda', f"CUDA not available — got device '{device}'"
    print(f"Using device: {torch.cuda.get_device_name(device)}")

    model = TCNLipNet(num_classes=len(CLASSES), dropout=0.3).to(device)  # width=384 (default)
    save_dir = os.path.join(SCRIPT_DIR, 'checkpoints_15_transfer')
    if not os.path.exists(os.path.join(save_dir, 'checkpoint_latest.pth')):
        load_transfer_weights(model, PRETRAIN_CKPT, device, skip_prefixes=('classifier',))

    best1, best5 = run_training(model, train_loader, val_loader, num_classes=len(CLASSES),
                                device=device, save_dir=save_dir,
                                num_epochs=NUM_EPOCHS, patience=PATIENCE)
    print(f"MS-TCN GLips15 (500-backbone transfer) done. Best top-1: {best1:.4f} | top-5: {best5:.4f}")


if __name__ == '__main__':
    main()
