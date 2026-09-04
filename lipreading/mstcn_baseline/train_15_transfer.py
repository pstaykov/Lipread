"""Fine-tune the MS-TCN baseline on GLips15, warm-started from the 500-class MS-TCN.

The MS-TCN counterpart of Transformer_based/train_15_transfer.py. Everything ---
Ameer's 15 classes, mouth-ROI data, stock split, augmentation, EMA, Mixup/CutMix,
schedule, early stopping --- is held IDENTICAL to mstcn_baseline/train_15.py and to
the Transformer transfer run, so the only variables are (a) ImageNet vs GLips-500
init and (b) Transformer vs MS-TCN back-end.

Whole-backbone warm start: the source is the same architecture (TCNLipNet, 500-class),
so the 3D-conv + ResNet frontend and the entire MS-TCN transfer by name+shape; only
the 15-way classifier is fresh.

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
# Both transfer runs plateau within ~12 epochs and early-stopped at different points
# (MS-TCN at 25, GLipsNet at 27). Capped at 27 with early stopping off so the two
# transfer curves span an identical 27 epochs. The LR schedule is unchanged: it was
# built for 80 epochs and is restored from checkpoint_latest.pth on resume.
NUM_EPOCHS = 27
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
