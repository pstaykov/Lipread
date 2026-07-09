"""MS-TCN baseline on the 15-word task under Ameer et al.'s EXACT data + augmentation.

The MS-TCN counterpart of Transformer_based/train_15_ameer.py: identical Ameer-matched
input (uncropped full-face, 128x128, 16 frames, min-max), horizontal-flip-only image
augmentation, same-class interpolation, and plain recipe (no EMA / no label smoothing).
Only augmentation/preprocessing matches Ameer; the model is our TCNLipNet (width 384).

Results go to checkpoints_15_ameer/ (separate from the full-augmentation runs).

    python mstcn_baseline/train_15_ameer.py
"""
import os
import sys

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LIPREAD_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, LIPREAD_DIR)   # glips15, train_loop, dataset
sys.path.insert(0, SCRIPT_DIR)    # this package's model.py (must win the `model` name)

from model import TCNLipNet  # noqa: E402
from dataset15 import CLASSES, make_15_datasets_ameer  # noqa: E402
from train_loop import make_loaders, run_training, same_class_interpolation  # noqa: E402

NUM_EPOCHS = 200
PATIENCE = 25
BATCH_SIZE = 32      # matches Ameer; ample headroom on the 3060


def main():
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

    train_ds, val_ds = make_15_datasets_ameer()
    print(f"[Ameer-matched] classes={len(CLASSES)} | train={len(train_ds)} | val={len(val_ds)} "
          f"| uncropped 128x128, 16 frames, min-max, flip-only")
    train_loader, val_loader = make_loaders(train_ds, val_ds, batch_size=BATCH_SIZE)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assert device.type == 'cuda', f"CUDA not available — got device '{device}'"
    print(f"Using device: {torch.cuda.get_device_name(device)}")

    model = TCNLipNet(num_classes=len(CLASSES), dropout=0.3).to(device)  # width=384 (default)
    best1, best5 = run_training(
        model, train_loader, val_loader, num_classes=len(CLASSES), device=device,
        save_dir=os.path.join(SCRIPT_DIR, 'checkpoints_15_ameer'),
        num_epochs=NUM_EPOCHS, patience=PATIENCE,
        use_ema=False, label_smoothing=0.0, mix_fn=same_class_interpolation)
    print(f"MS-TCN (Ameer-matched) done. Best top-1: {best1:.4f} | top-5: {best5:.4f}")


if __name__ == '__main__':
    main()
