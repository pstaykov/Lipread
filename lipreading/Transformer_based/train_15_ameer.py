"""GLipsNet on the 15-word task under Ameer et al.'s EXACT data + augmentation.

Head-to-head with Ameer et al. (NASNetMobile, GLips 0.484 acc / 0.485 F1): same
uncropped full-face input, resize 128x128, 16 frames, min-max normalization,
horizontal-flip-only image augmentation, same-class interpolation, and a plain
recipe (no EMA, no label smoothing, plain cross-entropy). Only the augmentation/
preprocessing matches Ameer — the model is our GLipsNet, so the comparison
isolates architecture under an identical data regime.

Separate from train_15.py (which keeps our full augmentation); results go to
checkpoints_15_ameer/ and do NOT overwrite the full-augmentation runs.

    python Transformer_based/train_15_ameer.py
"""
import os
import sys

import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_SCRIPT_DIR))  # lipreading/ for glips15, train_loop
sys.path.insert(0, _SCRIPT_DIR)                   # this package's model.py

from model import GLipsNet  # noqa: E402
from dataset15 import CLASSES, make_15_datasets_ameer  # noqa: E402
from train_loop import make_loaders, run_training, same_class_interpolation  # noqa: E402

NUM_EPOCHS = 200     # Ameer trained for 200 epochs
PATIENCE = 25        # our early-stopping policy still applies
BATCH_SIZE = 32      # matches Ameer; 128x128x16 peaks ~2 GB, ample headroom on the 3060


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

    # Our model (attentive pool + conv stem); only the data/aug matches Ameer.
    model = GLipsNet(num_classes=len(CLASSES), dropout=0.3, pool='attn').to(device)
    best1, best5 = run_training(
        model, train_loader, val_loader, num_classes=len(CLASSES), device=device,
        save_dir=os.path.join(_SCRIPT_DIR, 'checkpoints_15_ameer'),
        num_epochs=NUM_EPOCHS, patience=PATIENCE,
        use_ema=False, label_smoothing=0.0, mix_fn=same_class_interpolation)
    print(f"GLipsNet (Ameer-matched) done. Best top-1: {best1:.4f} | top-5: {best5:.4f}")


if __name__ == '__main__':
    main()
