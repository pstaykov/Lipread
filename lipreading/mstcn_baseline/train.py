"""Train the MS-TCN baseline (TCNLipNet, width 384) on all 500 GLips classes.

The MS-TCN counterpart of Transformer_based/train.py: identical CNN3D + ResNet-18
frontend, identical stock-split data (group_split=False), identical plain recipe
(no EMA / no Mixup) and early stopping, so the only difference is the temporal
back-end. Width is 384 (down from 768) to keep the baseline capacity-comparable
to GLipsNet. This 500-class model is the transfer source for the 15-class task.

Run from anywhere:
    python mstcn_baseline/train.py
"""
import os
import sys

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LIPREAD_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, LIPREAD_DIR)   # shared dataset.py + train_loop.py at the lipreading root
sys.path.insert(0, SCRIPT_DIR)    # this package's model.py (must win the `model` name)

from dataset import GLipsFullClipDataset, VideoAugment  # noqa: E402
from model import TCNLipNet  # noqa: E402
from train_loop import make_loaders, run_training  # noqa: E402
from dataset15 import stock_complete_classes  # noqa: E402

ROOT_DIR = os.path.join(LIPREAD_DIR, 'GLips_mouth', 'lipread_files')
NUM_FRAMES = 25
NUM_EPOCHS = 30
WARMUP_EPOCHS = 5
PATIENCE = 8


def main():
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

    train_transform = VideoAugment(crop_size=88, resize_size=96, is_train=True)
    val_transform = VideoAugment(crop_size=88, resize_size=96, is_train=False)
    classes = stock_complete_classes(ROOT_DIR)  # drop corpus-degenerate classes (no val)
    train_dataset = GLipsFullClipDataset(ROOT_DIR, split='train', num_frames=NUM_FRAMES,
                                         transform=train_transform, classes=classes,
                                         group_split=False)
    val_dataset = GLipsFullClipDataset(ROOT_DIR, split='validation', num_frames=NUM_FRAMES,
                                       transform=val_transform, classes=classes,
                                       group_split=False)
    num_classes = len(train_dataset.classes)
    print(f"[stock split] classes={num_classes} | train={len(train_dataset)} | val={len(val_dataset)}")
    train_loader, val_loader = make_loaders(train_dataset, val_dataset,
                                            batch_size=32, val_batch_size=16)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assert device.type == 'cuda', f"CUDA not available — got device '{device}'"
    print(f"Using device: {torch.cuda.get_device_name(device)}")

    model = TCNLipNet(num_classes=num_classes).to(device)  # width=384 (default)
    run_training(model, train_loader, val_loader, num_classes=num_classes, device=device,
                 save_dir=os.path.join(SCRIPT_DIR, 'checkpoints'),
                 num_epochs=NUM_EPOCHS, warmup_epochs=WARMUP_EPOCHS, weight_decay=0.01,
                 use_ema=False, use_mixup=False, patience=PATIENCE)


if __name__ == '__main__':
    main()
