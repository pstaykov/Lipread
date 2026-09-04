"""Train the Transformer-based GLipsNet on all 500 GLips classes (stock split); the transfer source for the 15-class task.

Run from anywhere:
    python Transformer_based/train.py
"""
import os
import sys

import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_SCRIPT_DIR))  # lipreading/ for dataset, train_loop
sys.path.insert(0, _SCRIPT_DIR)                   # this package's model.py

from dataset import GLipsFullClipDataset, VideoAugment  # noqa: E402
from model import GLipsNet  # noqa: E402
from train_loop import make_loaders, run_training  # noqa: E402
from dataset15 import stock_complete_classes  # noqa: E402

ROOT_DIR = os.path.join(os.path.dirname(_SCRIPT_DIR), 'GLips_mouth', 'lipread_files')
NUM_FRAMES = 25
NUM_EPOCHS = 30
WARMUP_EPOCHS = 5
PATIENCE = 8


def main():
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

    train_transform = VideoAugment(crop_size=88, resize_size=96, is_train=True)
    val_transform = VideoAugment(crop_size=88, resize_size=96, is_train=False)
    classes = stock_complete_classes(ROOT_DIR)
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

    model = GLipsNet(num_classes=num_classes, pool='attn').to(device)  # attn pool, to match the GLips15 model it transfers to
    run_training(model, train_loader, val_loader, num_classes=num_classes, device=device,
                 save_dir=os.path.join(_SCRIPT_DIR, 'checkpoints'),
                 num_epochs=NUM_EPOCHS, warmup_epochs=WARMUP_EPOCHS, weight_decay=0.01,
                 use_ema=False, use_mixup=False, patience=PATIENCE)


if __name__ == '__main__':
    main()
