"""
Original usage (from the full repo):
    python Transformer_based/train_500_finetune.py
"""
import os
import sys

import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from dataset import GLipsFullClipDataset, VideoAugment  # noqa: E402
from model import GLipsNet, load_transfer_weights  # noqa: E402
from train_loop import make_loaders, run_training  # noqa: E402
from dataset15 import stock_complete_classes  # noqa: E402

ROOT_DIR = os.path.join(_SCRIPT_DIR, 'GLips_mouth', 'lipread_files')  # not shipped, see note above
PRETRAIN_CKPT = os.path.join(_SCRIPT_DIR, 'checkpoints', 'best_model.pth')  # not shipped, see note above
SAVE_DIR = os.path.join(_SCRIPT_DIR, 'checkpoints_500')
NUM_FRAMES = 25
NUM_EPOCHS = 15
WARMUP_EPOCHS = 2
PATIENCE = 5


def main():
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

    if not os.path.exists(PRETRAIN_CKPT):
        raise SystemExit(f"Backbone checkpoint not found at {PRETRAIN_CKPT}.\n"
                         f"Run Transformer_based/train.py first.")

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

    model = GLipsNet(num_classes=num_classes, pool='attn').to(device)
    if not os.path.exists(os.path.join(SAVE_DIR, 'checkpoint_latest.pth')):
        load_transfer_weights(model, PRETRAIN_CKPT, device, skip_prefixes=('classifier',))

    best1, best5 = run_training(model, train_loader, val_loader, num_classes=num_classes,
                                device=device, save_dir=SAVE_DIR,
                                num_epochs=NUM_EPOCHS, warmup_epochs=WARMUP_EPOCHS,
                                weight_decay=0.01, use_ema=False, use_mixup=False,
                                patience=PATIENCE)
    print(f"500-class warm-start fine-tune done. Best val top-1: {best1:.4f} | top-5: {best5:.4f}")


if __name__ == '__main__':
    main()
