import os
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from train import GLipsFullClipDataset, VideoAugment, GLipsNet


class GLips15Dataset(GLipsFullClipDataset):
    """Restricts the dataset to the first N alphabetically-sorted classes."""
    def __init__(self, root_dir, split='train', transform=None, num_frames=25, num_classes=15):
        super().__init__(root_dir, split, transform, num_frames)
        idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.classes = self.classes[:num_classes]
        kept = set(self.classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples = [
            (path, self.class_to_idx[idx_to_class[old_idx]])
            for path, old_idx in self.samples
            if idx_to_class[old_idx] in kept
        ]


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

    num_workers = min(6, os.cpu_count() or 0)
    prefetch = 2 if num_workers > 0 else None
    root_dir = './GLips/lipread_files'
    NUM_CLASSES = 15

    train_transform = VideoAugment(crop_size=88, resize_size=96, is_train=True)
    val_transform = VideoAugment(crop_size=88, resize_size=96, is_train=False)
    train_dataset = GLips15Dataset(root_dir, split='train', num_frames=25,
                                   transform=train_transform, num_classes=NUM_CLASSES)
    val_dataset = GLips15Dataset(root_dir, split='validation', num_frames=25,
                                 transform=val_transform, num_classes=NUM_CLASSES)

    print(f"Classes ({NUM_CLASSES}): {train_dataset.classes}")
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=num_workers,
                              pin_memory=True, persistent_workers=num_workers > 0,
                              prefetch_factor=prefetch, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=num_workers,
                            pin_memory=True, persistent_workers=num_workers > 0,
                            prefetch_factor=prefetch)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == 'cuda', f"CUDA not available — got device '{device}'"
    print(f"Using device: {torch.cuda.get_device_name(device)}")

    model = GLipsNet(num_classes=NUM_CLASSES).to(device)

    # Differential LR: pretrained ResNet gets 10x lower rate than fresh components
    backbone_ids = {id(p) for p in model.cnn.resnet.parameters()}
    param_groups = [
        {'params': [p for p in model.parameters() if id(p) not in backbone_ids], 'lr': 1e-3},
        {'params': list(model.cnn.resnet.parameters()), 'lr': 1e-4},
    ]
    if hasattr(torch, 'compile'):
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    # bfloat16 has float32-level dynamic range — prevents NaN overflow in transformer attention
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=(amp_dtype == torch.float16))
    print(f"AMP dtype: {amp_dtype}")

    num_epochs = 30
    warmup_epochs = 3
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs - warmup_epochs, eta_min=1e-6)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs])

    best_val_acc = 0.0
    save_dir = './checkpoints_glips15'
    os.makedirs(save_dir, exist_ok=True)
    best_path = os.path.join(save_dir, 'best_model.pth')
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
        print(f"Resumed from {best_path}")
    metrics_path = os.path.join(save_dir, 'metrics.csv')
    with open(metrics_path, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_top1', 'val_top5', 'lr'])

    epoch_bar = tqdm(range(num_epochs), desc='Epochs', unit='epoch')
    for epoch in epoch_bar:
        model.train()
        running_loss, correct_train, num_samples = 0.0, 0, 0

        for data, target in tqdm(train_loader, desc=f'Train {epoch+1}/{num_epochs}', leave=False):
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                logits = model(data)
                loss = criterion(logits, target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * data.size(0)
            correct_train += (logits.detach().argmax(dim=1) == target).sum().item()
            num_samples += data.size(0)

        train_loss = running_loss / num_samples if num_samples else 0.0
        train_acc = correct_train / num_samples if num_samples else 0.0

        torch.cuda.empty_cache()
        model.eval()
        correct1, correct5, running_val_loss, total = 0, 0, 0.0, 0
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc='Val', leave=False):
                data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
                with torch.amp.autocast('cuda', dtype=amp_dtype):
                    logits = model(data)
                    running_val_loss += criterion(logits, target).item() * data.size(0)
                correct1 += (logits.argmax(dim=1) == target).sum().item()
                correct5 += (logits.topk(min(5, NUM_CLASSES), dim=1).indices == target.unsqueeze(1)).any(dim=1).sum().item()
                total += target.size(0)

        val_loss = running_val_loss / total if total else 0.0
        val_acc = correct1 / total if total else 0.0
        val_acc5 = correct5 / total if total else 0.0
        lr = scheduler.get_last_lr()[0]
        scheduler.step()
        epoch_bar.set_postfix(loss=f'{train_loss:.4f}', train_acc=f'{train_acc:.4f}',
                               top1=f'{val_acc:.4f}', top5=f'{val_acc5:.4f}', lr=f'{lr:.5f}')

        with open(metrics_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch + 1, f'{train_loss:.6f}', f'{train_acc:.6f}',
                                    f'{val_loss:.6f}', f'{val_acc:.6f}', f'{val_acc5:.6f}', f'{lr:.8f}'])

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            tqdm.write(f"Saved best model (top1={best_val_acc:.4f}, top5={val_acc5:.4f})")

    torch.save(model.state_dict(), os.path.join(save_dir, 'final_model.pth'))
    print(f"GLips15 done. Best top-1: {best_val_acc:.4f}. Metrics saved to {metrics_path}")
