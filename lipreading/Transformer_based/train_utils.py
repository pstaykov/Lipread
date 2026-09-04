"""Shared training loop (train_classifier) for the LRW-15 pretrainer and GLips15 transfer trainer; train_15.py stays self-contained."""
import os
import csv

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import GLipsNet, _model_state, _strip_orig_mod  # noqa: E402
from train_15 import EMA, mixup_cutmix  # single source of truth for these helpers


def make_loaders(train_dataset, val_dataset, batch_size=32):
    num_workers = min(6, os.cpu_count() or 0)
    prefetch = 2 if num_workers > 0 else None
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=num_workers > 0,
                              prefetch_factor=prefetch, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True,
                            persistent_workers=num_workers > 0,
                            prefetch_factor=prefetch)
    return train_loader, val_loader


def train_classifier(model, train_loader, val_loader, *, num_classes, device,
                     save_dir, num_epochs=80, warmup_epochs=3,
                     backbone_lr=1e-4, head_lr=1e-3, weight_decay=0.05,
                     label_smoothing=0.1, ema_decay=0.999, use_mixup=True,
                     resume=True):
    """Train `model` and checkpoint EMA weights (mirrors train_15.py's recipe exactly). Returns best val top-1."""
    import copy

    ema_model = copy.deepcopy(model).eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)
    ema = EMA(model, decay=ema_decay)

    backbone_ids = {id(p) for p in model.cnn.resnet.parameters()}
    param_groups = [
        {'params': [p for p in model.parameters() if id(p) not in backbone_ids], 'lr': head_lr},
        {'params': list(model.cnn.resnet.parameters()), 'lr': backbone_lr},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=(amp_dtype == torch.float16))
    print(f"AMP dtype: {amp_dtype}")

    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs - warmup_epochs, eta_min=1e-6)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs])

    os.makedirs(save_dir, exist_ok=True)
    latest_path = os.path.join(save_dir, 'checkpoint_latest.pth')
    best_path = os.path.join(save_dir, 'best_model.pth')    # EMA weights at best val
    final_path = os.path.join(save_dir, 'final_model.pth')  # EMA weights, last epoch
    metrics_path = os.path.join(save_dir, 'metrics.csv')

    start_epoch, best_val_acc = 0, 0.0
    if resume and os.path.exists(latest_path):
        ckpt = torch.load(latest_path, map_location=device)
        model.load_state_dict(_strip_orig_mod(ckpt['model']))
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        scaler.load_state_dict(ckpt['scaler'])
        ema.shadow = {k: v.to(device) for k, v in ckpt['ema'].items()}
        start_epoch, best_val_acc = ckpt['epoch'], ckpt['best_val_acc']
        print(f"Exact resume from epoch {start_epoch}/{num_epochs} (best_val_acc={best_val_acc:.4f})")

    write_header = not os.path.exists(metrics_path)
    with open(metrics_path, 'a', newline='') as f:
        if write_header:
            csv.writer(f).writerow(
                ['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_top1', 'val_top5', 'lr'])

    epoch_bar = tqdm(range(start_epoch, num_epochs), desc='Epochs', unit='epoch')
    for epoch in epoch_bar:
        model.train()
        running_loss, correct_train, num_samples = 0.0, 0, 0

        for data, target in tqdm(train_loader, desc=f'Train {epoch+1}/{num_epochs}', leave=False):
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            if use_mixup:
                data, y_a, y_b, lam = mixup_cutmix(data, target)
            else:
                y_a, y_b, lam = target, target, 1.0
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                logits = model(data)
                loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)
            running_loss += loss.item() * data.size(0)
            correct_train += (logits.detach().argmax(dim=1) == y_a).sum().item()  # approximate
            num_samples += data.size(0)

        train_loss = running_loss / num_samples if num_samples else 0.0
        train_acc = correct_train / num_samples if num_samples else 0.0

        torch.cuda.empty_cache()
        ema_model.load_state_dict(ema.shadow)
        ema_model.eval()
        correct1, correct5, running_val_loss, total = 0, 0, 0.0, 0
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc='Val(EMA)', leave=False):
                data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
                with torch.amp.autocast('cuda', dtype=amp_dtype):
                    logits = ema_model(data)
                    running_val_loss += criterion(logits, target).item() * data.size(0)
                correct1 += (logits.argmax(dim=1) == target).sum().item()
                correct5 += (logits.topk(min(5, num_classes), dim=1).indices
                             == target.unsqueeze(1)).any(dim=1).sum().item()
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
            torch.save(ema.shadow, best_path)
            tqdm.write(f"Saved best EMA model (top1={best_val_acc:.4f}, top5={val_acc5:.4f})")

        torch.save({'epoch': epoch + 1, 'model': _model_state(model),
                    'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                    'scaler': scaler.state_dict(), 'ema': ema.shadow,
                    'best_val_acc': best_val_acc}, latest_path)

    torch.save(ema.shadow, final_path)
    print(f"Done. Best EMA top-1: {best_val_acc:.4f}. Metrics: {metrics_path}")
    return best_val_acc
