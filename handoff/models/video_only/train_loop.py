"""
Training loop for video-only GLipsNet (Transformer) lip-reading model.
"""
import os
import csv
import copy
import math
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class EMA:
    """Exponential moving average of all float params/buffers."""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                s.copy_(v)


def mixup_cutmix(x, y, alpha=0.2, cutmix_alpha=1.0, prob=0.6, switch_prob=0.5):
    """Per-batch Mixup or CutMix on a video tensor (B, C, T, H, W).

    Returns (x, y_a, y_b, lam); loss = lam*CE(.,y_a) + (1-lam)*CE(.,y_b).
    """
    if random.random() > prob:
        return x, y, y, 1.0
    perm = torch.randperm(x.size(0), device=x.device)
    if random.random() < switch_prob:  # CutMix — swap a spatial box (same across T)
        lam = float(np.random.beta(cutmix_alpha, cutmix_alpha))
        _, _, _, H, W = x.shape
        rh, rw = int(H * math.sqrt(1 - lam)), int(W * math.sqrt(1 - lam))
        cy, cx = random.randint(0, H), random.randint(0, W)
        y1, y2 = max(cy - rh // 2, 0), min(cy + rh // 2, H)
        x1, x2 = max(cx - rw // 2, 0), min(cx + rw // 2, W)
        if y2 > y1 and x2 > x1:
            x[:, :, :, y1:y2, x1:x2] = x[perm][:, :, :, y1:y2, x1:x2]
            lam = 1.0 - (y2 - y1) * (x2 - x1) / (H * W)
        return x, y, y[perm], lam
    lam = float(np.random.beta(alpha, alpha))  # Mixup — blend whole clips
    x = lam * x + (1 - lam) * x[perm]
    return x, y, y[perm], lam


def same_class_interpolation(x, y, prob=0.5):
    """Ameer et al.'s 'interpolation' augmentation: for two samples of the SAME
    class, generate a new sample (x1 + x2)/2 with the label unchanged. Ameer apply
    it to extracted features; here it is applied at the clip level (well-defined for
    any backend, unlike their feature-noise which is calibrated to their NASNet
    features). Returns (x, y_a, y_b, lam) = (x, y, y, 1.0) so the loss stays plain
    cross-entropy on the (unchanged) label.
    """
    if random.random() > prob:
        return x, y, y, 1.0
    x = x.clone()
    for c in y.unique():
        idx = (y == c).nonzero(as_tuple=True)[0]
        if idx.numel() < 2:
            continue  # no same-class partner in this batch
        perm = idx[torch.randperm(idx.numel(), device=x.device)]
        x[idx] = 0.5 * x[idx] + 0.5 * x[perm]
    return x, y, y, 1.0


def make_loaders(train_dataset, val_dataset, batch_size=32, val_batch_size=None):
    num_workers = min(6, os.cpu_count() or 0)
    prefetch = 2 if num_workers > 0 else None
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=num_workers > 0,
                              prefetch_factor=prefetch, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size or batch_size,
                            shuffle=False, num_workers=num_workers, pin_memory=True,
                            persistent_workers=num_workers > 0,
                            prefetch_factor=prefetch)
    return train_loader, val_loader


def _model_state(model):
    return getattr(model, '_orig_mod', model).state_dict()


def _strip_orig_mod(state_dict):
    prefix = '_orig_mod.'
    if any(k.startswith(prefix) for k in state_dict):
        return {k[len(prefix):]: v for k, v in state_dict.items()}
    return state_dict


def macro_f1_from_confusion(conf):
    """Macro-averaged F1 from an integer confusion matrix ``conf[true, pred]``.

    Classes absent from both predictions and targets contribute F1=0. Returns a
    plain float; no sklearn dependency so it runs in the training loop.
    """
    conf = conf.double()
    tp = conf.diag()
    fp = conf.sum(0) - tp
    fn = conf.sum(1) - tp
    denom = 2 * tp + fp + fn
    f1 = torch.where(denom > 0, 2 * tp / denom, torch.zeros_like(denom))
    return f1.mean().item()


@torch.no_grad()
def _evaluate(eval_model, val_loader, criterion, device, amp_dtype, num_classes):
    """Returns (val_loss, top1, top5, macro_f1). Also accumulates a confusion matrix
    so macro-F1 is logged every epoch alongside accuracy."""
    eval_model.eval()
    correct1, correct5, running_val_loss, total = 0, 0, 0.0, 0
    conf = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for data, target in tqdm(val_loader, desc='Val', leave=False):
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        with torch.amp.autocast('cuda', dtype=amp_dtype):
            logits = eval_model(data)
            running_val_loss += criterion(logits, target).item() * data.size(0)
        pred = logits.argmax(dim=1)
        correct1 += (pred == target).sum().item()
        correct5 += (logits.topk(min(5, num_classes), dim=1).indices
                     == target.unsqueeze(1)).any(dim=1).sum().item()
        idx = target * num_classes + pred  # flat (true, pred) index
        conf += torch.bincount(idx.cpu(), minlength=num_classes ** 2).view(num_classes, num_classes)
        total += target.size(0)
    return (running_val_loss / total if total else 0.0,
            correct1 / total if total else 0.0,
            correct5 / total if total else 0.0,
            macro_f1_from_confusion(conf))


def run_training(model, train_loader, val_loader, *, num_classes, device, save_dir,
                 num_epochs, warmup_epochs=3, backbone_lr=1e-4, head_lr=1e-3,
                 weight_decay=0.05, label_smoothing=0.1, ema_decay=0.999,
                 use_ema=True, use_mixup=True, mix_fn=None, patience=None, resume=True,
                 grad_clip=1.0):
    """Train ``model``, early-stop on val top-1 plateau, checkpoint best top-1 AND
    best top-5 separately. Returns (best_top1, best_top5).

    patience=None disables early stopping (runs the full ``num_epochs``). When set,
    training stops once val top-1 has not improved for ``patience`` consecutive
    epochs. The cosine schedule is still sized to the full ``num_epochs`` budget, so
    an early stop simply cuts the tail rather than reshaping the LR curve.
    """
    os.makedirs(save_dir, exist_ok=True)
    latest_path = os.path.join(save_dir, 'checkpoint_latest.pth')
    best_path = os.path.join(save_dir, 'best_model.pth')            # EMA/plain weights @ best top-1
    best_top5_path = os.path.join(save_dir, 'best_top5_model.pth')  # ... @ best top-5
    final_path = os.path.join(save_dir, 'final_model.pth')          # weights, last epoch
    metrics_path = os.path.join(save_dir, 'metrics.csv')

    # EMA infra (only when use_ema): a frozen clone is loaded with the shadow weights
    # for validation and saved as the best checkpoint.
    # mix_fn is the batch mixing augmentation. Back-compat: use_mixup=True with no
    # explicit mix_fn keeps the original Mixup/CutMix; Ameer runs pass
    # same_class_interpolation; the plain 500-class runs pass neither.
    if mix_fn is None and use_mixup:
        mix_fn = mixup_cutmix

    ema = EMA(model, decay=ema_decay) if use_ema else None
    ema_model = None
    if use_ema:
        ema_model = copy.deepcopy(model).eval()
        for p in ema_model.parameters():
            p.requires_grad_(False)

    backbone_ids = {id(p) for p in model.cnn.resnet.parameters()}
    param_groups = [
        {'params': [p for p in model.parameters() if id(p) not in backbone_ids], 'lr': head_lr},
        {'params': list(model.cnn.resnet.parameters()), 'lr': backbone_lr},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=(amp_dtype == torch.float16))
    print(f"AMP dtype: {amp_dtype} | EMA: {use_ema} | Mixup: {use_mixup} | patience: {patience}")

    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=max(1, warmup_epochs))
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, num_epochs - warmup_epochs), eta_min=1e-6)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[max(1, warmup_epochs)])

    start_epoch, best_top1, best_top5, epochs_since_improve = 0, 0.0, 0.0, 0
    if resume and os.path.exists(latest_path):
        ckpt = torch.load(latest_path, map_location=device)
        getattr(model, '_orig_mod', model).load_state_dict(_strip_orig_mod(ckpt['model']))
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        scaler.load_state_dict(ckpt['scaler'])
        if use_ema and 'ema' in ckpt:
            ema.shadow = {k: v.to(device) for k, v in ckpt['ema'].items()}
        start_epoch = ckpt['epoch']
        best_top1 = ckpt.get('best_val_acc', 0.0)
        best_top5 = ckpt.get('best_val_top5', 0.0)
        epochs_since_improve = ckpt.get('epochs_since_improve', 0)
        print(f"Exact resume from epoch {start_epoch}/{num_epochs} "
              f"(best_top1={best_top1:.4f}, best_top5={best_top5:.4f})")

    write_header = not os.path.exists(metrics_path)
    with open(metrics_path, 'a', newline='') as f:
        if write_header:
            csv.writer(f).writerow(
                ['epoch', 'train_loss', 'train_acc', 'val_loss',
                 'val_top1', 'val_top5', 'val_f1', 'lr'])

    epoch_bar = tqdm(range(start_epoch, num_epochs), desc='Epochs', unit='epoch')
    for epoch in epoch_bar:
        model.train()
        running_loss, correct_train, num_samples = 0.0, 0, 0
        for data, target in tqdm(train_loader, desc=f'Train {epoch+1}/{num_epochs}', leave=False):
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
            if use_ema:
                ema.update(model)
            running_loss += loss.item() * data.size(0)
            correct_train += (logits.detach().argmax(dim=1) == y_a).sum().item()  # approximate
            num_samples += data.size(0)

        train_loss = running_loss / num_samples if num_samples else 0.0
        train_acc = correct_train / num_samples if num_samples else 0.0

        torch.cuda.empty_cache()
        if use_ema:
            ema_model.load_state_dict(ema.shadow)
            eval_model = ema_model
        else:
            eval_model = model
        val_loss, val_top1, val_top5, val_f1 = _evaluate(
            eval_model, val_loader, criterion, device, amp_dtype, num_classes)
        lr = scheduler.get_last_lr()[0]
        scheduler.step()
        epoch_bar.set_postfix(loss=f'{train_loss:.4f}', train_acc=f'{train_acc:.4f}',
                              top1=f'{val_top1:.4f}', top5=f'{val_top5:.4f}',
                              f1=f'{val_f1:.4f}', lr=f'{lr:.5f}')

        with open(metrics_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch + 1, f'{train_loss:.6f}', f'{train_acc:.6f}',
                                    f'{val_loss:.6f}', f'{val_top1:.6f}', f'{val_top5:.6f}',
                                    f'{val_f1:.6f}', f'{lr:.8f}'])

        # snapshot of the weights used for validation (EMA shadow, else live model)
        best_weights = ema.shadow if use_ema else _model_state(model)
        if val_top1 > best_top1:
            best_top1 = val_top1
            epochs_since_improve = 0
            torch.save(best_weights, best_path)
            tqdm.write(f"Saved best top-1 model (top1={best_top1:.4f}, top5={val_top5:.4f})")
        else:
            epochs_since_improve += 1
        if val_top5 > best_top5:
            best_top5 = val_top5
            torch.save(best_weights, best_top5_path)
            tqdm.write(f"Saved best top-5 model (top5={best_top5:.4f}, top1={val_top1:.4f})")

        ckpt = {'epoch': epoch + 1, 'model': _model_state(model),
                'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict(), 'best_val_acc': best_top1,
                'best_val_top5': best_top5, 'epochs_since_improve': epochs_since_improve}
        if use_ema:
            ckpt['ema'] = ema.shadow
        torch.save(ckpt, latest_path)

        if patience is not None and epochs_since_improve >= patience:
            tqdm.write(f"Early stop: val top-1 has not improved for {patience} epochs "
                       f"(best={best_top1:.4f} @ epoch {epoch + 1 - epochs_since_improve}).")
            break

    torch.save(_model_state(model) if not use_ema else ema.shadow, final_path)
    print(f"Done. Best top-1: {best_top1:.4f} | best top-5: {best_top5:.4f}. Metrics: {metrics_path}")
    return best_top1, best_top5
