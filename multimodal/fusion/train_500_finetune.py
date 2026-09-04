"""Warm-start the fusion model on the full 500-class GLips set from the old 498-class checkpoint,
carrying over every matching-shaped tensor (load_visual_weights) except the final classifier.

Run from multimodal/fusion/:
    python train_500_finetune.py
"""
import os
import sys
import csv
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import (  # noqa: E402
    GLipsNet, MultimodalGLipsDataset, VideoAugment, WhisperExtractor,
    collate_fn, _model_state, _strip_orig_mod, load_visual_weights,
    save_checkpoint, load_checkpoint, prune_periodic_checkpoints,
    WHISPER_MODEL_NAME,
)

PRETRAIN_CKPT = './models/checkpoints_reg/best_model.pth'
SAVE_DIR = './models/checkpoints_500'
ROOT_DIR = '../../lipreading/GLips_mouth/lipread_files'
AUDIO_ROOT = '../../lipreading/GLips/lipread_files'
AUDIO_CACHE_DIR = './cache/audio_cache'
NUM_FRAMES = 25
NUM_EPOCHS = 8
WARMUP_EPOCHS = 1
EARLY_STOP_PATIENCE = 3
FREEZE_BACKBONE_EPOCHS = 1
CHECKPOINT_EVERY = 5
# saves a mid-epoch checkpoint and exits cleanly every TIME_BUDGET_SEC; rerun to resume same epoch
TIME_BUDGET_SEC = int(os.environ.get('TIME_BUDGET_SEC', '900'))


def main():
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

    latest_path_check = os.path.join(SAVE_DIR, 'checkpoint_latest.pth')
    if not os.path.exists(latest_path_check) and not os.path.exists(PRETRAIN_CKPT):
        raise SystemExit(f"498-class fused checkpoint not found at {PRETRAIN_CKPT}.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assert device.type == 'cuda', f"CUDA not available — got '{device}'"
    print(f"Device: {torch.cuda.get_device_name(device)}")

    extractor = WhisperExtractor(WHISPER_MODEL_NAME, device=device)

    train_tf = VideoAugment(crop_size=88, resize_size=96, is_train=True, time_mask_max=5)
    val_tf = VideoAugment(crop_size=88, resize_size=96, is_train=False)

    CLASSES = sorted(d for d in os.listdir(ROOT_DIR) if os.path.isdir(os.path.join(ROOT_DIR, d)))
    print(f"classes={len(CLASSES)}")

    audio_cache = AUDIO_CACHE_DIR if os.path.exists(os.path.join(AUDIO_CACHE_DIR, 'index.json')) else None
    if audio_cache is None:
        print("No audio cache found — decoding .m4a on the fly.")

    train_ds = MultimodalGLipsDataset(ROOT_DIR, split='train', num_frames=NUM_FRAMES,
                                      transform=train_tf, audio_root=AUDIO_ROOT,
                                      cache_dir=audio_cache, group_split=False,
                                      classes=CLASSES)
    val_ds = MultimodalGLipsDataset(ROOT_DIR, split='validation', num_frames=NUM_FRAMES,
                                    transform=val_tf, audio_root=AUDIO_ROOT,
                                    cache_dir=audio_cache, group_split=False,
                                    classes=CLASSES)
    print(f"train={len(train_ds)}  val={len(val_ds)}")

    NUM_WORKERS = 4
    TRAIN_BATCH = 16

    def make_train_loader(epoch, skip_batches=0):
        """Deterministic per-epoch shuffle via randperm (not shuffle=True) so a mid-epoch resume can jump straight to the right indices."""
        g = torch.Generator()
        g.manual_seed(20260815 + epoch)
        perm = torch.randperm(len(train_ds), generator=g).tolist()
        n_batches_total = len(perm) // TRAIN_BATCH
        usable = perm[:n_batches_total * TRAIN_BATCH]
        remaining = usable[skip_batches * TRAIN_BATCH:]
        loader = DataLoader(train_ds, batch_size=TRAIN_BATCH, sampler=remaining,
                            num_workers=NUM_WORKERS, pin_memory=True,
                            persistent_workers=False,
                            drop_last=True, collate_fn=collate_fn)
        return loader, n_batches_total

    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True,
                            persistent_workers=NUM_WORKERS > 0,
                            collate_fn=collate_fn)

    model = GLipsNet(num_classes=len(CLASSES), use_audio=True)

    os.makedirs(SAVE_DIR, exist_ok=True)
    latest_path = os.path.join(SAVE_DIR, 'checkpoint_latest.pth')
    best_path = os.path.join(SAVE_DIR, 'best_model.pth')
    metrics_path = os.path.join(SAVE_DIR, 'metrics.csv')

    # only warm-start when starting fresh, so a resume isn't overwritten by the 498-class weights
    if not os.path.exists(latest_path):
        load_visual_weights(model, PRETRAIN_CKPT, device)

    model.to(device)

    backbone_ids = {id(p) for p in model.cnn.resnet.parameters()}
    param_groups = [
        {'params': [p for p in model.parameters() if id(p) not in backbone_ids], 'lr': 3e-4},
        {'params': list(model.cnn.resnet.parameters()), 'lr': 3e-5},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.05)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=(amp_dtype == torch.float16))

    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=max(1, WARMUP_EPOCHS))
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, NUM_EPOCHS - WARMUP_EPOCHS), eta_min=1e-6)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[max(1, WARMUP_EPOCHS)])

    start_epoch, best_val_acc = 0, 0.0
    if os.path.exists(latest_path):
        start_epoch, best_val_acc = load_checkpoint(
            latest_path, model, optimizer, scheduler, scaler, device, NUM_EPOCHS)
        print(f"Resumed at epoch {start_epoch}/{NUM_EPOCHS}, best={best_val_acc:.4f}")

    midepoch_path = os.path.join(SAVE_DIR, 'checkpoint_midepoch.pth')
    batch_skip = 0
    if os.path.exists(midepoch_path):
        mck = torch.load(midepoch_path, map_location=device)
        if mck['epoch'] == start_epoch:
            getattr(model, '_orig_mod', model).load_state_dict(_strip_orig_mod(mck['model']))
            optimizer.load_state_dict(mck['optimizer'])
            scaler.load_state_dict(mck['scaler'])
            batch_skip = mck['batch_idx']
            print(f"Resuming mid-epoch {start_epoch}: skipping {batch_skip} already-done batches")
        else:
            os.remove(midepoch_path)

    if start_epoch == 0 or not os.path.exists(metrics_path):
        with open(metrics_path, 'w', newline='') as f:
            csv.writer(f).writerow(['epoch', 'train_loss', 'train_acc', 'val_loss',
                                    'val_top1', 'val_top5', 'lr'])

    epochs_since_best = 0
    epoch_bar = tqdm(range(start_epoch, NUM_EPOCHS), desc='Epochs', unit='epoch', dynamic_ncols=True)
    for epoch in epoch_bar:
        model.train()
        backbone_frozen = epoch < FREEZE_BACKBONE_EPOCHS
        for p in model.cnn.resnet.parameters():
            p.requires_grad_(not backbone_frozen)
        if backbone_frozen:
            model.cnn.resnet.eval()

        skip = batch_skip if epoch == start_epoch else 0
        batch_skip = 0  # only the very first epoch resumed into can have a mid-epoch skip
        train_loader, n_batches = make_train_loader(epoch, skip_batches=skip)

        running_loss = correct_train = num_samples = 0
        t0 = time.time()
        bi = skip
        for video, wav, target in tqdm(train_loader, desc=f'Train {epoch+1}/{NUM_EPOCHS}',
                                       initial=skip, total=n_batches,
                                       leave=False, dynamic_ncols=True):
            video = video.to(device, non_blocking=True)
            wav = wav.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            with torch.no_grad():
                audio = extractor.encode(wav)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                logits = model(video, audio)
                loss = criterion(logits, target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * video.size(0)
            correct_train += (logits.detach().argmax(1) == target).sum().item()
            num_samples += video.size(0)
            bi += 1

            if TIME_BUDGET_SEC and (time.time() - t0) > TIME_BUDGET_SEC and bi < n_batches:
                torch.save({'epoch': epoch, 'batch_idx': bi, 'model': _model_state(model),
                           'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict()},
                          midepoch_path)
                tqdm.write(f"Time budget reached mid-epoch {epoch+1} at batch {bi}/{n_batches} "
                          f"— checkpointed and exiting cleanly. Re-run to continue.")
                return

        if os.path.exists(midepoch_path):
            os.remove(midepoch_path)
        train_loss = running_loss / num_samples if num_samples else 0.0
        train_acc = correct_train / num_samples if num_samples else 0.0

        torch.cuda.empty_cache()
        model.eval()
        correct1 = correct5 = running_val_loss = total = 0
        with torch.no_grad():
            for video, wav, target in tqdm(val_loader, desc='Val', leave=False, dynamic_ncols=True):
                video = video.to(device, non_blocking=True)
                wav = wav.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                audio = extractor.encode(wav)
                with torch.amp.autocast('cuda', dtype=amp_dtype):
                    logits = model(video, audio)
                    running_val_loss += criterion(logits, target).item() * video.size(0)
                correct1 += (logits.argmax(1) == target).sum().item()
                correct5 += (logits.topk(5, dim=1).indices == target.unsqueeze(1)).any(1).sum().item()
                total += target.size(0)

        val_loss = running_val_loss / total if total else 0.0
        val_acc = correct1 / total if total else 0.0
        val_acc5 = correct5 / total if total else 0.0
        lr = scheduler.get_last_lr()[0]
        scheduler.step()

        tqdm.write(f"[{epoch+1:>3}/{NUM_EPOCHS}] loss={train_loss:.4f} train={train_acc:.4f} "
                  f"top1={val_acc:.4f} top5={val_acc5:.4f} lr={lr:.5f}")

        with open(metrics_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch + 1, f'{train_loss:.6f}', f'{train_acc:.6f}',
                                    f'{val_loss:.6f}', f'{val_acc:.6f}', f'{val_acc5:.6f}', f'{lr:.8f}'])

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_since_best = 0
            torch.save(_model_state(model), best_path)
            tqdm.write(f"  ** best model saved (top1={best_val_acc:.4f}, top5={val_acc5:.4f})")
        else:
            epochs_since_best += 1

        save_checkpoint(latest_path, model, optimizer, scheduler, scaler, epoch + 1,
                        best_val_acc, NUM_EPOCHS)

        if (epoch + 1) % CHECKPOINT_EVERY == 0:
            periodic = os.path.join(SAVE_DIR, f'checkpoint_epoch_{epoch+1:04d}.pth')
            save_checkpoint(periodic, model, optimizer, scheduler, scaler, epoch + 1,
                            best_val_acc, NUM_EPOCHS)
            prune_periodic_checkpoints(SAVE_DIR, keep=3)

        if epochs_since_best >= EARLY_STOP_PATIENCE:
            tqdm.write(f"Early stop: {epochs_since_best} epochs without improving "
                      f"(best top1={best_val_acc:.4f}).")
            break

    torch.save(_model_state(model), os.path.join(SAVE_DIR, 'final_model.pth'))
    print(f"Done. Best top-1: {best_val_acc:.4f}. Metrics: {metrics_path}")


if __name__ == '__main__':
    main()
