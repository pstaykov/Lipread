import os
import csv
import signal
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from dataset import WordBoundaryDataset
from model import WordBoundaryDetector

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def _tqdm(iterable, **kwargs):
    if tqdm is not None:
        return tqdm(iterable, **kwargs)
    return iterable


def _tqdm_write(msg):
    if tqdm is not None:
        tqdm.write(msg)
    else:
        print(msg)


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, best_val_loss):
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'scaler': scaler.state_dict(),
        'best_val_loss': best_val_loss,
    }, path)


def load_checkpoint(path, model, optimizer, scheduler, scaler, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    scheduler.load_state_dict(ckpt['scheduler'])
    scaler.load_state_dict(ckpt['scaler'])
    return ckpt['epoch'], ckpt['best_val_loss']


def prune_periodic_checkpoints(save_dir, keep=3):
    import glob
    paths = sorted(glob.glob(os.path.join(save_dir, 'checkpoint_epoch_*.pth')))
    for old in paths[:-keep]:
        os.remove(old)


def train(video_paths, srt_paths, epochs=20, batch_size=8, lr=1e-4,
          device="cuda", save_dir='./checkpoints_wb'):
    dataset = WordBoundaryDataset(video_paths, srt_paths)

    val_size = max(1, int(0.1 * len(dataset)))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=4)

    model = WordBoundaryDetector().to(device)

    speaking_ratio = 0.4
    pos_weight = torch.tensor([(1 - speaking_ratio) / speaking_ratio]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=(amp_dtype == torch.float16))

    os.makedirs(save_dir, exist_ok=True)
    latest_path = os.path.join(save_dir, 'checkpoint_latest.pth')
    best_path = os.path.join(save_dir, 'best_model.pth')
    metrics_path = os.path.join(save_dir, 'metrics.csv')

    CHECKPOINT_EVERY = 5

    start_epoch = 0
    best_val_loss = float('inf')

    if os.path.exists(latest_path):
        try:
            start_epoch, best_val_loss = load_checkpoint(
                latest_path, model, optimizer, scheduler, scaler, device)
            print(f"Resumed from {latest_path} (epoch {start_epoch}, best_val_loss={best_val_loss:.4f})")
        except (RuntimeError, KeyError) as e:
            print(f"WARNING: {latest_path} incompatible — starting fresh. ({e})")

    if start_epoch == 0 or not os.path.exists(metrics_path):
        with open(metrics_path, 'w', newline='') as f:
            csv.writer(f).writerow(['epoch', 'train_loss', 'val_loss', 'lr'])

    _abort = False

    def _handle_sigint(sig, frame):
        nonlocal _abort
        print("\nInterrupt received — will save checkpoint after this epoch.")
        _abort = True

    signal.signal(signal.SIGINT, _handle_sigint)

    epoch_bar = _tqdm(range(start_epoch, epochs), desc='Epochs', unit='epoch')
    for epoch in epoch_bar:
        model.train()
        running_loss = 0.0
        num_batches = 0

        train_bar = _tqdm(train_loader, desc=f'Train {epoch+1}/{epochs}', leave=False, unit='batch')
        for clips, labels in train_bar:
            clips, labels = clips.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                loss = criterion(model(clips), labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            num_batches += 1
            if tqdm is not None:
                train_bar.set_postfix(loss=f'{loss.item():.4f}')

        train_loss = running_loss / num_batches if num_batches > 0 else 0.0

        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for clips, labels in _tqdm(val_loader, desc='Val', leave=False, unit='batch'):
                clips, labels = clips.to(device), labels.to(device)
                with torch.amp.autocast('cuda', dtype=amp_dtype):
                    val_loss += criterion(model(clips), labels).item()
                val_batches += 1

        val_loss = val_loss / val_batches if val_batches > 0 else 0.0
        lr = scheduler.get_last_lr()[0]
        scheduler.step()

        if tqdm is not None:
            epoch_bar.set_postfix(train=f'{train_loss:.4f}', val=f'{val_loss:.4f}', lr=f'{lr:.5f}')
        else:
            print(f"epoch {epoch+1}/{epochs} train {train_loss:.4f} val {val_loss:.4f} lr {lr:.6f}")

        with open(metrics_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch + 1, f'{train_loss:.6f}', f'{val_loss:.6f}', f'{lr:.8f}'])

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)
            _tqdm_write(f"Saved best model (val_loss={best_val_loss:.4f})")

        save_checkpoint(latest_path, model, optimizer, scheduler, scaler, epoch + 1, best_val_loss)

        if (epoch + 1) % CHECKPOINT_EVERY == 0:
            periodic_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1:04d}.pth')
            save_checkpoint(periodic_path, model, optimizer, scheduler, scaler, epoch + 1, best_val_loss)
            prune_periodic_checkpoints(save_dir, keep=3)
            _tqdm_write(f"Saved periodic checkpoint at epoch {epoch+1}")

        if _abort:
            _tqdm_write("Abort flag set — stopping training. Resume with the same command.")
            break

    torch.save(model.state_dict(), os.path.join(save_dir, 'final_model.pth'))
    print(f"Training finished. Best val loss: {best_val_loss:.4f}. Metrics saved to {metrics_path}")
    return model
