"""
Late fusion on the first 15 alphabetically-sorted GLips words.
Uses the GLipsNet checkpoint trained on these same 15 classes.
"""

import os
import csv
import signal
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from multimodal.av_dataset import GLipsAVDataset
from models import MetaLearner, VideoAugment, load_visual_encoder, load_audio_encoder

NUM_WORDS = 15


if __name__ == '__main__':
    root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', '..', 'lipreading', 'GLips_mouth', 'lipread_files')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assert device.type == 'cuda', f"CUDA not available — got device '{device}'"
    print(f"Using device: {torch.cuda.get_device_name(device)}")

    all_classes = sorted([d for d in os.listdir(root_dir)
                          if os.path.isdir(os.path.join(root_dir, d))])
    selected_classes = all_classes[:NUM_WORDS]
    print(f"Selected {NUM_WORDS} words: {selected_classes}")

    train_dataset = GLipsAVDataset(root_dir, split='train', classes=selected_classes,
                                   transform=VideoAugment(crop_size=88, resize_size=96, is_train=True))
    val_dataset = GLipsAVDataset(root_dir, split='validation', classes=selected_classes,
                                 transform=VideoAugment(crop_size=88, resize_size=96, is_train=False))

    num_workers = min(4, os.cpu_count() or 0)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=num_workers > 0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False,
                            num_workers=num_workers, pin_memory=True,
                            persistent_workers=num_workers > 0)

    ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             '..', '..', 'lipreading', 'Transformer_based', 'checkpoints_15_transfer', 'best_model.pth')
    visual_enc = load_visual_encoder(ckpt_path, device, num_classes=NUM_WORDS)
    audio_enc = load_audio_encoder(device)

    meta = MetaLearner(num_classes=NUM_WORDS).to(device)
    optimizer = torch.optim.AdamW(meta.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints_15')
    os.makedirs(save_dir, exist_ok=True)
    best_path = os.path.join(save_dir, 'best_meta.pth')
    metrics_path = os.path.join(save_dir, 'metrics.csv')

    best_val_acc = 0.0
    num_epochs = 100

    with open(metrics_path, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])

    _abort = False

    def _handle_sigint(sig, frame):
        global _abort
        print("\nInterrupt — stopping after this epoch.")
        _abort = True

    signal.signal(signal.SIGINT, _handle_sigint)

    for epoch in range(num_epochs):
        meta.train()
        running_loss, correct, total = 0.0, 0, 0

        for video, mel, labels in tqdm(train_loader, desc=f'Train {epoch+1}/{num_epochs}', leave=False):
            video = video.to(device, non_blocking=True)
            mel = mel.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.no_grad():
                v_feat = visual_enc(video)
                a_feat = audio_enc(mel).mean(dim=1)

            logits = meta(v_feat, a_feat)
            loss = criterion(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        meta.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for video, mel, labels in tqdm(val_loader, desc='Val', leave=False):
                video = video.to(device, non_blocking=True)
                mel = mel.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                v_feat = visual_enc(video)
                a_feat = audio_enc(mel).mean(dim=1)
                logits = meta(v_feat, a_feat)
                val_loss += criterion(logits, labels).item() * labels.size(0)
                val_correct += (logits.argmax(dim=1) == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total
        print(f"Epoch {epoch+1:3d}: train={train_loss:.4f}/{train_acc:.4f}  "
              f"val={val_loss:.4f}/{val_acc:.4f}")

        with open(metrics_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch + 1, f'{train_loss:.6f}', f'{train_acc:.6f}',
                                    f'{val_loss:.6f}', f'{val_acc:.6f}'])

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(meta.state_dict(), best_path)
            print(f"  Saved best meta-learner (val_acc={best_val_acc:.4f})")

        if _abort:
            break

    torch.save(meta.state_dict(), os.path.join(save_dir, 'final_meta.pth'))
    print(f"Done. Best val_acc: {best_val_acc:.4f}")
