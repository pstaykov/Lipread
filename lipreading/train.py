import os
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader

try:
    import torchvision.io as io
except RuntimeError:
    io = None


class GLipsFullClipDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, num_frames=25):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.num_frames = num_frames
        self.samples = []

        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        for cls_name in self.classes:
            split_folder = os.path.join(root_dir, cls_name, split)
            if not os.path.exists(split_folder):
                for alt in ['train', 'validation', 'val', 'test']:
                    alt_folder = os.path.join(root_dir, cls_name, alt)
                    if os.path.exists(alt_folder):
                        split_folder = alt_folder
                        break

            if os.path.exists(split_folder):
                for file in os.listdir(split_folder):
                    if file.endswith('.mp4'):
                        self.samples.append((os.path.join(split_folder, file), self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        video = None

        if io is not None and hasattr(io, 'read_video'):
            try:
                video, _, _ = io.read_video(video_path, pts_unit='sec', output_format='TCHW')
            except Exception:
                video = None

        if video is None:
            try:
                import imageio.v2 as imageio
            except Exception:
                try:
                    import imageio
                except Exception:
                    imageio = None

            if imageio is not None:
                frames = []
                try:
                    reader = imageio.get_reader(video_path, 'ffmpeg')
                    for frame in reader:
                        frames.append(frame)
                    reader.close()
                    video = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)
                except Exception:
                    video = None

        if video is None:
            try:
                import cv2
                cap = cv2.VideoCapture(video_path)
                frames = []
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                cap.release()
                if not frames:
                    raise RuntimeError(f"No frames read from {video_path}")
                video = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)
            except Exception as e:
                raise RuntimeError(f"Could not read video. Install imageio or opencv. Error: {e}")

        video = video.float() / 255.0
        T = video.size(0)

        if T > self.num_frames:
            indices = np.linspace(0, T - 1, num=self.num_frames).astype(int)
            video = video[indices]
        elif T < self.num_frames:
            pad_count = self.num_frames - T
            video = torch.cat([video, video[-1:].repeat(pad_count, 1, 1, 1)], dim=0)

        if self.transform:
            video = self.transform(video)

        return video.permute(1, 0, 2, 3), label


class VideoAugment:
    """Consistent spatial + temporal augmentation applied across all frames of a clip."""
    def __init__(self, crop_size=88, resize_size=96, is_train=True, time_mask_max=3):
        self.crop_size = crop_size
        self.resize_size = resize_size
        self.is_train = is_train
        self.time_mask_max = time_mask_max
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __call__(self, video):
        # video: (T, C, H, W), float32 in [0, 1]
        T = video.shape[0]

        if self.resize_size is not None:
            video = torch.stack([
                TF.resize(video[t], [self.resize_size, self.resize_size], antialias=True)
                for t in range(T)
            ])

        H, W = video.shape[2], video.shape[3]
        cs = self.crop_size
        if H >= cs and W >= cs:
            if self.is_train:
                top = torch.randint(0, H - cs + 1, (1,)).item()
                left = torch.randint(0, W - cs + 1, (1,)).item()
            else:
                top = (H - cs) // 2
                left = (W - cs) // 2
            video = video[:, :, top:top + cs, left:left + cs]

        if self.is_train:
            if torch.rand(1).item() < 0.5:
                video = torch.flip(video, dims=[3])

            if self.time_mask_max > 0 and T > 1:
                n = torch.randint(1, self.time_mask_max + 1, (1,)).item()
                start = torch.randint(0, max(1, T - n + 1), (1,)).item()
                video = video.clone()
                video[start:start + n] = 0.0

        video = (video - self.mean) / self.std
        return video


class Frontend3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )

    def forward(self, x):
        return self.layers(x)


FEAT_DIM = 512


class ResNet2DBackend(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.layers = nn.Sequential(*list(resnet.children())[4:8])

    def forward(self, x):
        return self.layers(x)


class CNN3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.frontend = Frontend3D()
        self.resnet = ResNet2DBackend()

    def forward(self, x):
        B, C, T, H, W = x.size()
        x = self.frontend(x)
        x = x.transpose(1, 2).contiguous()
        x = x.view(-1, 64, x.size(3), x.size(4))
        x = self.resnet(x)
        return x.view(B, T, FEAT_DIM, x.size(2), x.size(3))


D_MODEL = 256


class MSTemporalBlock(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        # three parallel depthwise branches capture short (k=3), mid (k=5), and broad (k=7) temporal patterns
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(d_model, d_model, k, padding=k // 2, groups=d_model),
                nn.Conv1d(d_model, d_model, 1),
                nn.BatchNorm1d(d_model),
                nn.GELU(),
            )
            for k in (3, 5, 7)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        xt = x.transpose(1, 2)
        out = sum(b(xt) for b in self.branches) / 3.0
        return self.norm(x + self.drop(out.transpose(1, 2)))


class GLipsNet(nn.Module):
    def __init__(self, num_classes=500):
        super().__init__()
        self.cnn = CNN3D()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Sequential(nn.Linear(FEAT_DIM, D_MODEL), nn.LayerNorm(D_MODEL))
        self.ms_tcn = nn.Sequential(MSTemporalBlock(D_MODEL), MSTemporalBlock(D_MODEL))
        self.pos_embed = nn.Parameter(torch.zeros(1, 250, D_MODEL))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=8, dim_feedforward=1024,
            dropout=0.1, activation='gelu', batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.classifier = nn.Linear(D_MODEL, num_classes)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        x = self.cnn(x)
        B, T, C, H, W = x.size()
        x = self.avgpool(x.view(B * T, C, H, W)).view(B, T, FEAT_DIM)
        x = self.proj(x)
        x = self.ms_tcn(x)
        x = x + self.pos_embed[:, :T, :]
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.classifier(x)


def _model_state(model):
    return getattr(model, '_orig_mod', model).state_dict()


def _strip_orig_mod(state_dict):
    """Strip _orig_mod. prefix left by torch.compile when saving state_dict."""
    prefix = '_orig_mod.'
    if any(k.startswith(prefix) for k in state_dict):
        return {k[len(prefix):]: v for k, v in state_dict.items()}
    return state_dict


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, best_val_acc):
    torch.save({
        'epoch': epoch,
        'model': _model_state(model),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'scaler': scaler.state_dict(),
        'best_val_acc': best_val_acc,
    }, path)


def load_checkpoint(path, model, optimizer, scheduler, scaler, device):
    ckpt = torch.load(path, map_location=device)
    getattr(model, '_orig_mod', model).load_state_dict(_strip_orig_mod(ckpt['model']))
    optimizer.load_state_dict(ckpt['optimizer'])
    scheduler.load_state_dict(ckpt['scheduler'])
    scaler.load_state_dict(ckpt['scaler'])
    return ckpt['epoch'], ckpt['best_val_acc']


def prune_periodic_checkpoints(save_dir, keep=3):
    """Keep only the most recent `keep` epoch-N checkpoints."""
    import glob
    paths = sorted(glob.glob(os.path.join(save_dir, 'checkpoint_epoch_*.pth')))
    for old in paths[:-keep]:
        os.remove(old)


if __name__ == '__main__':
    import signal
    from tqdm import tqdm

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

    CHECKPOINT_EVERY = 5   # save a named epoch-N checkpoint every N epochs

    num_workers = min(6, os.cpu_count() or 0)
    prefetch = 2 if num_workers > 0 else None
    root_dir = './GLips/lipread_files'
    train_transform = VideoAugment(crop_size=88, resize_size=96, is_train=True)
    val_transform = VideoAugment(crop_size=88, resize_size=96, is_train=False)
    train_dataset = GLipsFullClipDataset(root_dir, split='train', num_frames=25, transform=train_transform)
    val_dataset = GLipsFullClipDataset(root_dir, split='validation', num_frames=25, transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=num_workers,
                              pin_memory=True, persistent_workers=num_workers > 0,
                              prefetch_factor=prefetch, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=num_workers,
                            pin_memory=True, persistent_workers=num_workers > 0,
                            prefetch_factor=prefetch)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == 'cuda', f"CUDA not available — got device '{device}'"
    print(f"Using device: {torch.cuda.get_device_name(device)}")
    model = GLipsNet()
    model.to(device)

    # Differential LR: pretrained ResNet gets 10x lower rate than fresh components
    backbone_ids = {id(p) for p in model.cnn.resnet.parameters()}
    param_groups = [
        {'params': [p for p in model.parameters() if id(p) not in backbone_ids], 'lr': 1e-3},
        {'params': list(model.cnn.resnet.parameters()), 'lr': 1e-4},
    ]

    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    # bfloat16 has float32-level dynamic range — prevents NaN overflow in transformer attention
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=(amp_dtype == torch.float16))
    print(f"AMP dtype: {amp_dtype}")

    num_epochs = 50
    warmup_epochs = 5
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs - warmup_epochs, eta_min=1e-6)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs])

    save_dir = './checkpoints'
    os.makedirs(save_dir, exist_ok=True)
    latest_path = os.path.join(save_dir, 'checkpoint_latest.pth')
    best_path = os.path.join(save_dir, 'best_model.pth')
    metrics_path = os.path.join(save_dir, 'metrics.csv')

    start_epoch = 0
    best_val_acc = 0.0

    if os.path.exists(latest_path):
        try:
            start_epoch, best_val_acc = load_checkpoint(latest_path, model, optimizer, scheduler, scaler, device)
            print(f"Resumed from {latest_path} (epoch {start_epoch}, best_val_acc={best_val_acc:.4f})")
        except (RuntimeError, KeyError) as e:
            print(f"WARNING: {latest_path} incompatible with current architecture — starting fresh. ({e})")
    elif os.path.exists(best_path):
        try:
            # legacy: old runs only saved weights
            getattr(model, '_orig_mod', model).load_state_dict(
                _strip_orig_mod(torch.load(best_path, map_location=device)))
            print(f"Loaded weights from {best_path} (no optimizer state — starting fresh optimizer)")
        except (RuntimeError, KeyError) as e:
            print(f"WARNING: {best_path} incompatible with current architecture — starting fresh. ({e})")

    if hasattr(torch, 'compile') and os.name != 'nt':
        model = torch.compile(model)

    # Append to existing metrics or create fresh header
    if start_epoch == 0 or not os.path.exists(metrics_path):
        with open(metrics_path, 'w', newline='') as f:
            csv.writer(f).writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_top1', 'val_top5', 'lr'])

    # Allow Ctrl+C to save a checkpoint and exit cleanly
    _abort = False
    def _handle_sigint(sig, frame):
        global _abort
        print("\nInterrupt received — will save checkpoint after this epoch.")
        _abort = True
    signal.signal(signal.SIGINT, _handle_sigint)

    epoch_bar = tqdm(range(start_epoch, num_epochs), desc='Epochs', unit='epoch')
    for epoch in epoch_bar:
        model.train()
        running_loss = 0.0
        correct_train = 0
        num_samples = 0

        train_bar = tqdm(train_loader, desc=f'Train {epoch+1}/{num_epochs}', leave=False, unit='batch')
        for data, target in train_bar:
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
            train_bar.set_postfix(loss=f'{loss.item():.4f}')

        train_loss = running_loss / num_samples if num_samples > 0 else 0.0
        train_acc = correct_train / num_samples if num_samples > 0 else 0.0

        torch.cuda.empty_cache()
        model.eval()
        correct1 = 0
        correct5 = 0
        running_val_loss = 0.0
        total = 0
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc='Val', leave=False, unit='batch'):
                data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
                with torch.amp.autocast('cuda', dtype=amp_dtype):
                    logits = model(data)
                    running_val_loss += criterion(logits, target).item() * data.size(0)
                correct1 += (logits.argmax(dim=1) == target).sum().item()
                correct5 += (logits.topk(5, dim=1).indices == target.unsqueeze(1)).any(dim=1).sum().item()
                total += target.size(0)

        val_loss = running_val_loss / total if total > 0 else 0.0
        val_acc = correct1 / total if total > 0 else 0.0
        val_acc5 = correct5 / total if total > 0 else 0.0
        lr = scheduler.get_last_lr()[0]
        scheduler.step()
        epoch_bar.set_postfix(loss=f'{train_loss:.4f}', train_acc=f'{train_acc:.4f}',
                               top1=f'{val_acc:.4f}', top5=f'{val_acc5:.4f}', lr=f'{lr:.5f}')

        with open(metrics_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch + 1, f'{train_loss:.6f}', f'{train_acc:.6f}',
                                    f'{val_loss:.6f}', f'{val_acc:.6f}', f'{val_acc5:.6f}', f'{lr:.8f}'])

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(_model_state(model), best_path)
            tqdm.write(f"Saved best model (top1={best_val_acc:.4f}, top5={val_acc5:.4f})")

        # Always overwrite latest so any abort can resume from here
        save_checkpoint(latest_path, model, optimizer, scheduler, scaler, epoch + 1, best_val_acc)

        # Periodic named checkpoint every CHECKPOINT_EVERY epochs
        if (epoch + 1) % CHECKPOINT_EVERY == 0:
            periodic_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1:04d}.pth')
            save_checkpoint(periodic_path, model, optimizer, scheduler, scaler, epoch + 1, best_val_acc)
            prune_periodic_checkpoints(save_dir, keep=3)
            tqdm.write(f"Saved periodic checkpoint at epoch {epoch+1}")

        if _abort:
            tqdm.write("Abort flag set — stopping training. Resume with the same command.")
            break

    torch.save(_model_state(model), os.path.join(save_dir, 'final_model.pth'))
    print(f"Training finished. Best top-1: {best_val_acc:.4f}. Metrics saved to {metrics_path}")
