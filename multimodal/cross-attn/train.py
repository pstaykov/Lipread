"""
Multimodal lip reading: visual GLipsNet + Whisper audio cross-attention.

Loads pretrained visual weights from ../lipreading/checkpoints/best_model.pth
and fine-tunes with audio features extracted by the Whisper large encoder.

Install:  pip install openai-whisper
"""
import os
import sys
import csv
import signal
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'lipreading'))
from dataset import GLipsFullClipDataset  # noqa: F401

try:
    import whisper
except ImportError:
    raise ImportError("openai-whisper not found. Install with:  pip install openai-whisper")


def _ensure_ffmpeg():
    """Add imageio_ffmpeg's bundled binary to PATH if ffmpeg isn't already available."""
    import shutil
    if shutil.which('ffmpeg'):
        return
    try:
        import imageio_ffmpeg
        ffmpeg_dir = os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        # whisper calls 'ffmpeg' by name; symlink/rename won't work on Windows,
        # so we monkey-patch whisper.audio to use the full path instead.
        import subprocess, numpy as np
        import whisper as _whisper
        import whisper.audio as _wa

        def _patched_load_audio(file, sr=_wa.SAMPLE_RATE):
            cmd = [ffmpeg_exe, "-nostdin", "-threads", "0", "-i", file,
                   "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le",
                   "-ar", str(sr), "-"]
            out = subprocess.run(cmd, capture_output=True, check=True).stdout
            return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

        # patch both the submodule reference and the package-level re-export
        _wa.load_audio = _patched_load_audio
        _whisper.load_audio = _patched_load_audio
        print(f"ffmpeg not on PATH — using bundled: {ffmpeg_exe}")
    except ImportError:
        raise RuntimeError(
            "ffmpeg not found. Install it or run:  pip install imageio[ffmpeg]"
        )

_ensure_ffmpeg()


class VideoAugment:
    def __init__(self, crop_size=88, resize_size=96, is_train=True, time_mask_max=3):
        self.crop_size = crop_size
        self.resize_size = resize_size
        self.is_train = is_train
        self.time_mask_max = time_mask_max
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __call__(self, video):
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


# ---------------------------------------------------------------------------
# Visual backbone
# ---------------------------------------------------------------------------

FEAT_DIM = 512
D_MODEL = 256


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


class MSTemporalBlock(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        # three parallel depthwise branches: short (k=3), mid (k=5), broad (k=7)
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


# ---------------------------------------------------------------------------
# Multimodal model
# ---------------------------------------------------------------------------

WHISPER_MODEL_NAME = 'base'
AUDIO_DIM = 512    # Whisper base encoder hidden dim
AUDIO_T = 1500     # encoder time steps for 30-second padded audio


class CrossAttention(nn.Module):
    def __init__(self, d_model=D_MODEL, n_heads=8, audio_dim=AUDIO_DIM):
        super().__init__()
        self.audio_proj = nn.Linear(audio_dim, d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        # learned residual gate, starts closed so fusion is initially a no-op and
        # the random cross-attn can't corrupt the pretrained visual encoder
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, visual_tokens, audio_features):
        audio_proj = self.audio_proj(audio_features)
        attended, weights = self.cross_attn(
            query=visual_tokens,
            key=audio_proj,
            value=audio_proj,
        )
        return self.norm(visual_tokens + self.gate.tanh() * attended), weights


class GLipsNet(nn.Module):
    def __init__(self, num_classes=500, use_audio=False):
        super().__init__()
        self.use_audio = use_audio
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
        self.cross_attn = CrossAttention() if use_audio else None
        self.classifier = nn.Linear(D_MODEL, num_classes)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x, audio_features=None):
        x = self.cnn(x)
        B, T, C, H, W = x.size()
        x = self.avgpool(x.view(B * T, C, H, W)).flatten(1).view(B, T, FEAT_DIM)
        x = self.proj(x)
        x = self.ms_tcn(x)
        x = x + self.pos_embed[:, :T, :]
        x = self.transformer(x)
        if self.use_audio and audio_features is not None:
            x, _ = self.cross_attn(x, audio_features)
        x = x.mean(dim=1)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _model_state(model):
    return getattr(model, '_orig_mod', model).state_dict()


def _strip_orig_mod(state_dict):
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
    import glob
    paths = sorted(glob.glob(os.path.join(save_dir, 'checkpoint_epoch_*.pth')))
    for old in paths[:-keep]:
        os.remove(old)


def load_visual_weights(model, path, device):
    """Partial load from a visual-only checkpoint; new keys (cross_attn) stay random."""
    raw = torch.load(path, map_location=device, weights_only=True)
    state = _strip_orig_mod(raw if not isinstance(raw, dict) or 'model' not in raw
                            else raw['model'])
    current = model.state_dict()
    matched = {k: v for k, v in state.items()
               if k in current and current[k].shape == v.shape}
    missing = [k for k in current if k not in matched]
    model.load_state_dict(matched, strict=False)
    print(f"Loaded {len(matched)} visual tensors from {path}. "
          f"New (random-init) keys: {len(missing)}")


# ---------------------------------------------------------------------------
# Whisper feature extractor
# ---------------------------------------------------------------------------

class WhisperExtractor:
    def __init__(self, model_name=WHISPER_MODEL_NAME, device='cpu'):
        print(f"Loading Whisper {model_name} on {device} …")
        self.model = whisper.load_model(model_name, device=device)
        self.model.eval()
        self.device = device
        self.n_mels = self.model.dims.n_mels

    @torch.no_grad()
    def extract(self, path):
        """Return encoder features as a CPU tensor (AUDIO_T, AUDIO_DIM). Zeros if no audio."""
        try:
            audio = whisper.load_audio(path)
        except Exception:
            return torch.zeros(AUDIO_T, AUDIO_DIM)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio, n_mels=self.n_mels).to(self.device)
        return self.model.encoder(mel.unsqueeze(0)).squeeze(0).cpu()


# ---------------------------------------------------------------------------
# Multimodal dataset
# ---------------------------------------------------------------------------

class MultimodalGLipsDataset(GLipsFullClipDataset):
    def __init__(self, *args, extractor: WhisperExtractor = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.extractor = extractor

    def _audio_features(self, video_path):
        if self.extractor is None:
            return torch.zeros(AUDIO_T, AUDIO_DIM)
        return self.extractor.extract(video_path)

    def __getitem__(self, idx):
        video, label = super().__getitem__(idx)
        video_path, _ = self.samples[idx]
        audio = self._audio_features(video_path)
        return video, audio, label


def collate_fn(batch):
    videos, audios, labels = zip(*batch)
    return torch.stack(videos), torch.stack(audios), torch.tensor(labels)



# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

    CHECKPOINT_EVERY = 5
    VISUAL_CKPT = '../../lipreading/Transformer_based/checkpoints/best_model.pth'
    SAVE_DIR = './checkpoints'
    ROOT_DIR = '../../lipreading/GLips_mouth/lipread_files'
    NUM_FRAMES = 25
    NUM_EPOCHS = 100
    WARMUP_EPOCHS = 5

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assert device.type == 'cuda', f"CUDA not available — got '{device}'"
    print(f"Device: {torch.cuda.get_device_name(device)}")

    extractor = WhisperExtractor(WHISPER_MODEL_NAME, device=device)

    train_tf = VideoAugment(crop_size=88, resize_size=96, is_train=True)
    val_tf = VideoAugment(crop_size=88, resize_size=96, is_train=False)

    train_ds = MultimodalGLipsDataset(ROOT_DIR, split='train', num_frames=NUM_FRAMES,
                                      transform=train_tf, extractor=extractor)
    val_ds = MultimodalGLipsDataset(ROOT_DIR, split='validation', num_frames=NUM_FRAMES,
                                    transform=val_tf, extractor=extractor)

    # num_workers must be 0: Whisper runs on CUDA, which cannot be used in forked workers
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True,
                              num_workers=0, pin_memory=False,
                              drop_last=True,
                              collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False,
                            num_workers=0, pin_memory=False,
                            collate_fn=collate_fn)

    model = GLipsNet(num_classes=len(train_ds.classes), use_audio=True)

    if not os.path.exists(VISUAL_CKPT):
        raise FileNotFoundError(
            f"Visual checkpoint required but not found at {VISUAL_CKPT}. "
            "Train the visual-only model first and place its best checkpoint there."
        )
    load_visual_weights(model, VISUAL_CKPT, device)

    model.to(device)

    backbone_ids = {id(p) for p in model.cnn.resnet.parameters()}
    param_groups = [
        {'params': [p for p in model.parameters() if id(p) not in backbone_ids], 'lr': 1e-3},
        {'params': list(model.cnn.resnet.parameters()), 'lr': 1e-4},
    ]

    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=(amp_dtype == torch.float16))
    print(f"AMP dtype: {amp_dtype}")

    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_EPOCHS)
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS - WARMUP_EPOCHS, eta_min=1e-6)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[WARMUP_EPOCHS])

    os.makedirs(SAVE_DIR, exist_ok=True)
    latest_path = os.path.join(SAVE_DIR, 'checkpoint_latest.pth')
    best_path = os.path.join(SAVE_DIR, 'best_model.pth')
    metrics_path = os.path.join(SAVE_DIR, 'metrics.csv')

    start_epoch = 0
    best_val_acc = 0.0

    if os.path.exists(latest_path):
        try:
            ckpt = torch.load(latest_path, map_location=device)
            getattr(model, '_orig_mod', model).load_state_dict(
                _strip_orig_mod(ckpt['model']))
            start_epoch = ckpt['epoch']
            best_val_acc = ckpt.get('best_val_acc', 0.0)
            print(f"Resumed model weights from {latest_path} "
                  f"(epoch {start_epoch}, best={best_val_acc:.4f})")
            print("Optimizer/scheduler reset — LR schedule restarts from epoch 0.")
        except (RuntimeError, KeyError) as e:
            print(f"WARNING: {latest_path} incompatible — starting fresh. ({e})")

    if hasattr(torch, 'compile') and os.name != 'nt':
        model = torch.compile(model)

    if start_epoch == 0 or not os.path.exists(metrics_path):
        with open(metrics_path, 'w', newline='') as f:
            csv.writer(f).writerow(
                ['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_top1', 'val_top5', 'lr'])

    _abort = False

    def _handle_sigint(sig, frame):
        global _abort
        print("\nInterrupt — will save after this epoch.")
        _abort = True

    signal.signal(signal.SIGINT, _handle_sigint)

    recent: list[str] = []

    epoch_bar = tqdm(range(start_epoch, NUM_EPOCHS), desc='Epochs', unit='epoch',
                     dynamic_ncols=True)
    for epoch in epoch_bar:
        model.train()
        running_loss = correct_train = num_samples = 0

        train_bar = tqdm(train_loader, desc=f'Train {epoch+1}/{NUM_EPOCHS}',
                         leave=False, dynamic_ncols=True, unit='batch')
        for video, audio, target in train_bar:
            video = video.to(device, non_blocking=True)
            audio = audio.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

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
            train_bar.set_postfix({'loss': f'{loss.item():.4f}'}, refresh=False)

        train_loss = running_loss / num_samples if num_samples else 0.0
        train_acc = correct_train / num_samples if num_samples else 0.0

        torch.cuda.empty_cache()
        model.eval()
        correct1 = correct5 = running_val_loss = total = 0

        with torch.no_grad():
            val_bar = tqdm(val_loader, desc='Val', leave=False,
                           dynamic_ncols=True, unit='batch')
            for video, audio, target in val_bar:
                video = video.to(device, non_blocking=True)
                audio = audio.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
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

        recent.append(
            f"ep{epoch+1:02d} loss={train_loss:.3f} tr={train_acc:.3f} "
            f"top1={val_acc:.3f} top5={val_acc5:.3f}"
        )
        recent = recent[-5:]

        tqdm.write(
            f"[{epoch+1:>3}/{NUM_EPOCHS}] "
            f"loss={train_loss:.4f} train={train_acc:.4f} "
            f"top1={val_acc:.4f} top5={val_acc5:.4f} lr={lr:.5f}"
        )
        epoch_bar.set_description('  |  '.join(recent), refresh=True)

        with open(metrics_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch + 1, f'{train_loss:.6f}', f'{train_acc:.6f}',
                                    f'{val_loss:.6f}', f'{val_acc:.6f}', f'{val_acc5:.6f}',
                                    f'{lr:.8f}'])

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(_model_state(model), best_path)
            tqdm.write(f"  ** best model saved (top1={best_val_acc:.4f}, top5={val_acc5:.4f})")

        save_checkpoint(latest_path, model, optimizer, scheduler, scaler, epoch + 1, best_val_acc)

        if (epoch + 1) % CHECKPOINT_EVERY == 0:
            periodic = os.path.join(SAVE_DIR, f'checkpoint_epoch_{epoch+1:04d}.pth')
            save_checkpoint(periodic, model, optimizer, scheduler, scaler, epoch + 1, best_val_acc)
            prune_periodic_checkpoints(SAVE_DIR, keep=3)
            tqdm.write(f"  checkpoint saved: epoch {epoch+1}")

        if _abort:
            tqdm.write("Abort flag — stopping. Resume with the same command.")
            break

    torch.save(_model_state(model), os.path.join(SAVE_DIR, 'final_model.pth'))
    print(f"Done. Best top-1: {best_val_acc:.4f}. Metrics: {metrics_path}")
