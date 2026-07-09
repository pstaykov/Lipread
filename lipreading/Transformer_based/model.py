"""GLipsNet — the Transformer-based lip-reading model and its shared building blocks.

Contains the 3D-conv + ResNet-18 visual frontend (``CNN3D``, reused by the
``mstcn_baseline`` so the two architectures are directly comparable), the
Transformer temporal backend (``GLipsNet``), and the checkpoint/metrics helpers
shared by both projects' training scripts.
"""
import os
import csv

import torch
import torch.nn as nn
import torchvision.models as models


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


class AttentivePool(nn.Module):
    """Learned attention pooling over the time axis.

    Plain mean-pooling averages every frame equally, diluting the few frames where
    the word is actually articulated into the near-silent ones. This scores each of
    the T tokens and returns their softmax-weighted sum, so discriminative frames
    dominate.
    """
    def __init__(self, d_model):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.Tanh(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x):  # x: (B, T, D)
        w = torch.softmax(self.score(x), dim=1)  # (B, T, 1)
        return (w * x).sum(dim=1)                 # (B, D)


class GLipsNet(nn.Module):
    # dropout is a plain float knob (no learned params) so raising it doesn't change the
    # state_dict — checkpoints trained at a different dropout still load cleanly.
    # pool='mean' (default) keeps the original temporal mean-pool so older checkpoints
    # load unchanged; pool='attn' adds an AttentivePool head (extra params).
    # use_stem=True (default) keeps the multi-scale temporal conv stem before the
    # Transformer; use_stem=False replaces it with identity, isolating how much the
    # local temporal conv stem contributes vs. the Transformer alone (ablation).
    def __init__(self, num_classes=500, dropout=0.2, pool='mean', use_stem=True):
        super().__init__()
        self.cnn = CNN3D()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Sequential(nn.Linear(FEAT_DIM, D_MODEL), nn.LayerNorm(D_MODEL))
        self.use_stem = use_stem
        self.ms_tcn = nn.Sequential(MSTemporalBlock(D_MODEL, dropout=dropout),
                                    MSTemporalBlock(D_MODEL, dropout=dropout)) \
            if use_stem else nn.Identity()
        self.pos_embed = nn.Parameter(torch.zeros(1, 250, D_MODEL))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=8, dim_feedforward=1024,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True,
        )
        # 2 layers (down from 4): at ~500 samples/class a deep 4-layer stack overfits
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        # dropout right before the classifier — the cheapest, highest-leverage guard
        # against the large train/val gap seen on the small (15-class) subset
        self.head_drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(D_MODEL, num_classes)
        self.attn_pool = AttentivePool(D_MODEL) if pool == 'attn' else None
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        x = self.cnn(x)
        B, T, C, H, W = x.size()
        x = self.avgpool(x.view(B * T, C, H, W)).flatten(1).view(B, T, FEAT_DIM)
        x = self.proj(x)
        x = self.ms_tcn(x)
        x = x + self.pos_embed[:, :T, :]
        x = self.transformer(x)
        x = self.attn_pool(x) if self.attn_pool is not None else x.mean(dim=1)
        return self.classifier(self.head_drop(x))


# --- checkpoint / metrics helpers shared by the Transformer and baseline trainers ---

def _model_state(model):
    return getattr(model, '_orig_mod', model).state_dict()


def _strip_orig_mod(state_dict):
    """Strip _orig_mod. prefix left by torch.compile when saving state_dict."""
    prefix = '_orig_mod.'
    if any(k.startswith(prefix) for k in state_dict):
        return {k[len(prefix):]: v for k, v in state_dict.items()}
    return state_dict


def load_transfer_weights(model, ckpt_path, device, skip_prefixes=('classifier',)):
    """Warm-start ``model`` from a checkpoint trained on a different label set.

    Loads every parameter whose name and shape match (the 3D-conv + ResNet
    frontend, the MS-TCN, positional embedding, Transformer, and attentive-pool
    head — the language-agnostic feature extractor), and skips the classifier so
    a model with a different ``num_classes`` warm-starts cleanly. This is the
    LRW->GLips transfer step from Schwiebert et al.: the source-language head is
    discarded, the learned lip-motion features are kept.

    Accepts either a raw EMA ``state_dict`` (what ``best/final_model.pth`` hold)
    or a full training checkpoint dict (``checkpoint_latest.pth``, key 'model').
    Returns (loaded_keys, skipped_keys).
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    state = _strip_orig_mod(state)

    model_state = model.state_dict()
    loaded, skipped = [], []
    for k, v in state.items():
        if any(k.startswith(p) for p in skip_prefixes):
            skipped.append(k)
        elif k in model_state and model_state[k].shape == v.shape:
            model_state[k] = v
            loaded.append(k)
        else:
            skipped.append(k)
    model.load_state_dict(model_state)
    print(f"Transfer init from {os.path.basename(ckpt_path)}: "
          f"loaded {len(loaded)} tensors, skipped {len(skipped)} "
          f"(classifier + any shape mismatches).")
    if skipped:
        print(f"  skipped: {skipped[:6]}{' ...' if len(skipped) > 6 else ''}")
    return loaded, skipped


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


def best_epoch_from_metrics(metrics_path):
    """Return (epoch, val_top1) of the row with the highest val_top1 in metrics.csv.

    best_model.pth stores only weights, so the matching epoch is recovered from
    the logged metrics. Returns (0, 0.0) if the file is missing or unreadable.
    """
    if not os.path.exists(metrics_path):
        return 0, 0.0
    best_ep, best_acc = 0, 0.0
    try:
        with open(metrics_path, newline='') as f:
            for row in csv.DictReader(f):
                acc = float(row['val_top1'])
                if acc >= best_acc:
                    best_acc, best_ep = acc, int(row['epoch'])
    except (KeyError, ValueError, OSError) as e:
        print(f"WARNING: could not parse {metrics_path} ({e})")
        return 0, 0.0
    return best_ep, best_acc


def prune_periodic_checkpoints(save_dir, keep=3):
    """Keep only the most recent `keep` epoch-N checkpoints."""
    import glob
    paths = sorted(glob.glob(os.path.join(save_dir, 'checkpoint_epoch_*.pth')))
    for old in paths[:-keep]:
        os.remove(old)
