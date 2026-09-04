"""Fusion heads (late/concat/cross_attn/joint_tf) sharing the same (v_tok, a_tok) inputs so only the fusion mechanism differs."""
import torch
import torch.nn as nn


def _mask(B, p, training, device, dtype):
    """Per-sample keep multiplier for modality dropout; None when inactive."""
    if not training or p <= 0:
        return None
    return (torch.rand(B, 1, device=device) >= p).to(dtype)


def _keep(self, B, device, dtype, drop_audio):
    """Audio keep multiplier: all-zero when drop_audio forces visual-only (eval),
    else the training-time modality-dropout mask (None when inactive)."""
    if drop_audio:
        return torch.zeros(B, 1, device=device, dtype=dtype)
    return _mask(B, self.audio_dropout, self.training, device, dtype)


class LateFusion(nn.Module):
    def __init__(self, num_classes, d_v=256, d_a=512, dropout=0.3, audio_dropout=0.2):
        super().__init__()
        self.audio_dropout = audio_dropout
        self.v_head = nn.Sequential(nn.LayerNorm(d_v), nn.Dropout(dropout), nn.Linear(d_v, num_classes))
        self.a_head = nn.Sequential(nn.LayerNorm(d_a), nn.Dropout(dropout), nn.Linear(d_a, num_classes))

    def forward(self, v_tok, a_tok, drop_audio=False):
        v = v_tok.mean(1)
        a = a_tok.mean(1)
        lv = self.v_head(v)
        la = self.a_head(a)
        keep = _keep(self, v.size(0), v.device, v.dtype, drop_audio)
        if keep is not None:
            la = la * keep
        return lv + la


class ConcatFusion(nn.Module):
    def __init__(self, num_classes, d_v=256, d_a=512, hidden=768, dropout=0.3, audio_dropout=0.2):
        super().__init__()
        self.audio_dropout = audio_dropout
        self.vn = nn.LayerNorm(d_v)
        self.an = nn.LayerNorm(d_a)
        self.mlp = nn.Sequential(
            nn.Linear(d_v + d_a, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, v_tok, a_tok, drop_audio=False):
        v = self.vn(v_tok.mean(1))
        a = self.an(a_tok.mean(1))
        keep = _keep(self, v.size(0), v.device, v.dtype, drop_audio)
        if keep is not None:
            a = a * keep
        return self.mlp(torch.cat([v, a], dim=-1))


class CrossAttnFusion(nn.Module):
    """Visual tokens attend over audio via a gated residual, then pool + classify."""
    def __init__(self, num_classes, d_v=256, d_a=512, n_heads=8, dropout=0.3, audio_dropout=0.2):
        super().__init__()
        self.audio_dropout = audio_dropout
        self.audio_proj = nn.Linear(d_a, d_v)
        self.attn = nn.MultiheadAttention(d_v, n_heads, batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(d_v)
        self.gate = nn.Parameter(torch.zeros(1))
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_v, num_classes))

    def forward(self, v_tok, a_tok, drop_audio=False):
        a = self.audio_proj(a_tok)
        attended, _ = self.attn(v_tok, a, a)
        contrib = self.gate.tanh() * attended
        keep = _keep(self, v_tok.size(0), v_tok.device, v_tok.dtype, drop_audio)
        if keep is not None:
            contrib = contrib * keep.unsqueeze(1)
        fused = self.norm(v_tok + contrib)
        return self.head(fused.mean(1))


class JointTransformerFusion(nn.Module):
    """AV-HuBERT-style: concat visual+audio tokens with modality embeddings, run a shared Transformer, classify the CLS token."""
    def __init__(self, num_classes, d_v=256, d_a=512, d_model=256, n_layers=3,
                 n_heads=8, dropout=0.3, audio_dropout=0.2):
        super().__init__()
        self.audio_dropout = audio_dropout
        self.v_proj = nn.Linear(d_v, d_model)
        self.a_proj = nn.Linear(d_a, d_model)
        self.type_emb = nn.Parameter(torch.zeros(2, d_model))   # 0=visual, 1=audio
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=4 * d_model,
                                           dropout=dropout, activation='gelu',
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(dropout),
                                  nn.Linear(d_model, num_classes))
        nn.init.trunc_normal_(self.type_emb, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self, v_tok, a_tok, drop_audio=False):
        B = v_tok.size(0)
        v = self.v_proj(v_tok) + self.type_emb[0]
        a = self.a_proj(a_tok) + self.type_emb[1]
        # modality dropout via attention mask: drop all audio tokens for chosen samples
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, v, a], dim=1)
        key_padding = None
        keep = _keep(self, B, v.device, v.dtype, drop_audio)
        if keep is not None:
            drop = (keep.squeeze(1) == 0)                 # (B,) samples losing audio
            if drop.any():
                Ta = a_tok.size(1)
                key_padding = torch.zeros(B, x.size(1), dtype=torch.bool, device=v.device)
                key_padding[:, 1 + v_tok.size(1):] = drop.unsqueeze(1)  # mask audio span
        x = self.encoder(x, src_key_padding_mask=key_padding)
        return self.head(x[:, 0])                         # CLS token


HEADS = {
    'late': LateFusion,
    'concat': ConcatFusion,
    'cross_attn': CrossAttnFusion,
    'joint_tf': JointTransformerFusion,
}
