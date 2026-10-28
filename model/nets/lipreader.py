"""Word classifier for every input variant: per-frame encoder -> DC-TCN (Ma et al.) -> mean over time -> linear.

Encoders: NASNet-Mobile on images (V0-V3), a coordinate MLP on landmarks (V4, V5), both concatenated (V6).
"""
import torch
import torch.nn as nn

from ..data.variants import coord_dim, uses_video
from .dctcn import DCTCN, DenseTemporalConvNet
from .nasnet import OUT_DIM as NASNET_DIM
from .nasnet import nasnet_mobile

COORD_DIM_ALONE = NASNET_DIM   # V4/V5: same width as the NASNet embedding
COORD_DIM_FUSED = 256          # V6: side branch next to the 1056-d NASNet embedding


# encoders + full model 
class CoordEncoder(nn.Module):
    """Per-frame MLP over flattened landmark coordinates: (B, T, F) -> (B, T, out_dim). The input BN
    standardizes the tiny Procrustes-scaled values."""

    def __init__(self, in_dim, out_dim, hidden=512):
        super().__init__()
        self.net = nn.Sequential(nn.BatchNorm1d(in_dim),
                                 nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
                                 nn.Linear(hidden, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU())

    def forward(self, x):
        b, t, f = x.shape
        return self.net(x.reshape(b * t, f)).view(b, t, -1)


class LipreadingModel(nn.Module):
    def __init__(self, variant, num_classes, pretrained=True):
        super().__init__()
        self.variant = variant
        self.video_encoder = nasnet_mobile(pretrained=pretrained) if uses_video(variant) else None
        fused = self.video_encoder is not None
        self.coord_encoder = (CoordEncoder(coord_dim(variant), COORD_DIM_FUSED if fused else COORD_DIM_ALONE)
                              if coord_dim(variant) else None)
        in_dim = ((NASNET_DIM if fused else 0)
                  + (COORD_DIM_FUSED if fused else COORD_DIM_ALONE) * (self.coord_encoder is not None))
        self.tcn = DenseTemporalConvNet(in_dim, **DCTCN)
        self.head = nn.Linear(self.tcn.out_dim, num_classes)

    def forward(self, video=None, coords=None, lengths=None):
        """video (B, T, 3, H, W) in [-1, 1]; coords (B, T, F); lengths (B,) valid frames, None = all."""
        feats = []
        if self.video_encoder is not None:
            b, t = video.shape[:2]
            feats.append(self.video_encoder(video.flatten(0, 1)).view(b, t, -1))
        if self.coord_encoder is not None:
            feats.append(self.coord_encoder(coords))
        x = self.tcn(torch.cat(feats, -1).transpose(1, 2))           # (B, C, T)
        if lengths is None:
            x = x.mean(2)
        else:                                                        # mpc001's _average_batch
            x = torch.stack([x[i, :, :n].mean(1) for i, n in enumerate(lengths)])
        return self.head(x)
