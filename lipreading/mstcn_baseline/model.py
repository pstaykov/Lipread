"""
Traditional lip-reading baseline: 3D-Conv frontend -> 2D ResNet-18 -> MS-TCN.
"""
import os
import importlib.util

import torch
import torch.nn as nn

# Reuse the shared visual frontend, feature dim, and checkpoint helpers from the Transformer model so the two architectures are directly comparable
_TM_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'Transformer_based', 'model.py')
_spec = importlib.util.spec_from_file_location('transformer_model', _TM_PATH)
transformer_model = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(transformer_model)

CNN3D, FEAT_DIM = transformer_model.CNN3D, transformer_model.FEAT_DIM
# Re-export shared training helpers so train.py/test.py can import them from here.
_model_state = transformer_model._model_state
_strip_orig_mod = transformer_model._strip_orig_mod
load_transfer_weights = transformer_model.load_transfer_weights
save_checkpoint = transformer_model.save_checkpoint
load_checkpoint = transformer_model.load_checkpoint
best_epoch_from_metrics = transformer_model.best_epoch_from_metrics
prune_periodic_checkpoints = transformer_model.prune_periodic_checkpoints


class TemporalBranch(nn.Module):
    """
    One dilated temporal-conv unit: Conv1d -> BatchNorm -> ReLU -> Dropout.
    """
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class MultibranchTCNBlock(nn.Module):
    """
    Residual block of two multi-branch temporal-conv layers.
    """
    def __init__(self, in_ch, out_ch, kernel_sizes, dilation, dropout):
        super().__init__()
        n_branches = len(kernel_sizes)
        assert out_ch % n_branches == 0, \
            f"out_ch ({out_ch}) must be divisible by #branches ({n_branches})"
        branch_ch = out_ch // n_branches

        self.branches1 = nn.ModuleList(
            [TemporalBranch(in_ch, branch_ch, k, dilation, dropout) for k in kernel_sizes])
        self.branches2 = nn.ModuleList(
            [TemporalBranch(out_ch, branch_ch, k, dilation, dropout) for k in kernel_sizes])

        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = torch.cat([b(x) for b in self.branches1], dim=1)
        out = torch.cat([b(out) for b in self.branches2], dim=1)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class MultiscaleMultibranchTCN(nn.Module):
    """
    Stack of multi-branch TCN blocks with exponentially growing dilation (1,2,4,...).
    """
    def __init__(self, in_ch, num_channels, kernel_sizes, dropout):
        super().__init__()
        layers = []
        for i, out_ch in enumerate(num_channels):
            layers.append(MultibranchTCNBlock(
                in_ch if i == 0 else num_channels[i - 1],
                out_ch, kernel_sizes, dilation=2 ** i, dropout=dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, T, C) -> (B, C, T) for Conv1d -> back to (B, T, C)
        x = self.network(x.transpose(1, 2))
        return x.transpose(1, 2)


class TCNLipNet(nn.Module):
    """
    3D-Conv -> ResNet-18 -> MS-TCN -> temporal mean-pool -> linear classifier.
    """
    def __init__(self, num_classes=500, width=384, num_layers=4,
                 kernel_sizes=(3, 5, 7), dropout=0.2):
        super().__init__()
        self.cnn = CNN3D()                              # shared visual frontend (-> 512-d/frame)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.tcn = MultiscaleMultibranchTCN(
            FEAT_DIM, [width] * num_layers, list(kernel_sizes), dropout)
        self.classifier = nn.Linear(width, num_classes)

    def forward(self, x):
        x = self.cnn(x)                                 # (B, T, 512, H', W')
        B, T, C, H, W = x.size()
        x = self.avgpool(x.view(B * T, C, H, W)).flatten(1).view(B, T, FEAT_DIM)
        x = self.tcn(x)                                 # (B, T, width)
        x = x.mean(dim=1)                               # temporal average pooling
        return self.classifier(x)
