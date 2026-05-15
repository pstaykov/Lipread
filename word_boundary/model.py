import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class TCNBlock(nn.Module):
    def __init__(self, channels, dilation, dropout=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3,
                      padding=dilation, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=3,
                      padding=dilation, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.conv(x) + x


class WordBoundaryDetector(nn.Module):
    def __init__(self, tcn_channels=256, num_tcn_layers=5, dropout=0.1):
        super().__init__()

        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.spatial_encoder = nn.Sequential(*list(backbone.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.input_proj = nn.Linear(512, tcn_channels)
        self.tcn = nn.Sequential(*[
            TCNBlock(tcn_channels, dilation=2**i, dropout=dropout)
            for i in range(num_tcn_layers)
        ])
        self.classifier = nn.Linear(tcn_channels, 1)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.pool(self.spatial_encoder(x)).view(B, T, -1)
        x = self.input_proj(x).transpose(1, 2)
        x = self.tcn(x).transpose(1, 2)
        return self.classifier(x).squeeze(-1)