import os
import sys
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'lipreading', 'Transformer_based'))
from train import GLipsNet, _strip_orig_mod, D_MODEL, FEAT_DIM, VideoAugment  # noqa: F401

try:
    import whisper as whisper_lib
except ImportError:
    raise ImportError("openai-whisper not found. Install with:  pip install openai-whisper")


class VisualFeatureExtractor(nn.Module):
    """GLipsNet backbone without the final classifier; outputs (B, D_MODEL)."""
    def __init__(self, glips_net):
        super().__init__()
        self.m = glips_net

    def forward(self, x):
        x = self.m.cnn(x)
        B, T, C, H, W = x.size()
        x = self.m.avgpool(x.view(B * T, C, H, W)).view(B, T, FEAT_DIM)
        x = self.m.proj(x)
        x = self.m.ms_tcn(x)
        x = x + self.m.pos_embed[:, :T, :]
        x = self.m.transformer(x)
        return x.mean(dim=1)  # (B, D_MODEL)


class MetaLearner(nn.Module):
    # whisper-small encoder output dim is 768
    def __init__(self, visual_dim=D_MODEL, audio_dim=768, hidden_dim=512, num_classes=500):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(visual_dim + audio_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, v_feat, a_feat):
        return self.net(torch.cat([v_feat, a_feat], dim=-1))


def load_visual_encoder(ckpt_path, device, num_classes=500):
    glips = GLipsNet(num_classes=num_classes)
    glips.load_state_dict(_strip_orig_mod(torch.load(ckpt_path, map_location='cpu')))
    enc = VisualFeatureExtractor(glips).to(device)
    enc.eval()
    for p in enc.parameters():
        p.requires_grad = False
    return enc


def load_audio_encoder(device):
    model = whisper_lib.load_model("small", device=device)
    enc = model.encoder
    enc.eval()
    for p in enc.parameters():
        p.requires_grad = False
    return enc
