"""DC-TCN back-end, ported unchanged from mpc001/Lipreading_using_Temporal_Convolutional_Networks
(lipreading/models/densetcn.py, se_module.py, swish.py), with the LRW config of configs/lrw_resnet18_dctcn.json."""
from collections import OrderedDict

import torch
import torch.nn as nn

# configs/lrw_resnet18_dctcn.json
DCTCN = dict(block_config=[3, 3, 3, 3], growth_rate_set=[384, 384, 384, 384], reduced_size=512,
             kernel_size_set=[3, 5, 7], dilation_size_set=[1, 2, 5], squeeze_excitation=True,
             dropout=0.2, relu_type='swish')


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


def _act(relu_type, n):
    return {'relu': lambda: nn.ReLU(), 'prelu': lambda: nn.PReLU(num_parameters=n), 'swish': Swish}[relu_type]()


class SELayer(nn.Module):
    def __init__(self, channel, reduction=2):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(nn.Linear(channel, channel // reduction, bias=False), Swish(),
                                nn.Linear(channel // reduction, channel, bias=False), nn.Sigmoid())

    def forward(self, x):
        b, c, _ = x.size()
        return x * self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size   # symmetric chomp, as in the LRW model

    def forward(self, x):
        return x if self.chomp_size == 0 else x[:, :, self.chomp_size // 2:-self.chomp_size // 2].contiguous()


class TemporalConvLayer(nn.Sequential):
    def __init__(self, n_in, n_out, k, dilation, relu_type):
        pad = (k - 1) * dilation
        super().__init__(nn.Conv1d(n_in, n_out, k, padding=pad, dilation=dilation),
                         nn.BatchNorm1d(n_out), Chomp1d(pad), _act(relu_type, n_out))


class _ConvBatchChompRelu(nn.Module):
    def __init__(self, n_in, n_out, kernel_size_set, dilation, dropout, relu_type, se):
        super().__init__()
        n_branch = n_out // len(kernel_size_set)
        assert n_out % len(kernel_size_set) == 0
        self.se = nn.ModuleList([SELayer(n_in, reduction=16) for _ in kernel_size_set]) if se else None
        self.cbcr0 = nn.ModuleList([TemporalConvLayer(n_in, n_branch, k, dilation, relu_type) for k in kernel_size_set])
        self.cbcr1 = nn.ModuleList([TemporalConvLayer(n_out, n_branch, k, dilation, relu_type) for k in kernel_size_set])
        self.dropout0, self.dropout1 = nn.Dropout(dropout), nn.Dropout(dropout)
        self.downsample = nn.Conv1d(n_in, n_out, 1) if n_in != n_out else None
        self.relu_final = _act(relu_type, n_out)

    def forward(self, prev_features):
        x = torch.cat(prev_features, 1)
        out0 = torch.cat([conv(self.se[i](x) if self.se is not None else x) for i, conv in enumerate(self.cbcr0)], 1)
        out0 = self.dropout0(out0)
        out1 = self.dropout1(torch.cat([conv(out0) for conv in self.cbcr1], 1))
        res = x if self.downsample is None else self.downsample(x)
        return self.relu_final(out1 + res)


class _DenseBlock(nn.ModuleDict):
    def __init__(self, num_layers, n_in, growth_rate, kernel_size_set, dilation_size_set, dropout, relu_type, se):
        super().__init__()
        for i in range(num_layers):
            self.add_module(f'denselayer{i + 1}', _ConvBatchChompRelu(
                n_in + i * growth_rate, growth_rate, kernel_size_set,
                dilation_size_set[i % len(dilation_size_set)], dropout, relu_type, se))

    def forward(self, x):
        features = [x]
        for layer in self.values():
            features.append(layer(features))
        return torch.cat(features, 1)


def _transition(n_in, n_out, relu_type):
    return nn.Sequential(nn.Conv1d(n_in, n_out, 1, bias=False), nn.BatchNorm1d(n_out), _act(relu_type, n_out))


class DenseTemporalConvNet(nn.Module):
    """(B, input_size, T) -> (B, out_dim, T)."""

    def __init__(self, input_size, block_config, growth_rate_set, reduced_size, kernel_size_set, dilation_size_set,
                 dropout, relu_type, squeeze_excitation):
        super().__init__()
        layers = OrderedDict(transition0=_transition(input_size, reduced_size, 'prelu'))
        n = reduced_size
        for i, num_layers in enumerate(block_config):
            layers[f'denseblock{i + 1}'] = _DenseBlock(num_layers, n, growth_rate_set[i], kernel_size_set,
                                                       dilation_size_set, dropout, relu_type, squeeze_excitation)
            n += num_layers * growth_rate_set[i]
            if i != len(block_config) - 1:
                layers[f'transition{i + 1}'] = _transition(n, reduced_size, relu_type)
                n = reduced_size
        layers['norm5'] = nn.BatchNorm1d(n)
        self.features = nn.Sequential(layers)
        self.out_dim = n

    def forward(self, x):
        return self.features(x)

