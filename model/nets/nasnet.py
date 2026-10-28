"""NASNet-A Mobile (Zoph et al. 2018) in PyTorch, built layer for layer like keras.applications.nasnet so the
original Google ImageNet weights (Keras `NASNet-mobile.h5`) load by Keras layer name.

    weights: https://storage.googleapis.com/tensorflow/keras-applications/nasnet/NASNet-mobile.h5
             -> model/nets/weights/NASNet-mobile.h5 (gitignored)

Input: (N, 3, H, W) scaled to [-1, 1] (Keras 'tf' preprocessing). Output: (N, 1056) pooled features,
or ImageNet logits if built with imagenet_head=True.
"""
import os

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights', 'NASNet-mobile.h5')

# NASNetMobile() arguments in Keras
PENULTIMATE_FILTERS = 1056
NUM_BLOCKS = 4
STEM_FILTERS = 32
FILTER_MULT = 2
SKIP_REDUCTION = False
OUT_DIM = PENULTIMATE_FILTERS


def _correct_pad(x, k):
    """Keras imagenet_utils.correct_pad: TF 'same' padding for a stride-2 window, extra pixel at bottom/right."""
    h, w = x.shape[-2:]
    return F.pad(x, (k // 2 - (1 - w % 2), k // 2, k // 2 - (1 - h % 2), k // 2))


def _avg3_same(x):
    """Keras AveragePooling2D((3, 3), strides=1, padding='same'): padded cells are not counted."""
    return F.avg_pool2d(x, 3, 1, 1, count_include_pad=False)


class SeparableConv2d(nn.Module):
    """Keras SeparableConv2D (depth_multiplier=1, no bias): depthwise k×k then pointwise 1×1."""

    def __init__(self, cin, cout, k, stride):
        super().__init__()
        self.stride, self.k = stride, k
        self.depthwise = nn.Conv2d(cin, cin, k, stride, padding=0 if stride == 2 else k // 2, groups=cin, bias=False)
        self.pointwise = nn.Conv2d(cin, cout, 1, bias=False)

    def forward(self, x):
        if self.stride == 2:
            x = _correct_pad(x, self.k)
        return self.pointwise(self.depthwise(x))


class NASNetMobile(nn.Module):
    """All weighted layers live in `self.k`, keyed by their Keras names. They are created by one shape-tracing
    pass in __init__, so the cell code below stays a line-by-line mirror of keras/src/applications/nasnet.py."""

    def __init__(self, imagenet_head=False):
        super().__init__()
        self.k = nn.ModuleDict()
        self._building = True
        with torch.no_grad():
            self._trunk(torch.zeros(1, 3, 96, 96))
        self._building = False
        self.predictions = nn.Linear(OUT_DIM, 1000) if imagenet_head else None

    # -- layers, created on the build pass --------------------------------------------------------
    def _layer(self, name, make):
        if name not in self.k:
            assert self._building, f'layer {name} missing after build'
            self.k[name] = make()
        return self.k[name]

    def _conv(self, name, x, cout, k=1, stride=1):
        return self._layer(name, lambda: nn.Conv2d(x.shape[1], cout, k, stride, bias=False))(x)

    def _bn(self, name, x):
        return self._layer(name, lambda: nn.BatchNorm2d(x.shape[1], eps=1e-3))(x)

    def _sep(self, name, x, cout, k, stride):
        return self._layer(name, lambda: SeparableConv2d(x.shape[1], cout, k, stride))(x)

    # -- Keras blocks ------------------------------------------------------------------------------
    def _separable_conv_block(self, ip, filters, k=3, stride=1, block_id=None):
        x = F.relu(ip)
        x = self._sep(f'separable_conv_1_{block_id}', x, filters, k, stride)
        x = self._bn(f'separable_conv_1_bn_{block_id}', x)
        x = F.relu(x)
        x = self._sep(f'separable_conv_2_{block_id}', x, filters, k, 1)
        return self._bn(f'separable_conv_2_bn_{block_id}', x)

    def _adjust_block(self, p, ip, filters, block_id):
        if p is None:
            return ip
        if p.shape[-1] != ip.shape[-1]:
            p = F.relu(p)
            p1 = self._conv(f'adjust_conv_1_{block_id}', p[:, :, ::2, ::2], filters // 2)
            p2 = F.pad(p, (0, 1, 0, 1))[:, :, 1:, 1:][:, :, ::2, ::2]  # ZeroPadding((0,1),(0,1)) + Cropping((1,0),(1,0))
            p2 = self._conv(f'adjust_conv_2_{block_id}', p2, filters // 2)
            return self._bn(f'adjust_bn_{block_id}', torch.cat([p1, p2], 1))
        if p.shape[1] != filters:
            p = self._conv(f'adjust_conv_projection_{block_id}', F.relu(p), filters)
            return self._bn(f'adjust_bn_{block_id}', p)
        return p

    def _normal_a_cell(self, ip, p, filters, block_id):
        p = self._adjust_block(p, ip, filters, block_id)
        h = self._bn(f'normal_bn_1_{block_id}', self._conv(f'normal_conv_1_{block_id}', F.relu(ip), filters))
        x1 = (self._separable_conv_block(h, filters, 5, block_id=f'normal_left1_{block_id}')
              + self._separable_conv_block(p, filters, block_id=f'normal_right1_{block_id}'))
        x2 = (self._separable_conv_block(p, filters, 5, block_id=f'normal_left2_{block_id}')
              + self._separable_conv_block(p, filters, 3, block_id=f'normal_right2_{block_id}'))
        x3 = _avg3_same(h) + p
        x4 = _avg3_same(p) + _avg3_same(p)
        x5 = self._separable_conv_block(h, filters, block_id=f'normal_left5_{block_id}') + h
        return torch.cat([p, x1, x2, x3, x4, x5], 1), ip

    def _reduction_a_cell(self, ip, p, filters, block_id):
        p = self._adjust_block(p, ip, filters, block_id)
        h = self._bn(f'reduction_bn_1_{block_id}', self._conv(f'reduction_conv_1_{block_id}', F.relu(ip), filters))
        h3 = _correct_pad(h, 3)
        x1 = (self._separable_conv_block(h, filters, 5, 2, block_id=f'reduction_left1_{block_id}')
              + self._separable_conv_block(p, filters, 7, 2, block_id=f'reduction_right1_{block_id}'))
        x2 = F.max_pool2d(h3, 3, 2) + self._separable_conv_block(p, filters, 7, 2, block_id=f'reduction_right2_{block_id}')
        x3 = F.avg_pool2d(h3, 3, 2) + self._separable_conv_block(p, filters, 5, 2, block_id=f'reduction_right3_{block_id}')
        x4 = _avg3_same(x1) + x2
        x5 = (self._separable_conv_block(x1, filters, 3, block_id=f'reduction_left4_{block_id}')
              + F.max_pool2d(h3, 3, 2))
        return torch.cat([x2, x3, x4, x5], 1), ip

    def _trunk(self, x):
        filters = PENULTIMATE_FILTERS // 24
        x = self._bn('stem_bn1', self._conv('stem_conv1', x, STEM_FILTERS, 3, 2))
        x, p = self._reduction_a_cell(x, None, filters // FILTER_MULT ** 2, 'stem_1')
        x, p = self._reduction_a_cell(x, p, filters // FILTER_MULT, 'stem_2')
        for i in range(NUM_BLOCKS):
            x, p = self._normal_a_cell(x, p, filters, f'{i}')
        x, p0 = self._reduction_a_cell(x, p, filters * FILTER_MULT, f'reduce_{NUM_BLOCKS}')
        p = p if SKIP_REDUCTION else p0
        for i in range(NUM_BLOCKS):
            x, p = self._normal_a_cell(x, p, filters * FILTER_MULT, f'{NUM_BLOCKS + i + 1}')
        x, p0 = self._reduction_a_cell(x, p, filters * FILTER_MULT ** 2, f'reduce_{2 * NUM_BLOCKS}')
        p = p if SKIP_REDUCTION else p0
        for i in range(NUM_BLOCKS):
            x, p = self._normal_a_cell(x, p, filters * FILTER_MULT ** 2, f'{2 * NUM_BLOCKS + i + 1}')
        return F.relu(x)

    def forward(self, x):
        x = self._trunk(x).mean((2, 3))
        return self.predictions(x) if self.predictions is not None else x


def load_keras_weights(model, path=DEFAULT_WEIGHTS):
    """Copy every weighted Keras layer into the matching module; fails loudly on any name or shape mismatch."""
    loaded = set()
    with h5py.File(path, 'r') as f:
        g = f['model_weights'] if 'model_weights' in f else f
        for name in (n.decode() if isinstance(n, bytes) else n for n in g.attrs['layer_names']):
            wnames = [w.decode() if isinstance(w, bytes) else w for w in g[name].attrs['weight_names']]
            if not wnames:
                continue
            w = {wn.split('/')[-1].split(':')[0]: torch.from_numpy(np.array(g[name][wn])) for wn in wnames}
            if name == 'predictions':
                if model.predictions is not None:
                    model.predictions.weight.data.copy_(w['kernel'].T)
                    model.predictions.bias.data.copy_(w['bias'])
                continue
            m = model.k[name]
            if isinstance(m, nn.BatchNorm2d):
                for ours, theirs in [('weight', 'gamma'), ('bias', 'beta'),
                                     ('running_mean', 'moving_mean'), ('running_var', 'moving_variance')]:
                    getattr(m, ours).data.copy_(w[theirs])
            elif isinstance(m, SeparableConv2d):
                m.depthwise.weight.data.copy_(w['depthwise_kernel'].permute(2, 3, 0, 1))   # (k,k,in,1) -> (in,1,k,k)
                m.pointwise.weight.data.copy_(w['pointwise_kernel'].permute(3, 2, 0, 1))   # (1,1,in,out) -> (out,in,1,1)
            else:
                m.weight.data.copy_(w['kernel'].permute(3, 2, 0, 1))                      # (k,k,in,out) -> (out,in,k,k)
            loaded.add(name)
    missing = set(model.k.keys()) - loaded
    assert not missing, f'{len(missing)} layers without weights, e.g. {sorted(missing)[:5]}'
    return model


def nasnet_mobile(pretrained=True, imagenet_head=False):
    model = NASNetMobile(imagenet_head=imagenet_head)
    return load_keras_weights(model) if pretrained else model
