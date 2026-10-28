"""Training augmentation as in BWKI's VideoAugment: one random draw per clip, the same for every frame.

- All variants: time masking, 1-2 blocks of 1-4 frames set to 0, the same frames in video and coordinates.
- Images: random affine (rotation ±10°, scale ±10 %, shift ±6 % of the crop, black fill) and horizontal flip; no flip
  when coordinates come along (V6), which it would contradict. RGB inputs (V0, V1, V6) also get brightness and
  contrast ±20 %.
- Coordinates: the same affine on the aligned x, y around the clip's mean point, z scaled along. No flip: it would
  need a left-right map of the FaceMesh points.
"""
import math

import torch
import torch.nn.functional as F

ROT, SCALE, SHIFT, LIGHT = 10.0, 0.1, 0.06, 0.2
MASKS, MASK_LEN = 2, 4
GREY = (0.299, 0.587, 0.114)


def _sym(n, mag, dev):
    return (torch.rand(n, device=dev) * 2 - 1) * mag


def _affine(n, dev):
    """Per-clip rotation x scale matrices (n, 2, 2) and shifts (n, 2) as a fraction of the extent."""
    a, s = _sym(n, math.radians(ROT), dev), 1 + _sym(n, SCALE, dev)
    rs = torch.stack([torch.stack([a.cos(), -a.sin()], -1), torch.stack([a.sin(), a.cos()], -1)], -2)
    return rs * s[:, None, None], _sym(2 * n, SHIFT, dev).view(n, 2)


def _time_mask(b, t, dev):
    keep, frames = torch.ones(b, t, device=dev), torch.arange(t, device=dev)
    masks = torch.randint(1, MASKS + 1, (b,), device=dev)
    for m in range(MASKS):
        n = torch.randint(1, MASK_LEN + 1, (b,), device=dev)
        start = (torch.rand(b, device=dev) * (t - n + 1)).long()
        keep[(frames >= start[:, None]) & (frames < (start + n)[:, None]) & (m < masks)[:, None]] = 0
    return keep


def _video(v, rgb, flip):
    b, t, dev = v.shape[0], v.shape[1], v.device
    v = (v + 1) / 2
    if rgb:
        v = v * (1 + _sym(b, LIGHT, dev)).view(b, 1, 1, 1, 1)
        grey = (v * torch.tensor(GREY, device=dev).view(1, 1, 3, 1, 1)).sum(2, keepdim=True).mean((1, 2, 3, 4), True)
        v = ((v - grey) * (1 + _sym(b, LIGHT, dev)).view(b, 1, 1, 1, 1) + grey).clamp(0, 1)
    rs, shift = _affine(b, dev)
    theta = torch.cat([rs, 2 * shift[..., None]], -1).repeat_interleave(t, 0)     # shift in [-1, 1] grid units
    flat = v.flatten(0, 1)
    flat = F.grid_sample(flat, F.affine_grid(theta, flat.shape, align_corners=False), align_corners=False)
    v = flat.view_as(v) * 2 - 1                                                   # zero padding -> black
    if flip:
        v = torch.where((torch.rand(b, device=dev) < 0.5).view(b, 1, 1, 1, 1), v.flip(-1), v)
    return v


def _coords(c):
    b, t = c.shape[:2]
    p = c.view(b, t, -1, 3)
    rs, shift = _affine(b, c.device)
    xy, mid = p[..., :2], p[..., :2].mean((1, 2), keepdim=True)
    extent = (xy[..., 0].amax((1, 2)) - xy[..., 0].amin((1, 2))).view(b, 1, 1, 1)
    xy = torch.einsum('bij,btkj->btki', rs, xy - mid) + mid + shift.view(b, 1, 1, 2) * extent
    z = p[..., 2:] * rs.det().sqrt().view(b, 1, 1, 1)
    return torch.cat([xy, z], -1).view_as(c)


def augment(x, rgb):
    """x: {'video': (B, T, 3, H, W) in [-1, 1] and/or 'coords': (B, T, F)} -> the same, augmented.
    rgb: the video is a camera image (brightness and contrast apply)."""
    ref = next(iter(x.values()))
    b, t = ref.shape[:2]
    keep = _time_mask(b, t, ref.device)
    out = {}
    if 'video' in x:
        out['video'] = _video(x['video'], rgb, flip='coords' not in x) * keep.view(b, t, 1, 1, 1)
    if 'coords' in x:
        out['coords'] = _coords(x['coords']) * keep[..., None]
    return out
