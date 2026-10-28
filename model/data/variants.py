"""The input variants of docs/MODELS.md: which encoder each uses and how one clip becomes model input."""
import numpy as np
import torch

from . import images
from .landmarks import CONTOUR_EDGES, CONTOUR_IDX, LIP_EDGES, LIP_IDX, N_MESH, coord_features

FACE_IDX = np.arange(N_MESH)

# id: (image input or None, coordinate subset or None)
VARIANTS = {
    'V0_face':  ('rgb_face', None),          # RGB, face crop (same crop as V1/V2)
    'V0_mouth': ('rgb_mouth', None),         # RGB, mouth crop (same crop as V3, as in BWKI)
    'V1':       ('overlay_face', None),      # RGB + contour mesh at 100 % opacity
    'V2':       ('render_face', None),       # contour mesh on black, frontal
    'V3':       ('render_mouth', None),      # lip mesh on black, frontal
    'V4':       (None, CONTOUR_IDX),         # 128 contour points
    'V5':       (None, LIP_IDX),             # 40 lip points
    'V6':       ('rgb_mouth', LIP_IDX),      # V0_mouth embedding ⊕ V5 coordinates
}


def coord_dim(variant):
    idx = VARIANTS[variant][1]
    return 0 if idx is None else 3 * len(idx)


def uses_video(variant):
    return VARIANTS[variant][0] is not None


def make_inputs(variant, seq_px, norm, frames=None, size=images.ENC_SIZE):
    """One clip -> model inputs.

    seq_px: (T, 468, 3) landmarks in pixels (clips.load_landmarks); norm: landmarks.FaceNorm (training speakers);
    frames: (T, H, W, 3) uint8 RGB ROI frames, needed for rgb_* and overlay_*.
    Returns {'video': (T, 3, size, size) float in [-1, 1]} and/or {'coords': (T, F) float}.
    """
    image, idx = VARIANTS[variant]
    out = {}
    if image is not None:
        kind, region = image.split('_')
        box_idx, margin = (FACE_IDX, images.FACE_MARGIN) if region == 'face' else (LIP_IDX, images.MOUTH_MARGIN)
        if kind == 'render':
            # Frontal: aligned, pose-corrected mesh in a box fixed by the reference face, the same for every clip.
            x = norm(seq_px)[0]
            box = np.repeat(images.clip_boxes(norm.ref[None], box_idx, margin), len(x), 0)
            clip = images.render_clip(x, CONTOUR_EDGES if region == 'face' else LIP_EDGES, box, size)
        else:
            box = images.clip_boxes(seq_px, box_idx, margin)
            clip = (images.rgb_clip(frames, box, size) if kind == 'rgb'
                    else images.overlay_clip(frames, seq_px, CONTOUR_EDGES, box, size))
        out['video'] = torch.from_numpy(clip).permute(0, 3, 1, 2).float() / 127.5 - 1   # Keras 'tf' scaling
    if idx is not None:
        out['coords'] = torch.from_numpy(coord_features(seq_px, norm, idx))
    return out
