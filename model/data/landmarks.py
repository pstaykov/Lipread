"""FaceMesh point sets and the landmark normalization of docs/MODELS.md.

Landmarks come from the word clips (clips.py): (29, 478, 3), x/y as fractions of the 256 px face crop, z scaled
by the same side. 468-477 are iris points and are dropped.

Normalization (FaceNorm, fitted once on the training speakers): every frame is Procrustes-aligned (translation,
scale, 3D rotation) to the generalized-Procrustes mean shape of the training speakers, fitted on all 468 points.
Fitting on the lips alone would scale away mouth opening.
"""
import numpy as np
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_CONTOURS, FACEMESH_LIPS

N_MESH = 468
CONTOUR_EDGES = sorted(FACEMESH_CONTOURS)
LIP_EDGES = sorted(FACEMESH_LIPS)
CONTOUR_IDX = np.array(sorted({i for e in FACEMESH_CONTOURS for i in e}))   # 128 points: V2 / V4
LIP_IDX = np.array(sorted({i for e in FACEMESH_LIPS for i in e}))           # 40 points:  V3 / V5


def to_pixels(lm, roi_px):
    """(..., >=468, 3) normalized landmarks -> (..., 468, 3) in ROI pixels; z uses the width scale like x."""
    w, h = roi_px
    return lm[..., :N_MESH, :].astype(np.float32) * np.array([w, h, w], np.float32)


def _normalize(x):
    """Move the centroid to 0 and scale to unit norm, for (..., 468, 3)."""
    x = x - x.mean(-2, keepdims=True)
    return x / np.linalg.norm(x, axis=(-2, -1), keepdims=True)


def procrustes_align(seq, ref):
    """Align every frame of seq (T, 468, 3) to ref (468, 3) by Procrustes on all points -> (T, 468, 3)."""
    ref, x = _normalize(ref), _normalize(seq.astype(np.float64))
    u, _, vt = np.linalg.svd(np.swapaxes(x, 1, 2) @ ref)
    u[:, :, 2] *= np.sign(np.linalg.det(u @ vt))[:, None]          # no reflections
    return x @ (u @ vt)


class FaceNorm:
    """ref: (468, 3) generalized-Procrustes mean shape of the training speakers (an average face, nobody's own)."""

    def __init__(self, ref):
        self.ref = ref

    @classmethod
    def fit(cls, seqs, iters=3):
        """seqs: list of (T, 468, 3) training clips."""
        ref = _normalize(np.mean([s.mean(0) for s in seqs], 0))
        for _ in range(iters):
            ref = _normalize(np.mean([procrustes_align(s, ref).mean(0) for s in seqs], 0))
        return cls(ref)

    def __call__(self, seq):
        """(T, 468, 3) in pixels -> (T, 468, 3) aligned, in reference units."""
        return procrustes_align(seq, self.ref).astype(np.float32)

    def save(self, path):
        np.savez(path, ref=self.ref)

    @classmethod
    def load(cls, path):
        return cls(np.load(path)['ref'])


def coord_features(seq_px, norm, idx, level='aligned', speaker_mean=None):
    """Per-frame coordinate vector for V4/V5/V6: points of subset idx, flattened to (T, 3 * len(idx)).

    Normalization levels (privacy ablation): 'raw' pixels -> 'aligned' (default, Procrustes to norm.ref); plus
    speaker_mean (len(idx), 3), the speaker's mean aligned shape -> speaker mean removed.
    """
    x = seq_px if level == 'raw' else norm(seq_px)
    x = x[:, idx]
    if speaker_mean is not None:
        x = x - speaker_mean
    return x.reshape(len(x), -1).astype(np.float32)
