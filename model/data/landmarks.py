"""FaceMesh point sets and the landmark normalization of docs/MODELS.md ("Head pose").

Landmarks come from the word clips (clips.py): (29, 478, 3), x/y as fractions of the 256 px face crop, z scaled
by the same side. The old per-speech npz (DE-0210004073 only) uses fractions of the ROI
width/height instead. 468-477 are iris points and are dropped.

Normalization, fitted once on the training speakers (FaceNorm):
1. Rigid alignment: a similarity transform (translation, scale, 3D rotation) fitted on RIGID_IDX only. The head
   pose estimate then doesn't depend on the mouth, and jaw opening stays in the lip coordinates.
2. Pose correction: FaceMesh depth is guessed from one image, so a tilted head leaves a deformation after
   alignment (flatter lips, chin closer to the mouth) that looks like articulation. It is learned per point as a
   quadratic function of the head angles, within speakers, and subtracted: every frame is moved to the
   reference pose, i.e. the speaker's resting face rotated with the head is what counts as "no motion".
"""
import numpy as np
from scipy.spatial.transform import Rotation
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_CONTOURS, FACEMESH_LIPS

N_MESH = 468
CONTOUR_EDGES = sorted(FACEMESH_CONTOURS)
LIP_EDGES = sorted(FACEMESH_LIPS)
CONTOUR_IDX = np.array(sorted({i for e in FACEMESH_CONTOURS for i in e}))   # 128 points: V2 / V4
LIP_IDX = np.array(sorted({i for e in FACEMESH_LIPS for i in e}))           # 40 points:  V3 / V5
# Points that don't move when speaking: forehead and temples, nose bridge, eye corners, cheekbones at the ears.
RIGID_IDX = np.array([10, 151, 109, 67, 103, 54, 21, 338, 297, 332, 284, 251,
                      168, 6, 197, 195, 33, 133, 263, 362, 127, 356, 234, 454])


def to_pixels(lm, roi_px):
    """(..., >=468, 3) normalized landmarks -> (..., 468, 3) in ROI pixels; z uses the width scale like x."""
    w, h = roi_px
    return lm[..., :N_MESH, :].astype(np.float32) * np.array([w, h, w], np.float32)


def _normalize(x):
    """Move the rigid points' centroid to 0 and scale them to unit norm, for (..., 468, 3)."""
    x = x - x[..., RIGID_IDX, :].mean(-2, keepdims=True)
    return x / np.linalg.norm(x[..., RIGID_IDX, :], axis=(-2, -1), keepdims=True)


def rigid_align(seq, ref):
    """Align every frame of seq (T, 468, 3) to ref on the rigid points.

    Returns the aligned frames (T, 468, 3) and the head angles (T, 3) relative to ref in radians
    (rotation about x = pitch / nod, y = yaw / turn, z = roll / tilt)."""
    ref, x = _normalize(ref), _normalize(seq.astype(np.float64))
    u, _, vt = np.linalg.svd(np.swapaxes(x[:, RIGID_IDX], 1, 2) @ ref[RIGID_IDX])
    u[:, :, 2] *= np.sign(np.linalg.det(u @ vt))[:, None]          # no reflections
    rot = u @ vt
    return x @ rot, Rotation.from_matrix(rot).as_euler('xyz')


def pose_features(angles):
    """(T, 3) head angles -> (T, 9): linear, squared and cross terms. Zero at the reference pose."""
    p, y, r = angles.T
    return np.stack([p, y, r, p * p, y * y, r * r, p * y, p * r, y * r], 1)


class FaceNorm:
    """Reference face and pose correction, fitted on the training speakers only.

    ref: (468, 3) generalized-Procrustes mean shape (an average face, nobody's own).
    pose_coef: (9, 1404) how each point shifts with the head angles (pose_features), after alignment.
    """

    def __init__(self, ref, pose_coef):
        self.ref, self.pose_coef = ref, pose_coef

    @classmethod
    def fit(cls, seqs, speakers, iters=3, ridge=1e-6):
        """seqs: list of (T, 468, 3) clips; speakers: their speaker IDs."""
        ref = _normalize(np.mean([s.mean(0) for s in seqs], 0))
        for _ in range(iters):
            ref = _normalize(np.mean([rigid_align(s, ref)[0].mean(0) for s in seqs], 0))
        # Within-speaker regression: subtracting each speaker's means keeps face shape (identity) out of the fit.
        by = {}
        for s, spk in zip(seqs, speakers):
            by.setdefault(spk, []).append(s)
        ftf, ftd = np.zeros((9, 9)), np.zeros((9, 3 * N_MESH))
        for clips in by.values():
            al = [rigid_align(s, ref) for s in clips]
            d = np.concatenate([a.reshape(len(a), -1) for a, _ in al])
            f = pose_features(np.concatenate([g for _, g in al]))
            d, f = d - d.mean(0), f - f.mean(0)
            ftf += f.T @ f
            ftd += f.T @ d
        return cls(ref, np.linalg.solve(ftf + ridge * np.eye(9), ftd))

    def __call__(self, seq, correct_pose=True):
        """(T, 468, 3) in pixels -> (T, 468, 3) aligned (and pose-corrected) in reference units, (T, 3) angles."""
        x, angles = rigid_align(seq, self.ref)
        if correct_pose:
            x = x - (pose_features(angles) @ self.pose_coef).reshape(x.shape)
        return x.astype(np.float32), angles.astype(np.float32)

    def save(self, path):
        np.savez(path, ref=self.ref, pose_coef=self.pose_coef)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        return cls(d['ref'], d['pose_coef'])


def coord_features(seq_px, norm, idx, level='pose', speaker_mean=None):
    """Per-frame coordinate vector for V4/V5/V6: points of subset idx, flattened to (T, 3 * len(idx)).

    Normalization levels (privacy ablation): 'raw' pixels -> 'aligned' (rigid alignment to norm.ref) -> 'pose'
    (default, also pose-corrected); plus speaker_mean (len(idx), 3), the speaker's mean normalized shape ->
    speaker mean removed.
    """
    x = seq_px if level == 'raw' else norm(seq_px, correct_pose=level == 'pose')[0]
    x = x[:, idx]
    if speaker_mean is not None:
        x = x - speaker_mean
    return x.reshape(len(x), -1).astype(np.float32)
