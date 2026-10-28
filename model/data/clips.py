"""BTLips word clips (dataset/README.md) as input for variants.make_inputs.

BTLips/lipread_files/<word>/<split>/<word>_<speaker>-<instance>.mp4: 256 × 256 face crop, 25 fps, 29 frames with
the word centred. BTLips_facemesh/<split>.npy: FaceMesh landmarks (N, 29, 478, 3) float16 in the order of
<split>.txt, x/y as fractions of the crop, z scaled by the same side, NaN where no face was found (<= 2 frames).
The split is speaker-disjoint and given by the folders.
"""
from pathlib import Path

import cv2
import numpy as np

from .landmarks import N_MESH

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / 'BTLips'
FACEMESH = REPO / 'BTLips_facemesh'
CROP = 256
SPLITS = ('train', 'val', 'test')


def words(root=ROOT):
    return sorted(p.name for p in (root / 'lipread_files').iterdir())


def read_split(split, fm=FACEMESH):
    """Clips of one split in facemesh order: dicts with name ('<word>/<split>/<file stem>'), word, speaker (int)."""
    out = []
    for line in (fm / f'{split}.txt').read_text(encoding='utf-8').split():
        word, _, stem = line.split('/')
        out.append({'name': line, 'word': word, 'speaker': int(stem.rsplit('_', 1)[1].split('-')[0])})
    return out


def facemesh(split, fm=FACEMESH):
    """(N, 29, 478, 3) float16 memory map."""
    return np.load(fm / f'{split}.npy', mmap_mode='r')


def fill_missing(lm):
    """Replace NaN frames (no face found, at most 2 per clip) by linear interpolation of the neighbouring frames."""
    bad = np.isnan(lm).any(axis=(1, 2))
    if bad.any():
        t = np.arange(len(lm))
        flat = lm.reshape(len(lm), -1)
        for j in range(flat.shape[1]):
            flat[bad, j] = np.interp(t[bad], t[~bad], flat[~bad, j])
    return lm


def to_pixels(lm):
    """One clip's facemesh rows -> (29, 468, 3) float32 in crop pixels, the seq_px of variants.make_inputs."""
    return fill_missing(np.array(lm[:, :N_MESH], np.float32)) * CROP


def load_frames(name, root=ROOT):
    """(29, 256, 256, 3) uint8 RGB frames."""
    cap = cv2.VideoCapture(str(root / 'lipread_files' / f'{name}.mp4'))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames)
