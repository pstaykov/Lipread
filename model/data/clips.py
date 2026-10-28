"""Bundestag word clips (dataset/mine_clips.py, docs/MINING.md "Clip mining") as input for variants.make_inputs.

Each clip is bundestag_data/clips/<word>/<speechID>_<frame>.{mp4,npz}: a 256 × 256 face crop, 25 fps, 29 frames
with the word centred, and FaceMesh landmarks (29, 478, 3) with x/y as fractions of the crop and z scaled by
the same side. clips.jsonl holds one line per clip with word and speaker_id.
"""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .landmarks import N_MESH

ROOT = Path(__file__).resolve().parents[2] / 'bundestag_data' / 'clips'
CROP = 256
SPLIT = (0.8, 0.1, 0.1)    # train / val / test fractions of the speakers


def read_index(root=ROOT):
    with open(root / 'clips.jsonl', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def split_of(speaker_id):
    """'train' / 'val' / 'test', fixed per speaker (hash of the Wikidata ID), so the split is speaker-disjoint
    and stays the same while the mine grows."""
    u = int(hashlib.md5(speaker_id.encode()).hexdigest(), 16) % 1000 / 1000
    return 'train' if u < SPLIT[0] else 'val' if u < SPLIT[0] + SPLIT[1] else 'test'


def fill_missing(lm):
    """Replace NaN frames (no face found, at most 2 per clip) by linear interpolation of the neighbouring frames."""
    bad = np.isnan(lm).any(axis=(1, 2))
    if bad.any():
        t = np.arange(len(lm))
        flat = lm.reshape(len(lm), -1)
        for j in range(flat.shape[1]):
            flat[bad, j] = np.interp(t[bad], t[~bad], flat[~bad, j])
    return lm


def load_landmarks(entry, root=ROOT):
    """(29, 468, 3) float32 landmarks in crop pixels, the seq_px of variants.make_inputs."""
    lm = np.load(root / f"{entry['path']}.npz")['landmarks'][:, :N_MESH].astype(np.float32)
    return fill_missing(lm) * CROP


def load_frames(entry, root=ROOT):
    """(29, 256, 256, 3) uint8 RGB frames."""
    cap = cv2.VideoCapture(str(root / f"{entry['path']}.mp4"))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames)
