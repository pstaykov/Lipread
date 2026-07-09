"""Ablation: GLipsNet on the UNCROPPED full-face frames (no mouth-ROI crop).

Feeds the raw GLips clips (whole 256x256 head, resized to 96 -> 88 like every other
run) instead of the mouth-ROI crops, with the identical GLipsNet architecture and
recipe. This measures how much of the accuracy comes from the face-detection /
mouth-ROI preprocessing versus the architecture itself — i.e. how much of the gap
to Ameer et al. is preprocessing rather than model.

    python Transformer_based/train_15_noroi.py        -> checkpoints_15_noroi/
"""
import os

from train_15 import run_15
from dataset15 import FULL_ROOT

if __name__ == '__main__':
    run_15(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints_15_noroi'),
           root_dir=FULL_ROOT)
