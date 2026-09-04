"""Ablation: GLipsNet on the uncropped full-face frames (no mouth-ROI crop), to isolate preprocessing vs. architecture.

    python Transformer_based/train_15_noroi.py        -> checkpoints_15_noroi/
"""
import os

from train_15 import run_15
from dataset15 import FULL_ROOT

if __name__ == '__main__':
    run_15(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints_15_noroi'),
           root_dir=FULL_ROOT)
