"""Ablation: GLipsNet WITHOUT the multi-scale temporal conv stem (use_stem=False).

The MS-TCN stem is replaced by identity, so per-frame features go straight into
the Transformer (+ attentive pool). Everything else matches the main from-scratch
GLips15 run, isolating how much the local multi-scale temporal convolutions add on
top of the Transformer's global attention.

    python Transformer_based/train_15_nostem.py       -> checkpoints_15_nostem/
"""
import os

from train_15 import run_15

if __name__ == '__main__':
    run_15(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints_15_nostem'),
           use_stem=False)
