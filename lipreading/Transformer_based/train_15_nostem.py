"""Ablation: GLipsNet without the multi-scale temporal conv stem (use_stem=False), isolating its contribution over the Transformer alone.

    python Transformer_based/train_15_nostem.py       -> checkpoints_15_nostem/
"""
import os

from train_15 import run_15

if __name__ == '__main__':
    run_15(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints_15_nostem'),
           use_stem=False)
