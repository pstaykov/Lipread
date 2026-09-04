"""Ablation: GLipsNet with plain temporal mean pooling instead of attentive pooling; isolates the pool head's contribution.

    python Transformer_based/train_15_meanpool.py     -> checkpoints_15_meanpool/
"""
import os

from train_15 import run_15

if __name__ == '__main__':
    run_15(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints_15_meanpool'),
           pool='mean')
