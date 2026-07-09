"""Ablation: GLipsNet with plain temporal MEAN pooling instead of attentive pooling.

Everything else (classes, stock split, MS-TCN stem, Transformer, recipe, early
stopping) is identical to the main from-scratch GLips15 run, so the delta isolates
the contribution of the attentive-pool head. Run as its own experiment rather than
bolted on afterwards.

    python Transformer_based/train_15_meanpool.py     -> checkpoints_15_meanpool/
"""
import os

from train_15 import run_15

if __name__ == '__main__':
    run_15(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints_15_meanpool'),
           pool='mean')
