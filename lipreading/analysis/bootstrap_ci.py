"""Bootstrap confidence interval for a 0/1 per-sample correctness array.

No retraining needed: resamples the test set with replacement B times and
reports the percentile interval of the resulting accuracy distribution. Used to
put an uncertainty band on single-seed test-set accuracy numbers.
"""
import numpy as np


def bootstrap_ci(correct, n_boot=10000, alpha=0.05, seed=0):
    """correct: 1D array-like of 0/1 (or bool). Returns (mean, lo, hi, half_width)."""
    correct = np.asarray(correct, dtype=np.float64)
    n = correct.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = correct[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    mean = correct.mean()
    return mean, lo, hi, (hi - lo) / 2
