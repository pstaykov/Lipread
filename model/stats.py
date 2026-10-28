"""Test results with speaker-level uncertainty (docs/MODELS.md, section 5).

    python -m model.stats V4 V5 V0_mouth

Reads test_pred.npy of every seed of a variant (runs/<variant>, runs/<variant>_s<seed>; written by model.train).
Clips of one speaker are not independent, so the 95 % intervals come from a bootstrap over the test speakers: whole
speakers are resampled with replacement (2,000 times) and each resample is scored as the mean over the seeds.
Writes runs/<variant>/stats.json.
"""
import json
import sys

import numpy as np
from sklearn.metrics import f1_score

from .data import clips
from .train import RUNS

BOOT = 2000


def scores(pred, y):
    return np.array([(pred[:, 0] == y).mean(), (pred == y[:, None]).any(1).mean(),
                     f1_score(y, pred[:, 0], average='macro')])


def main():
    items = clips.read_split('test')
    word_ix = {w: i for i, w in enumerate(clips.words())}
    y = np.array([word_ix[c['word']] for c in items])
    _, spk = np.unique([c['speaker'] for c in items], return_inverse=True)
    members = [np.flatnonzero(spk == k) for k in range(spk.max() + 1)]
    rng = np.random.default_rng(0)
    draws = [np.concatenate([members[k] for k in rng.integers(0, len(members), len(members))]) for _ in range(BOOT)]
    for variant in sys.argv[1:]:
        runs = sorted(d for d in [RUNS / variant, *RUNS.glob(f'{variant}_s*')] if (d / 'test_pred.npy').exists())
        preds = [np.load(d / 'test_pred.npy').astype(np.int64) for d in runs]
        per_seed = np.array([scores(p, y) for p in preds])
        boot = np.array([np.mean([scores(p[j], y[j]) for p in preds], 0) for j in draws])
        lo, hi = np.percentile(boot, [2.5, 97.5], 0)
        res = {'variant': variant, 'seeds': [d.name for d in runs], 'speakers': len(members)}
        for i, k in enumerate(('top1', 'top5', 'f1')):
            res[k] = {'mean': per_seed[:, i].mean(), 'seed_std': per_seed[:, i].std(ddof=1) if len(preds) > 1 else None,
                      'ci95': [lo[i], hi[i]]}
            print(f'{variant} {k}: {res[k]["mean"]:.4f} [{lo[i]:.4f}, {hi[i]:.4f}], {len(preds)} seed(s)')
        (RUNS / variant / 'stats.json').write_text(json.dumps(res, indent=1, default=float))


if __name__ == '__main__':
    main()
